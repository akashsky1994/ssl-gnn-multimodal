
import os
import torch
import numpy as np
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import MLPAggregation

from Trainers import MMGNNTrainer
from Models.Encoder import ImageEncoder,TextEncoder,ProjectionHead
from Models.GMAE import GMAE
from Models.GAT import DeepGAT
from Models.MLPClassifier import MLPClassifier

from config import PROJECTION_DIM,GNN_OUT_CHANNELS

from utils import evaluate_graph_embeddings_using_svm

class GMAETrainer(MMGNNTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)
        print("Trainable Models",self.trainable_models)

    def build_model(self):
        # Model
        print('==> Building model..')
        self.models = {
            'image_encoder': ImageEncoder().to(self.device),
            'text_encoder': TextEncoder().to(self.device),
            'image_projection': ProjectionHead(2048,PROJECTION_DIM).to(self.device),
            'text_projection': ProjectionHead(768,PROJECTION_DIM).to(self.device),
            'graph_encoder': DeepGAT(in_channels=PROJECTION_DIM,hidden_channels=512,out_channels=1024,num_layers=3,in_heads=4,out_heads=1).to(self.device),
            'graph_decoder':DeepGAT(in_channels=1024,hidden_channels=512,out_channels=PROJECTION_DIM,num_layers=1,in_heads=1,out_heads=1).to(self.device)
        }
        self.models['graph'] = GMAE(self.models['graph_encoder'],self.models['graph_decoder']).to(self.device)
        self.trainable_models = ['image_encoder','text_encoder','image_projection','text_projection','graph']
        
        if self.pretrain is not True:
            max_num_nodes_in_graph = 22
            self.models['readout_aggregation'] = MLPAggregation(GNN_OUT_CHANNELS,2*GNN_OUT_CHANNELS,max_num_nodes_in_graph,num_layers=1)
            self.models['classifier'] = MLPClassifier(2*GNN_OUT_CHANNELS,1, 2,self.models['readout_aggregation'], True,0.5).to(self.device)
            self.trainable_models = ['graph','classifier']

    def train_epoch(self,epoch):
        self.setTrain()
        train_loss = 0
        total = 0
        for images, image_features, tokenized_text, attention_masks, labels in self.train_loader:
            images, image_features, tokenized_text, attention_masks, labels = images.to(self.device), image_features.to(self.device), tokenized_text.to(self.device), attention_masks.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            text_embeddings = self.models['text_projection'](self.models['text_encoder'](input_ids=tokenized_text, attention_mask=attention_masks))
            image_feat_data = self.get_image_feature_embeddings_v2(image_features)
            image_embeddings = self.models['image_projection'](self.models['image_encoder'](images))
            g_data_loader = self.generate_subgraph_v2(image_embeddings,image_feat_data,text_embeddings,labels)
            
            g_data = next(iter(g_data_loader))
            g_data = g_data.to(self.device)
            
            loss,_ = self.models['graph'](g_data.x,g_data.edge_index)
            
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            total += labels.size(0)
            
        
        print("Training --- Epoch : {} | Loss : {}".format(epoch,train_loss/total))
        return epoch,train_loss/total

    def evaluate(self, epoch, data_type, data_loader):
        self.setEval()
        test_loss = 0
        total = 0
        preds = None
        out_label_ids = None
        with torch.no_grad():
            for images, image_features, tokenized_text, attention_masks, labels in data_loader:
                images, image_features, tokenized_text, attention_masks, labels = images.to(self.device), image_features.to(self.device), tokenized_text.to(self.device), attention_masks.to(self.device), labels.to(self.device)
            
                text_embeddings = self.models['text_projection'](self.models['text_encoder'](input_ids=tokenized_text, attention_mask=attention_masks))
                image_feat_data = self.get_image_feature_embeddings_v2(image_features)
                image_embeddings = self.models['image_projection'](self.models['image_encoder'](images))
                g_data_loader = self.generate_subgraph_v2(image_embeddings,image_feat_data,text_embeddings,labels)
                
                g_data = next(iter(g_data_loader))
                g_data = g_data.to(self.device)
                
                if self.pretrain is True:
                    loss,_ = self.models['graph'](g_data.x,g_data.edge_index)
                    enc_rep = self.models['graph'].encoder(g_data.x, g_data.edge_index)
                    pooler = "mean"
                    if pooler == "mean":
                        graph_emb = global_mean_pool(enc_rep, g_data.batch)
                    elif pooler == "max":
                        graph_emb = global_max_pool(enc_rep, g_data.batch)
                    elif pooler == "sum":
                        graph_emb = global_add_pool(enc_rep, g_data.batch)
                    else:
                        raise NotImplementedError
                    
                    if preds is None:
                        preds = graph_emb.detach().cpu().numpy()
                    else:
                        preds = np.append(preds,graph_emb.detach().cpu().numpy(),axis=0)
                        
                    if out_label_ids is None:
                        out_label_ids = labels.detach().cpu().numpy()
                    else:
                        out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

                else:
                # hateful classification
                    enc_rep = self.models['graph'].encoder(g_data.x, g_data.edge_index)
                    output = self.models['classifier'](enc_rep,g_data)
                    loss = self.criterion(output, g_data.y)

                    # Metrics Calculation
                    if preds is None:
                        preds = torch.sigmoid(output).detach().cpu().numpy() > 0.5
                    else:
                        preds = np.append(preds, torch.sigmoid(output).detach().cpu().numpy() > 0.5, axis=0)
                    if proba is None:
                        proba = torch.sigmoid(output).detach().cpu().numpy()
                    else:
                        proba = np.append(proba, torch.sigmoid(output).detach().cpu().numpy(), axis=0)
                    
                    if out_label_ids is None:
                        out_label_ids = labels.detach().cpu().numpy()
                    else:
                        out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
                
                test_loss += loss.item()
                total += labels.size(0)

        metrics = evaluate_graph_embeddings_using_svm(preds,out_label_ids)  
        metrics["loss"] = test_loss/total

        print("{} --- Epoch : {}".format(data_type,epoch),str(metrics))  

        return metrics   


    def save_checkpoint(self,epoch, metrics):
        outpath = None
        try:
            if self.pretrain:
                training_type = "pretrain"
                if metrics['loss']<self.best_loss:
                    outpath = os.path.join('./checkpoints',self.model_name, "{}_{}".format(training_type,metrics['loss']))
                    self.best_loss = metrics['loss']
            else:
                training_type = "classifier"
                if metrics['auc'] > self.best_auc:
                    outpath = os.path.join('./checkpoints',self.model_name, "{}_{}_{}".format(training_type,metrics['auc'],metrics['avg_precision']))
                    self.best_auc = metrics['auc']

            if outpath:
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                print('Saving..')
                for name in self.trainable_models:
                    savePath = os.path.join(outpath, "{}.pth".format(name))
                    toSave = self.models[name].state_dict()
                    torch.save(toSave, savePath)
                savePath = os.path.join(outpath, "{}.pth".format(self.optim.lower()))
                torch.save(self.optimizer.state_dict(), savePath)
            
                print("best metrics:", str(metrics))

        except Exception as e:
            print("Error:",e)