import os
import gc
from Models.DeepVGAE import DeepVGAE,GCNVGAEEncoder,GATVGAEEncoder
from Trainers import MMGNNTrainer
from Models.MLPClassifier import MLPClassifier

import torch
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import MLP, MLPAggregation,SetTransformerAggregation,DeepSetsAggregation,GRUAggregation
from sklearn.metrics import average_precision_score, roc_auc_score,accuracy_score,f1_score
import numpy as np

import torch_geometric.transforms as T
from torch_geometric.data import Data as GraphData
from torch_geometric.loader import DataLoader as GDataLoader

class VGAETrainer(MMGNNTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)
        print("Trainable Models",self.trainable_models)

    def build_model(self):
        GNN_OUT_CHANNELS = self.config['gnn_out_channels']
        PROJECTION_DIM = self.config['projection_dim']
        super().build_model()
        self.trainable_models = ['image_encoder','text_encoder','image_projection','text_projection','graph']
        self.models['gnn_encoder'] = GATVGAEEncoder(PROJECTION_DIM,2*GNN_OUT_CHANNELS,GNN_OUT_CHANNELS,4,0.3)
        self.models['graph'] = DeepVGAE(self.models['gnn_encoder']).to(self.device)
        if self.pretrain is not True:
            max_num_nodes_in_graph = 22
            self.models['readout_aggregation'] = MLPAggregation(GNN_OUT_CHANNELS,2*GNN_OUT_CHANNELS,max_num_nodes_in_graph,num_layers=1)
            self.models['classifier'] = MLPClassifier(2*GNN_OUT_CHANNELS,1, 2,self.models['readout_aggregation'], True,0.5).to(self.device)
            self.trainable_models = ['graph','classifier']

    

    def train_epoch(self,epoch):
        self.setTrain(self.trainable_models)
        train_loss = 0
        total = 0
        for images, image_features, tokenized_text, attention_masks, labels in self.train_loader:
            images, image_features, tokenized_text, attention_masks, labels = images.to(self.device), image_features.to(self.device), tokenized_text.to(self.device), attention_masks.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            # images, image_features, text_features,labels
            text_embeddings = self.models['text_projection'](self.models['text_encoder'](input_ids=tokenized_text, attention_mask=attention_masks))
            image_feat_data = self.get_image_feature_embeddings_v2(image_features)
            image_embeddings = self.models['image_projection'](self.models['image_encoder'](images))
            g_data_loader = self.generate_subgraph_v2(image_embeddings,image_feat_data,text_embeddings,labels)
            
            g_data = next(iter(g_data_loader))
            g_data = g_data.to(self.device)
            
            z = self.models['graph'].encode(g_data.x, g_data.edge_index)
            print(z.shape)
            if self.pretrain is True:
            # Pretraining  
                loss = self.models['graph'].loss(z, g_data)
            else:
            # hateful classification
                output = self.models['classifier'](z,g_data)
                loss = self.criterion(output, g_data.y)
            
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            total += labels.size(0)

        
        print("Training --- Epoch : {} | Loss : {}".format(epoch,train_loss/total))    
        return epoch,train_loss/total
    
    def evaluate(self, epoch, data_loader):
        self.setEval()
        test_loss = 0
        total = 0
        preds = None
        proba = None
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

                z = self.models['graph'].encode(g_data.x, g_data.edge_index)
                
                if self.pretrain is True:
                # Pretraining  
                    loss = self.models['graph'].loss(z, g_data)
                    
                    neg_edge_index = negative_sampling(g_data.edge_index, z.size(0))
        
                    pos_y = z.new_ones(g_data.edge_index.size(1))
                    neg_y = z.new_zeros(neg_edge_index.size(1))
                    y = torch.cat([pos_y, neg_y], dim=0)

                    pos_pred = self.models['graph'].decoder(z, g_data.edge_index, sigmoid=True)
                    neg_pred = self.models['graph'].decoder(z, neg_edge_index, sigmoid=True)
                    pred = torch.cat([pos_pred, neg_pred], dim=0)
                    if proba is None:
                        proba = pred.detach().cpu().numpy() 
                    else:
                        proba = np.append(proba,pred.detach().cpu().numpy(),axis=0)
                    
                    if preds is None:
                        preds = pred.detach().cpu().numpy() > 0.5
                    else:
                        preds = np.append(preds,pred.detach().cpu().numpy()> 0.5,axis=0)
                        
                    if out_label_ids is None:
                        out_label_ids = y.detach().cpu().numpy()
                    else:
                        out_label_ids = np.append(out_label_ids,y.detach().cpu().numpy())
                    
                else:
                # hateful classification
                    output = self.models['classifier'](z,g_data)
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
        
        metrics = {
            "loss": test_loss/total,
            "auc": round(roc_auc_score(out_label_ids,proba),3),
            "avg_precision": round(average_precision_score(out_label_ids,proba),3),
            "accuracy":round(accuracy_score(out_label_ids, preds),3),
            "micro_f1":round(f1_score(out_label_ids, preds, average="micro"),3)
        }
        print("Evaluation --- Epoch : {} | roc_auc_score : {} | average_precision_score : {} | accuracy : {} | micro_f1 : {}".format(epoch,metrics['auc'],metrics['avg_precision'],metrics['accuracy'],metrics['micro_f1']))    
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
                for name in self.models:
                    savePath = os.path.join(outpath, "{}.pth".format(name))
                    toSave = self.models[name].state_dict()
                    torch.save(toSave, savePath)
                savePath = os.path.join(outpath, "{}.pth".format(self.optim.lower()))
                torch.save(self.optimizer.state_dict(), savePath)
            
                print("best metrics:", str(metrics))

        except Exception as e:
            print("Error:",e)
