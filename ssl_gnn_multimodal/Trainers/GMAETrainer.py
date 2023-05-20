
import os
import torch
import numpy as np
from torch_geometric.nn import MLPAggregation

from Trainers import MMGNNTrainer
from Models.Encoder import ImageEncoder,TextEncoder,ProjectionHead
from Models.GMAE import GMAE
from Models.GAT import DeepGAT
from Models.MLPClassifier import MLPClassifier

from utils import evaluate_graph_embeddings_using_svm,graph_emb_pooling
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score

class GMAETrainer(MMGNNTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)
        if self.pretrain is not True: # finetuning 1/10 of pretrain lr 
            self.lr = self.lr*1e-1
        print("Trainable Models:",self.trainable_models)

    def build_model(self):
        GNN_OUT_CHANNELS = self.config['gnn_out_channels']
        PROJECTION_DIM = self.config['projection_dim']
        # Model
        print('==> Building model..')
        self.models = {
            'image_encoder': ImageEncoder(trainable=self.pretrain).to(self.device),
            'text_encoder': TextEncoder(trainable=self.pretrain).to(self.device),
            'image_projection': ProjectionHead(2048,PROJECTION_DIM,trainable=self.pretrain).to(self.device),
            'text_projection': ProjectionHead(768,PROJECTION_DIM,trainable=self.pretrain).to(self.device)
        }
        graph_encoder = DeepGAT(in_channels=PROJECTION_DIM,hidden_channels=2*GNN_OUT_CHANNELS,out_channels=GNN_OUT_CHANNELS,num_layers=3,nheads=4,last_layer=True,jk=self.config.get('jk'),norm_type="graph_norm",activation_type="prelu",dropout=0.3).to(self.device)
        graph_decoder = DeepGAT(in_channels=GNN_OUT_CHANNELS,hidden_channels=GNN_OUT_CHANNELS,out_channels=PROJECTION_DIM,num_layers=1,nheads=4,last_layer=False,norm_type="graph_norm",activation_type="prelu",dropout=0.3).to(self.device)
        self.models['graph'] = GMAE(graph_encoder,graph_decoder).to(self.device)
        self.trainable_models = ['image_encoder','text_encoder','image_projection','text_projection','graph']
        
        if self.pretrain is not True:
            max_num_nodes_in_graph = 22
            readout_aggregation = MLPAggregation(GNN_OUT_CHANNELS,2*GNN_OUT_CHANNELS,max_num_nodes_in_graph,num_layers=1)
            self.models['classifier'] = MLPClassifier(2*GNN_OUT_CHANNELS,1, 2,readout_aggregation, True,0.5).to(self.device)
            self.trainable_models = ['graph','classifier']

    def train_epoch(self,epoch):
        self.setTrain(self.trainable_models)
        train_loss = 0
        total = 0
        
        for images, image_features, tokenized_text, attention_masks, labels in self.train_loader:
            images, image_features, tokenized_text, attention_masks, labels = images.to(self.device), image_features.to(self.device), tokenized_text.to(self.device), attention_masks.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            text_embeddings = self.models['text_projection'](self.models['text_encoder'](input_ids=tokenized_text, attention_mask=attention_masks))
            image_feat_data = self.get_image_feature_embeddings_v2(image_features)
            image_embeddings = self.models['image_projection'](self.models['image_encoder'](images))
            g_data = self.generate_subgraph_v2(image_embeddings,image_feat_data,text_embeddings,labels)
                   
            if self.pretrain is True:
                recon_loss,encoder_loss = self.models['graph'](g_data)
                loss = recon_loss.sum()
                if encoder_loss is not None:
                    loss = self.config.get('recon_loss_coef',1.0)*loss + self.config.get('encoder_loss_coef',4.0)*(encoder_loss.sum())
            else:
                # hateful classification
                y = g_data.y.to(self.device)
                enc_rep = self.models['graph'].embed(g_data)
                output = self.models['classifier'](enc_rep,g_data)
                loss = self.models['classifier'].criterion(output, y)
                
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
                g_data = self.generate_subgraph_v2(image_embeddings,image_feat_data,text_embeddings,labels)
                
                if self.pretrain is True: #pretraining
                    # graph_emb = graph_emb_pooling("mean",enc_rep,g_data)
                    recon_loss,encoder_loss = self.models['graph'](g_data)
                    loss = recon_loss.sum()
                    if encoder_loss is not None:
                        loss = self.config.get('recon_loss_coef',1.0)*loss + self.config.get('encoder_loss_coef',4.0)*(encoder_loss.sum())

                else: # hateful classification
                    y = g_data.y.to(self.device)
                    enc_rep = self.models['graph'].embed(g_data)
                    output = self.models['classifier'](enc_rep,g_data)
                    loss = self.criterion(output, y)
                    if out_label_ids is None:
                        out_label_ids = labels.detach().cpu().numpy()
                    else:
                        out_label_ids = np.append(out_label_ids,labels.detach().cpu().numpy())

                    if proba is None:
                        proba = torch.sigmoid(output).detach().cpu().numpy()
                    else:
                        proba = np.append(proba, torch.sigmoid(output).detach().cpu().numpy(), axis=0)

                test_loss += loss.item()
                total += labels.size(0)
        
        metrics = {
            "loss": test_loss/total
        }
        if self.pretrain is not True:
            preds = (proba > 0.5).astype(proba.dtype)
            evaluate_metrics = {
                "auc": round(roc_auc_score(out_label_ids,proba),3),
                "avg_precision": round(average_precision_score(out_label_ids,proba),3),
                "accuracy":round(accuracy_score(out_label_ids, preds),3),
                "micro_f1":round(f1_score(out_label_ids, preds, average="micro"),3)
            }
            metrics = {**metrics,**evaluate_metrics}

        print("Evaluation --- Epoch : {}".format(epoch),str(metrics))  
        # if self.pretrain is True:
        #     graph_emb_metrics = evaluate_graph_embeddings_using_svm(graph_embs,out_label_ids)  
        #     print("Graph Embedding SVC Metrics:", str(graph_emb_metrics))

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