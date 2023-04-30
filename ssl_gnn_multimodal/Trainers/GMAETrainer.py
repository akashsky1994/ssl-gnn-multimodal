
import os
import torch
import torchvision
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling

from Trainers import MMGNNTrainer
from Models.Encoder import ImageEncoder,TextEncoder,ProjectionHead
from Models.GMAE import GMAE
from Models.GAT import GAT
from config import PROJECTION_DIM
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
            'graph_encoder': GAT(in_channels=PROJECTION_DIM,hidden_channels=512,out_channels=1024,num_layers=3,in_heads=4,out_heads=1).to(self.device),
            'graph_decoder':GAT(in_channels=1024,hidden_channels=512,out_channels=PROJECTION_DIM,num_layers=1,in_heads=1,out_heads=1).to(self.device)
        }
        self.models['graph'] = GMAE(self.models['graph_encoder'],self.models['graph_decoder']).to(self.device)
        self.trainable_models = ['image_encoder','text_encoder','image_projection','text_projection','graph']
        
        if self.pretrain is not True:
            raise NotImplementedError("Classifier not implemented") #TODO

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
            
            if self.pretrain is True:
                loss,_ = self.models['graph'](g_data.x,g_data.edge_index)
            else:
                raise NotImplementedError("Not implemented") #TODO
            
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
        proba = None
        out_label_ids = None
        with torch.no_grad():
            for images, image_features, tokenized_text, attention_masks, labels in data_loader:
                images, image_features, tokenized_text, attention_masks, labels = images.to(self.device), image_features.to(self.device), tokenized_text.to(self.device), attention_masks.to(self.device), labels.to(self.device)
            
                text_embeddings = self.models['text_projection'](self.models['text_encoder'](input_ids=tokenized_text, attention_mask=attention_masks))
                image_feat_embeddings = self.get_image_feature_embeddings(images)
                image_embeddings = self.models['image_projection'](self.models['image_encoder'](images))
                g_data_loader = self.generate_subgraph_v2(image_embeddings,image_feat_embeddings,text_embeddings,labels)
                
                g_data = next(iter(g_data_loader))
                g_data = g_data.to(self.device)
                
                
                
                if self.pretrain is True:
                    loss,_ = self.models['graph'](g_data.x,g_data.edge_index)
                    z = self.models['graph'].encoder(g_data.x, g_data.edge_index)
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
                    raise NotImplementedError("Not implemented") #TODO

                
                test_loss += loss.item()
                total += labels.size(0)

        metrics = {
            "loss": test_loss/total,
            "auc": round(roc_auc_score(out_label_ids,proba),3),
            "avg_precision": round(average_precision_score(out_label_ids,proba),3),
            "accuracy":round(accuracy_score(out_label_ids, preds),3),
            "micro_f1":round(f1_score(out_label_ids, preds, average="micro"),3)
        }
        print("{} --- Epoch : {} | roc_auc_score : {} | average_precision_score : {} | accuracy : {} | micro_f1 : {}".format(data_type,epoch,metrics['auc'],metrics['avg_precision'],metrics['accuracy'],metrics['micro_f1']))    
        return metrics   
    
    def save_checkpoint(self,epoch, metrics):
        training_type = "classifier"
        if self.pretrain:
            training_type = "pretrain"
        try:
            if metrics['auc'] > self.best_auc:
                outpath = os.path.join('./checkpoints',self.model_name, "{}_{}_{}".format(training_type,metrics['auc'],metrics['avg_precision']))
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                
                    print('Saving..')
                    for name in self.trainable_models:
                        savePath = os.path.join(outpath, "{}.pth".format(name))
                        toSave = self.models[name].state_dict()
                        torch.save(toSave, savePath)
                    savePath = os.path.join(outpath, "{}.pth".format(self.optim.lower()))
                    torch.save(self.optimizer.state_dict(), savePath)
                    # self.best_acc = metrics['accuracy']
                    self.best_auc = metrics['auc']
                    print("best auc:", metrics['auc'],"avg_precision:",metrics['avg_precision'])
        except Exception as e:
            print("Error:",e)