
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import os
import time
from tqdm import tqdm
from Dataset import load_dataset
from torch.utils.data import DataLoader
from utils import get_device,get_tokenizer

class BaseTrainer():
    def __init__(self,config) -> None:
        self.config = config
        self.model_name = config['model_name']
        self.lr = config['lr']
        self.optim = config['optimizer']
        self.num_workers = config['workers']
        self.epochs = config['epochs']
        self.batch_size = config['batchsize']
        self.dataset_name = config['dataset']
        self.data_path = config['datapath']
        self.pretrain = config['pretrain']
        self.resume = config.get('resume')
        self.best_acc = 0
        self.best_auc = 0
        self.best_loss = float("inf")
        self.trainable_models = []
        self.set_device()
        
    def getTrainableParams(self):
        self.totalTrainableParams = 0
        self.trainableParameters = []
        for key in self.trainable_models:
            self.trainableParameters += list(self.models[key].parameters())
            self.totalTrainableParams += sum(p.numel() for p in self.models[key].parameters() if p.requires_grad)    

    def set_device(self):
        self.n_gpus = 1
        if bool(self.config['cpu']) is not False:
            self.device = 'cpu'
        else:
            self.device , self.n_gpus = get_device()


    def load_dataset(self):
        tokenizer = get_tokenizer(self.config.get('tokenizer','distilbert'))
        train_dataset,dev_dataset,test_dataset,collate_fn = load_dataset(self.dataset_name,self.data_path,tokenizer=tokenizer)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=True, num_workers=self.num_workers,collate_fn=collate_fn)

        self.dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=False, num_workers=self.num_workers,collate_fn=collate_fn)

        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=False, num_workers=self.num_workers,collate_fn=collate_fn)


    def setup_optimizer_losses(self):
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()
        if self.optim=='SGD':
            self.optimizer = optim.SGD(self.trainableParameters, lr=self.lr,momentum=0.9, weight_decay=5e-4)
        elif self.optim=='SGDN':
            self.optimizer = optim.SGD(self.trainableParameters, lr=self.lr,momentum=0.9, weight_decay=5e-4,nesterov=True)
        else:
            self.optimizer = eval("optim."+self.optim)(self.trainableParameters, lr=self.lr, weight_decay=float(self.config['optim_weight_decay']))
        print("Optimizer:",self.optimizer) 
        if self.config['lr_scheduler']=="CosineLRDecay":
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / self.epochs) ) * 0.5
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=scheduler)
        elif self.config['lr_scheduler']=="CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)


    def setTrain(self,model_keys=[]):
        for key in model_keys:
            self.models[key].train()

    def setEval(self):
        for model in self.models.values():
            if model is not None:
                model.eval()

    def build_model(self):
        raise NotImplementedError

    def train(self):
        try:
            print("Total Trainable Parameters : {}".format(self.totalTrainableParams))
            
            for epoch in tqdm(range(self.epochs)):
                self.train_epoch(epoch)
                metrics = self.evaluate(epoch, self.dev_loader)
                self.scheduler.step()
                self.save_checkpoint(epoch,metrics)
                print('*' * 89)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
        print("Testing")
        unseen_metrics = self.evaluate(epoch, self.test_loader)
        print(unseen_metrics)
             
    def train_epoch(self,epoch):
        raise NotImplementedError

    def evaluate(self,epoch):
        raise NotImplementedError

    def save_checkpoint(self,epoch, metrics):
        try:
            if metrics['auc'] > self.best_auc:
                outpath = os.path.join('./checkpoints',self.model_name, "{}_{}".format(metrics['auc'],metrics['accuracy']))
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                
                    print('Saving..')
                    print("Saved Model - Metrics",metrics)
                    for name in self.models:
                        savePath = os.path.join(outpath, "{}.pth".format(name))
                        toSave = self.models[name].state_dict()
                        torch.save(toSave, savePath)
                    savePath = os.path.join(outpath, "{}.pth".format(self.optim.lower()))
                    torch.save(self.optimizer.state_dict(), savePath)
                    self.best_acc = metrics['accuracy']
                    self.best_auc = metrics['auc']
                    print("best auc:", metrics['auc'])
        except Exception as e:
            print("Error:",e)

    def load_checkpoint(self):
        raise NotImplementedError