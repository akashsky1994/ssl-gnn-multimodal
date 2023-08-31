from typing import Any, Callable, Dict, Optional, Union
import torch
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv,SuperGATConv, global_mean_pool
from torch_geometric.nn.models import GAT #TODO
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge

from utils import get_normalization,get_activation

class GATClassifier(torch.nn.Module):
    def __init__(self,num_features,num_classes=1,training=True):
        super(GATClassifier, self).__init__()
        self.training = training
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        
        self.conv1 = GATv2Conv(num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATv2Conv(self.hid*self.in_head, 16, concat=False,
                             heads=self.out_head, dropout=0.6)
        self.classifier = Linear(16, num_classes)
    
    def forward(self,x, edge_index, batch):    
        # Dropout before the GAT layer helps avoid overfitting
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=True)
        out = self.classifier(x)
        out = out.view(-1)
        return out,x    #,F.log_softmax(x, dim=1)
    

class DeepGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,num_layers,nheads,norm_type="graph_norm",activation_type="prelu",dropout=0.3,jk=None,last_layer=True,**kwargs):
        super(DeepGNN, self).__init__()
        self.in_channels = -1
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.nheads = nheads
        self.dropout = dropout

        normFn = get_normalization(norm_type)
        activation = get_activation(activation_type)
        self.gnn_layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        last_norm = torch.nn.Identity()
        last_act = torch.nn.Identity()
        if last_layer is True:
            last_norm = normFn(out_channels)
            last_act = activation
        if num_layers == 1:
            
            self.gnn_layers.append(self.init_convs(self.in_channels, out_channels,nheads,dropout,concat=False,**kwargs))
            self.norms.append(last_norm)
            self.acts.append(last_act)
        else:
            # input projection

            self.gnn_layers.append(self.init_convs(self.in_channels, (hidden_channels//nheads),nheads,dropout,**kwargs))
            self.norms.append(normFn(hidden_channels))
            self.acts.append(activation)
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gnn_layers.append(self.init_convs(hidden_channels, (hidden_channels//nheads),nheads,dropout,**kwargs))
                self.norms.append(normFn(hidden_channels))
                self.acts.append(activation)
            
            # output projection
            if jk is not None and isinstance(jk, str):
                self.jkmode = jk

                self.gnn_layers.append(self.init_convs(hidden_channels, (hidden_channels//nheads),nheads,dropout,**kwargs))
                self.norms.append(normFn(hidden_channels))
                self.acts.append(activation)

                self.jk = JumpingKnowledge(jk,hidden_channels,num_layers) #TODO:REVIEW
                in_channels_lin = hidden_channels
                if self.jkmode == 'cat':
                    in_channels_lin = num_layers * hidden_channels
                self.lin = Linear(in_channels_lin, out_channels)
            else:            
                self.gnn_layers.append(self.init_convs(hidden_channels, out_channels,nheads,dropout,concat=False,**kwargs))
                self.norms.append(last_norm)
                self.acts.append(last_act)
            
        self.head = torch.nn.Identity()

    def forward(self, x, edge_index):
        h = x
        hidden_list = []
        for i, (gnn_layer, norm, act) in enumerate(zip(self.gnn_layers, self.norms,self.acts)):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = act(norm(gnn_layer(h,edge_index)))
            hidden_list.append(h)
        
        # output projection
        h = self.lin(self.jk(hidden_list)) if hasattr(self, 'jk') else h
        h = self.head(h)
        return h, hidden_list
        
    def get_loss(self):
        return None
        

class DeepGAT(DeepGNN):
    def init_convs(self,in_channels,out_channels,nheads,dropout,concat=True,**kwargs):
        return GATv2Conv(-1, out_channels,nheads,concat=concat,dropout=dropout,add_self_loops=False)
    

class DeepSuperGAT(DeepGNN):
    def init_convs(self,in_channels,out_channels,nheads,dropout,concat=True,**kwargs):
        attention_type = kwargs.pop('attention_type', 'MX')
        edge_sample_ratio = kwargs.pop('edge_sample_ratio', 0.8)
        is_undirected = kwargs.pop('is_undirected', True)
        return SuperGATConv(-1, out_channels,nheads,concat=concat,dropout=dropout,attention_type=attention_type,edge_sample_ratio=edge_sample_ratio,is_undirected=is_undirected)
    
    def get_loss(self):
        att_loss = 0
        for i in range(self.num_layers):
            att_loss += self.gnn_layers[i].get_attention_loss()

        return att_loss