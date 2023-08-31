import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F

from typing import Optional, Tuple
from functools import partial
from itertools import chain

from torch_geometric.nn import to_hetero
from torch_geometric.nn.models import GAE
from torch_geometric.utils import dropout_edge, add_self_loops, remove_self_loops,negative_sampling

class GMAE(GAE):
    def __init__(self, encoder: Module, decoder: Module,**kwargs):
        super().__init__(encoder, decoder)
        
        mask_rate = kwargs.pop('mask_rate',0.3)
        replace_rate = kwargs.pop('replace_rate',0.1)
        concat_hidden = kwargs.pop('concat_hidden',False)
        loss_fn = kwargs.pop('loss_fn',"sce")
  
        self._mask_rate = mask_rate
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self._drop_edge_rate = 0.0
        self._concat_hidden = concat_hidden
        self.enc_mask_token = nn.Parameter(torch.zeros(1, encoder.in_channels))
        self._decoder_type = None #TODO :Unused
        self.mask_attr_prediction = AttributeMasking()
        
        if self._concat_hidden:
            self.encoder_to_decoder = nn.Linear(decoder.in_channels * encoder.num_layers, decoder.in_channels, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(decoder.in_channels, decoder.in_channels, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, 2)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def sce_loss(self,x, y, alpha=3):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        loss = loss.mean()
        return loss

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(self.sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def forward(self, x,edge_index):
        use_x,use_edge_index,mask_nodes,keep_nodes = self.mask_attr_prediction(x,edge_index)

        enc_rep, all_hidden = self.encoder(use_x, use_edge_index)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "linear") :
            recon = self.decoder(rep)
        else:
            recon = self.decoder(rep, use_edge_index)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)

        encoder_loss = self.encoder.get_loss()
        # loss_item = {"loss": loss.item()}
        return loss,encoder_loss
    
    def embed(self, g_data):
        x, edge_index = g_data.x,g_data.edge_index
        rep, hidden_list = self.encoder(x, edge_index)
        return rep
    
    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
    


class HeteroGMAE(GMAE):
    def __init__(self, encoder: Module, decoder: Module, **kwargs):
        super().__init__(encoder, decoder, **kwargs)

        # graph_metadata = kwargs.pop('graph_metadata')
        node_types = ['text', 'image','image_feats']
        edge_types = [
            ('image', 'have', 'image_feats'),
            ('text', 'implicit_relations', 'image_feats'), 
            ('image_feats', 'rev_have', 'image'), 
            ('image_feats', 'rev_implicit_relations', 'text')
        ]
        graph_metadata = (node_types, edge_types)
        self.encoder = to_hetero(encoder,metadata=graph_metadata)


    # def mask_attr_prediction(self, g_data):
    #     pass

    # def encoding_mask(self):
    #     pass


class AttributeMasking(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mask_rate = 
        self._replace_rate = 

    def forward(self, x, edge_index):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(self._mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        num_noise_nodes = int(self._replace_rate * num_mask_nodes)
        if self._replace_rate > 0 and num_noise_nodes>0:
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
            try:
                use_x = x.clone()
                use_x[token_nodes] = 0.0
                use_x[noise_nodes] = x[noise_to_be_chosen]
            except Exception as e:
                print(e)
                print(num_nodes,noise_nodes,noise_to_be_chosen)
        else:
            use_x = x.clone()
            token_nodes = mask_nodes
            use_x[mask_nodes] = 0.0

        use_x[token_nodes] += self.enc_mask_token

        if self._drop_edge_rate > 0:
            use_edge_index, masked_edges = dropout_edge(edge_index, self._drop_edge_rate)
            use_edge_index = add_self_loops(use_edge_index)[0]
        else:
            use_edge_index = edge_index
            
        return use_x,use_edge_index,mask_nodes,keep_nodes
