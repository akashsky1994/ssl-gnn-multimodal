from Trainers import GMAETrainer
from Models.GAT import DeepSuperGAT
from Models.GMAE import GMAE

class SuperGATTrainer(GMAETrainer):
    def __init__(self, config) -> None:
        super().__init__(config)

    def build_model(self):
        GNN_OUT_CHANNELS = self.config['gnn_out_channels']
        PROJECTION_DIM = self.config['projection_dim']
        super().build_model()
        graph_encoder = DeepSuperGAT(in_channels=PROJECTION_DIM,hidden_channels=2*GNN_OUT_CHANNELS,out_channels=GNN_OUT_CHANNELS,num_layers=3,nheads=4,last_layer=True,jk=self.config.get('jk'),norm_type="graph_norm",activation_type="elu",dropout=0.3).to(self.device)
        graph_decoder = DeepSuperGAT(in_channels=GNN_OUT_CHANNELS,hidden_channels=GNN_OUT_CHANNELS,out_channels=PROJECTION_DIM,num_layers=1,nheads=4,last_layer=False,norm_type="graph_norm",activation_type="elu",dropout=0.3).to(self.device)
        self.models['graph'] = GMAE(graph_encoder,graph_decoder).to(self.device)