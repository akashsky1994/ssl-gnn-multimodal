
from Models.GAT import GATClassifier
from Trainers import MMGNNTrainer

class MMGATTrainer(MMGNNTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.build_model()
        self.getTrainableParams()
        self.setup_optimizer_losses()

    def build_model(self):
        super().build_model()
        PROJECTION_DIM = self.config['projection_dim']
        self.models['graph'] = GATClassifier(PROJECTION_DIM,num_classes=1).to(self.device)