from Models.SAGE import GraphSAGE
from Trainers import MMGNNTrainer

class MMSAGETrainer(MMGNNTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.build_model()
        self.getTrainableParams()
        self.setup_optimizer_losses()

    def build_model(self):
        super().build_model()
        PROJECTION_DIM = self.config['projection_dim']
        self.models['graph'] = GraphSAGE(PROJECTION_DIM,64,1).to(self.device)