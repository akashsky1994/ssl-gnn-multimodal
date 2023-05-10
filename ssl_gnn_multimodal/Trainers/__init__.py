from Trainers.MMGNNTrainer import MMGNNTrainer 
from Trainers.MMGATTrainer import MMGATTrainer 
from Trainers.MMSAGETrainer import MMSAGETrainer
from Trainers.VGAETrainer import VGAETrainer
from Trainers.GMAETrainer import GMAETrainer
from Trainers.SuperGATTrainer import SuperGATTrainer


def getTrainer(config):
    print(config['model'],"============================================")
    if config['model']=='MMGCN':
        net = MMGNNTrainer(config)
    elif config['model']=='MMGAT':
        net = MMGATTrainer(config)
    elif config['model']=='MMSAGE':
        net = MMSAGETrainer(config)
    elif config['model']=="VGAE":
        net = VGAETrainer(config)
    elif config['model']=="GMAE":
        net = GMAETrainer(config)
    elif config['model']=="GMAE_SUPERGAT":
        net = SuperGATTrainer(config)
    else:
        raise NotImplementedError("Model not available")

    return net



