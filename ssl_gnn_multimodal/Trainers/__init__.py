from Trainers.MMGNNTrainer import MMGNNTrainer 
from Trainers.MMGATTrainer import MMGATTrainer 
from Trainers.MMSAGETrainer import MMSAGETrainer
from Trainers.VGAETrainer import VGAETrainer
from Trainers.GMAETrainer import GMAETrainer
from Trainers.SuperGATTrainer import SuperGATTrainer


def getTrainer(config):
    print(config['trainer'],config['model'],"============================================")
    if config['trainer']=='MMGCN':
        net = MMGNNTrainer(config)
    elif config['trainer']=='MMGAT':
        net = MMGATTrainer(config)
    elif config['trainer']=='MMSAGE':
        net = MMSAGETrainer(config)
    elif config['trainer']=="VGAE":
        net = VGAETrainer(config)
    elif config['trainer']=="GMAE":
        net = GMAETrainer(config)
    elif config['trainer']=="GMAE_SUPERGAT":
        net = SuperGATTrainer(config)
    else:
        raise NotImplementedError("Model not available")

    return net



