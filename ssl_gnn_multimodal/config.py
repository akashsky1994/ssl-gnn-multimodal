import configparser
import argparse


def load_config(args):
    config_path = args.conf_file
    model_name = args.model
    dataset = args.dataset
    argsDict = {k: v for k, v in vars(args).items() if v is not None}
    # loading .cfg config file from global scope
    if config_path is None:
        config_path = 'model_config.cfg'
    configDict = {}
    config = configparser.ConfigParser()
    config.read(config_path)
    configDict =  {**config['GLOBAL'],**config[model_name.upper()],**config[dataset.upper()],**argsDict}
    configDict['model_name'] = model_name
    
    configDict['gnn_out_channels'] = int(configDict['gnn_out_channels'])
    configDict['projection_dim'] = int(configDict['projection_dim'])
    configDict['recon_loss_coef'] = float(configDict['recon_loss_coef'])
    configDict['encoder_loss_coef'] = float(configDict['encoder_loss_coef'])
    configDict['batchsize'] = int(configDict['batchsize'])
    configDict['epochs'] = int(configDict['epochs'])
    configDict['workers'] = int(configDict['workers'])
    configDict['lr'] = float(configDict['lr'])

    return configDict