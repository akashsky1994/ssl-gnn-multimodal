
import argparse
import torch
from Trainers import getTrainer
from config import load_config
import json

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', type=float, dest='lr', help='learning rate')
    parser.add_argument('--resume', '-r', help='resume from checkpoint')
    parser.add_argument('--cpu', '-c', action='store_true',help='Use CPU only')
    parser.add_argument('--workers', '-w', type=int, dest='workers', help='no of workers')    
    parser.add_argument('--epochs', '-e', type=int, dest='epochs', help='Epochs')
    parser.add_argument('--optim', '-o', type=str, dest='optimizer', help='optimizer type')
    parser.add_argument('--batchsize', '-bs',type=int, dest='batchsize',help='Batch Size')
    parser.add_argument('--model', '-m',default='GMAE', type=str)
    parser.add_argument('--data_path', '-d', type=str, dest='datapath')
    parser.add_argument('--dataset', '-ds',default='HATEFULMEME', dest='dataset', type=str)
    parser.add_argument('--pretrain', '-pt',action='store_true')
    parser.add_argument(
        "--conf", action="store", dest="conf_file",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    config = load_config(args)
    print(json.dumps(config,indent=4))
    net = getTrainer(config)
    net.train()

    print("Model Training Completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        torch.cuda.empty_cache()
        raise e
