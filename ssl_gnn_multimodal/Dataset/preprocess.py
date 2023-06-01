import os
import json
import time
import argparse

from PIL import Image

from glob import glob

from tqdm import tqdm
import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.resnet import ResNet50_Weights
import numpy as np

def generate_and_store_image_features(data,image_feature_model,image_transform,device,BATCHSIZE):
    n = len(data)
    print("data size",n)
    for i in tqdm(range(0,n,BATCHSIZE)):
        j = n if (i+BATCHSIZE)>n else i+BATCHSIZE
        image_path = [data[k] for k in range(i,j)]
        images = torch.stack([image_transform(Image.open(data[k]).convert("RGB")) for k in range(i,j)]).to(device)
        outputs = image_feature_model(images)
        for m,output in enumerate(outputs):
            if sum(output['scores']>0.4)>=5: #get features with more than 40% confidence score upto 20 features
                indices = torch.argsort((output['scores']*(output['scores']>0.4)))[-20:]
            else:
                indices = torch.argsort(output['scores'])[-5:] #get top 5 features
            
            masks = output['masks'][indices]
            img_feats = masks*images[m]
            torch.save(img_feats.detach().cpu(),"{}.pt".format(os.path.splitext(image_path[m])[0]))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--dataset', '-d', type=str, dest='dataset', default='mmhs150k') 
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    args = parser.parse_args()


    device = 'cuda'
    image_feature_model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
            weights_backbone=ResNet50_Weights.DEFAULT
        ).to(device).eval()
    image_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(224, 224)),
                    torchvision.transforms.ToTensor()
                ]
            )
    
    if args.dataset=="hateful_meme":
        data_path = "../datasets/hateful_memes/"
        data = [img_path for img_path in glob(os.path.join(data_path,"img/*.jpg"), recursive = True)]
    elif args.dataset=="cc12m":
        data_path = "../datasets/cc12m/"
        data = [img_path for img_path in glob(os.path.join(data_path,"*/*.jpg"), recursive = True) if os.path.exists(img_path.replace("jpg","txt"))]
    elif args.dataset=="mmhs150k":
        data_path = "../datasets/multimodal-hate-speech/"
        data = [img_path for img_path in glob(os.path.join(data_path,"img_resized/*.jpg"), recursive = True)]
        # data = [img_path for img_path in glob(os.path.join(data_path,"img_resized/*.jpg"), recursive = True) if os.path.exists(img_path.replace("jpg","json").replace("img_resized","img_txt"))]

    generate_and_store_image_features(data,image_feature_model,image_transform,device,args.batchsize)
    
