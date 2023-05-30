import os
import json
from PIL import Image

import torch

from torch.nn.utils.rnn import pad_sequence

class MMHS150K(torch.utils.data.Dataset):
    def __init__(self,data_path,data_type, image_transform,tokenizer) -> None:
        super().__init__()
        self.data_ids = [x.split('\t') for x in open(os.path.join(data_path,'splits',data_type+'_ids.txt'))]
        self.data = [json.loads(l) for l in open(os.path.join(data_path,'MMHS150K_GT.json'))]
        self.data = {tweet_id:self.data[tweet_id]['labels'] for tweet_id in self.data_ids}
        self.data_dir = data_path
        self.image_transform = image_transform
        self.tokenizer = tokenizer  

    def __len__(self):
        return len(self.data_ids)
    
    def __getitem__(self, index):
        label = 1
        if len(set(self.data[self.data_ids[index]]).intersection(set([0,0,0])))>1:
            label = 0
        image_path = os.path.join(self.data_dir,'img_resized',self.data_ids[index]+'.jpg')
        image = Image.open(image_path).convert("RGB")
        text = [json.loads(l) for l in open(os.path.join(self.data_dir,'img_text',self.data_ids[index]+'.json'))][0]["img_text"]
        img_feat_path = image_path.replace("jpg","pt")
        image_features = torch.load(img_feat_path)
        return image,text,label,image_features
    
    def collate_fn(self,batch):
        # Image Tensor
        tensor_img = torch.stack(
            [self.image_transform(row[0]) for row in batch]
        )

        img_feats = pad_sequence([row[3] for row in batch],batch_first=True)

        # Tokenized Text Tensor 
        encoded_queries = self.tokenizer([row[1] for row in batch])
        lens = [len(row) for row in encoded_queries['input_ids']]
        text_tensor = torch.zeros(len(batch),max(lens),dtype=torch.long)
        attention_mask = torch.zeros(len(batch),max(lens),dtype=torch.long)
        
        for i_batch in range(len(batch)):
            length = lens[i_batch]
            text_tensor[i_batch, :length] = torch.tensor(encoded_queries['input_ids'][i_batch])
            attention_mask[i_batch, :length] = torch.tensor(encoded_queries['attention_mask'][i_batch])
        

        #Label Tensor
        label_tensor = torch.stack([torch.tensor([row[2]],dtype=torch.float32) for row in batch])

        return tensor_img, img_feats, text_tensor,attention_mask,label_tensor