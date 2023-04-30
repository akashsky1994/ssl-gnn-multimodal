import os
import json
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
class HatefulMemeDataset(torch.utils.data.Dataset):
    def __init__(self,data_path,data_type, image_transform, tokenizer) -> None:
        super().__init__()
        self.data = [json.loads(l) for l in open(os.path.join(data_path,data_type+'.jsonl'))]
        self.data_dir = data_path
        self.image_transform = image_transform
        self.tokenizer = tokenizer       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load images on the fly.
        # print(os.path.join(self.data_dir, self.data[index]["img"]))
        image_path = os.path.join(self.data_dir, self.data[index]["img"])
        image = Image.open(image_path).convert("RGB")
        image_features = torch.load("{}.pt".format(os.path.splitext(image_path)[0]))
        text = self.data[index].get("text")
        label = self.data[index].get("label")
        
        return image, text, label, image_features

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