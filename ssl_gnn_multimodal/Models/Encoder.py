import torch
import torch.nn as nn
import timm
from transformers import DistilBertModel, DistilBertConfig, RobertaModel ,RobertaConfig

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet50', pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        self.out_features = 2048 #TODO make it generic
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class TextEncoder(nn.Module):
    def __init__(self, model_name='distilbert', pretrained=True, trainable=True):
        super().__init__()
        
        self.model,self.out_features = resolve_text_model(model_name,pretrained)
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.1,
        trainable=True
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
        for p in self.parameters():
            p.requires_grad = trainable
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    

def resolve_text_model(model_name,pretrained):
    if model_name=='distilbert':
        if pretrained:
            model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        else:
            model = DistilBertModel(config=DistilBertConfig())
        out_features = model.transformer.layer[-1].ffn.lin2.out_features
    elif model_name=='roberta':
        if pretrained:
            model = RobertaModel.from_pretrained('roberta-base')
        else:
            model = RobertaModel(config=RobertaConfig())
        out_features = model.pooler.dense.out_features
    return model,out_features

if __name__ == "__main__":
    print(ImageEncoder())