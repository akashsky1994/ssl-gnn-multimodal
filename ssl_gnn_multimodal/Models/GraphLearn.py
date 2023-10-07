import torch
from Models.Encoder import ImageEncoder,TextEncoder,ProjectionHead
from torch_geometric.data import HeteroData,Data as GraphData,Batch
from torch_geometric.loader import DataLoader as GDataLoader,DataListLoader
import torch_geometric.transforms as T
from utils import get_device

class GraphLearn(torch.nn.Module):
    '''
    Graph Learning Module to convert and optimize the text and image data to a fused graph.
    '''
    def __init__(self, image_model='resnet50',text_model='distilbert', **kwargs) -> None:
        super().__init__()
        projection_dim = kwargs.pop('projection_dim',512)
        trainable = kwargs.pop('trainable',True)
        self.image_encoder = ImageEncoder(model_name=image_model,pretrained=True,trainable=trainable)
        self.text_encoder = TextEncoder(model_name=text_model, pretrained=True, trainable=trainable)
        self.image_projection = ProjectionHead(self.image_encoder.out_features,projection_dim,trainable=trainable)
        self.text_projection = ProjectionHead(self.text_encoder.out_features,projection_dim,trainable=trainable)
    
    def forward(self,images,image_features,tokenized_text,attention_masks,labels):
        text_embeddings = self.text_projection(self.text_encoder(input_ids=tokenized_text, attention_mask=attention_masks))
        image_embeddings = self.image_projection(self.image_encoder(images))
        image_feat_data = self.get_image_feature_embeddings(image_features)
        g_data = self.generate_subgraph(image_embeddings,image_feat_data,text_embeddings,labels)
        return g_data

    def get_image_feature_embeddings(self,image_features):
        embeddings = []
        b,n = image_features.shape[0],image_features.shape[1]
        batch_mapping = []
        reshaped_tensor  = []
        for i in range(b):
            for j in range(n):
                if torch.count_nonzero(image_features[i][j])!=0:
                    reshaped_tensor.append(image_features[i][j])
                    batch_mapping.append(i)
        reshaped_tensor = torch.stack(reshaped_tensor)
        embeddings = self.image_projection(self.image_encoder(reshaped_tensor))
        return embeddings,batch_mapping

    def generate_subgraph(self,image_embeddings,image_feat_data,text_embeddings,labels):
        graph_list = []
        image_feat_embeddings, batch_mapping = image_feat_data
        j,k= 0,0
        for i in range(len(image_embeddings)):
            while len(batch_mapping)>k and batch_mapping[k]==i:
                k+=1
            n_img_features = k-j
            data = GraphData()
            data.x = torch.cat([image_embeddings[i].unsqueeze(0),text_embeddings[i].unsqueeze(0),image_feat_embeddings[j:k]])
            j = k
            imgEdges = torch.tensor([[0]*(n_img_features),[i+2 for i in range(n_img_features)]],dtype=torch.long)
            textEdges = torch.tensor([[1]*(n_img_features),[i+2 for i in range(n_img_features)]],dtype=torch.long)

            data.edge_index = torch.cat([imgEdges,textEdges],dim=1)
            data.y = labels[i]
            data = T.ToUndirected()(data)
            data = T.NormalizeFeatures()(data)
            graph_list.append(data)
        
        device,n_gpus = get_device()
        if n_gpus>1:
            g_loader = DataListLoader(graph_list,batch_size=self.batch_size*n_gpus,shuffle=True)
            batched_graph = next(iter(g_loader))
        else:
            batched_graph = Batch.from_data_list(graph_list)
            batched_graph = batched_graph.to(labels.device)
        
        return batched_graph


class HeteroGraphLearn(GraphLearn):
    '''
    Graph Learning Module to convert and optimize the text and image data to a fused heterogenous graph.
    '''
    def __init__(self, image_model='resnet50',text_model='distilbert', **kwargs) -> None:
        super().__init__(image_model,text_model,**kwargs)
        self.image_projection = torch.nn.Identity()
        self.text_projection = torch.nn.Identity()

    def generate_subgraph(self,image_embeddings,image_feat_data,text_embeddings,labels):
        graph_list = []
        image_feat_embeddings, batch_mapping = image_feat_data
        j,k= 0,0
        for i in range(len(image_embeddings)):
            while len(batch_mapping)>k and batch_mapping[k]==i:
                k+=1
            n_img_features = k-j
            data = HeteroData().to(labels.device)
            
            # node
            data['text'].x = torch.cat([text_embeddings[i].unsqueeze(0)])
            data['image'].x = torch.cat([image_embeddings[i].unsqueeze(0)])
            data['image_feats'].x = torch.cat([image_feat_embeddings[j:k]])
            j = k
            
            # edge index
            imgEdges = torch.tensor([[0]*(n_img_features),[i for i in range(n_img_features)]],dtype=torch.long)
            data['image','have','image_feats'].edge_index = imgEdges

            textEdges = torch.tensor([[0]*(n_img_features),[i for i in range(n_img_features)]],dtype=torch.long)
            data['text','implicit_relations','image_feats'].edge_index = textEdges
            
            #label
            data.y = labels[i]
            data = T.ToUndirected()(data)
            # data = T.NormalizeFeatures()(data) #TODO why
            graph_list.append(data)

        device,n_gpus = get_device()
        if n_gpus>1:
            g_loader = DataListLoader(graph_list,batch_size=self.batch_size*n_gpus,shuffle=True)
            batched_graph = next(iter(g_loader))
        else:
            batched_graph = Batch.from_data_list(graph_list)
            batched_graph = batched_graph.to(labels.device)
        
        return batched_graph

    @property
    def metadata(self):
        node_types = ['text', 'image','image_feats']
        edge_types = [
            ('image', 'have', 'image_feats'),
            ('text', 'implicit_relations', 'image_feats'), 
            ('image_feats', 'rev_have', 'image'), 
            ('image_feats', 'rev_implicit_relations', 'text')
        ]
        metadata = (node_types, edge_types)
        return metadata

if __name__ == "__main__":
    gl = HeteroGraphLearn()