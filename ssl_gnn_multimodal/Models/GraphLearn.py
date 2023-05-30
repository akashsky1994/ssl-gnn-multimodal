import torch
from Models.Encoder import ImageEncoder,TextEncoder,ProjectionHead
from torch_geometric.data import HeteroData,Data as GraphData,Batch
from torch_geometric.loader import DataLoader as GDataLoader,DataListLoader
import torch_geometric.transforms as T

class GraphLearn(torch.nn.Module):
    def __init__(self, projection_dim, trainable=True) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(model_name='resnet50',pretrained=True,trainable=trainable).to(self.device)
        self.text_encoder = TextEncoder(model_name='distilbert-base-uncased', pretrained=True, trainable=trainable).to(self.device)
        self.image_projection = ProjectionHead(2048,projection_dim,trainable=trainable).to(self.device)
        self.text_projection = ProjectionHead(768,projection_dim,trainable=trainable).to(self.device)
    
    def forward(self,images,image_features,tokenized_text,attention_masks,labels):
        text_embeddings = self.text_projection(self.text_encoder(input_ids=tokenized_text, attention_mask=attention_masks))
        image_embeddings = self.models['image_projection'](self.models['image_encoder'](images))
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
        embeddings = self.models['image_projection'](self.models['image_encoder'](reshaped_tensor))
        # #explicit garbage collection
        # del image_features
        # del reshaped_tensor
        # gc.collect()
        return embeddings,batch_mapping

    def generate_subgraph(self,image_embeddings,image_feat_data,text_embeddings,labels):
        graph_list = []
        image_feat_embeddings, batch_mapping = image_feat_data
        j,k= 0,0
        for i in range(len(image_embeddings)):
            while len(batch_mapping)>k and batch_mapping[k]==i:
                k+=1
            n_img_features = k-j
            data = GraphData().to(self.device)
            data.x = torch.cat([image_embeddings[i].unsqueeze(0),text_embeddings[i].unsqueeze(0),image_feat_embeddings[j:k]])
            j = k
            imgEdges = torch.tensor([[0]*(n_img_features),[i+2 for i in range(n_img_features)]],dtype=torch.long)
            textEdges = torch.tensor([[1]*(n_img_features),[i+2 for i in range(n_img_features)]],dtype=torch.long)

            data.edge_index = torch.cat([imgEdges,textEdges],dim=1)
            data.y = labels[i]
            data = T.ToUndirected()(data)
            data = T.NormalizeFeatures()(data)
            graph_list.append(data)
        
        if self.n_gpus>1:
            g_loader = DataListLoader(graph_list,batch_size=self.batch_size*self.n_gpus,shuffle=True)
            batched_graph = next(iter(g_loader))
        else:
            batched_graph = Batch.from_data_list(graph_list)
            batched_graph = batched_graph.to(self.device)
        
        return batched_graph