import torch
import numpy as np
from mlreco.models.layers.sparse_autoencoder import EncoderLayer
from mlreco.models.gnn.cluster_geo_encoder import ClustGeoNodeEncoder, ClustGeoEdgeEncoder
from mlreco.utils.gnn.fragment_reducer import build_image_filled, build_image_no_filling, build_image_wth_tre

class ClustSparseNodeEncoder(torch.nn.Module):
    """
    Uses a CNN to produce node features for cluster GNN

    """
    def __init__(self, model_config):
        super(ClustSparseNodeEncoder, self).__init__()
        
        # Take the parameters from the config
        self.geo_features = model_config.get('add_geo', False)
        self.reduction = model_config.get('reduction', 'basic')
        
        self.encoder = EncoderLayer(model_config)
        self.geo_encoder = ClustGeoNodeEncoder(model_config)

    def forward(self, data, clusts):

        # Use cluster ID as a batch ID, pass through CNN
        device = data.device
        cnn_data = torch.empty((0,5), device=device, dtype=torch.float)
        image_data = torch.empty((0,4), device=device, dtype=torch.float)
        
        for i, c in enumerate(clusts):
            
            if (self.reduction == 'basic'):
                image, scaling, offset = build_image_no_filling(data[c,:3].detach().numpy(), i, values=data[c,4].detach().numpy())
            elif (self.reduction == 'fill'):
                image, scaling, offset = build_image_filled(data[c,:3].detach().numpy(), i, values=data[c,4].detach().numpy())
            elif (self.reduction == 'threshold'):
                image, scaling, offset = build_image_wth_tre(data[c,:3].detach().numpy(), i, values=data[c,4].detach().numpy())
            else :
                raise ValueError('Reduction mode not recognized')
                
            aux_cnn = torch.tensor(image, device=device, dtype=torch.float)
            aux_image = torch.tensor([[scaling, offset[0],offset[1],offset[2]]], device=device, dtype=torch.float)
            cnn_data = torch.cat((cnn_data, aux_cnn))
            image_data = torch.cat((image_data, aux_image))
            l = image[:,0].size
            cnn_data[-l:,3] = i*torch.ones(l).to(device)
            cnn_data[-l:,4] = torch.ones(l).to(device)
            
        if self.geo_features:
            return torch.cat((image_data, self.encoder(cnn_data), self.geo_encoder(data, clusts)),1)
        else:
            return torch.cat((image_data, self.encoder(cnn_data)),1)

class ClustSparseEdgeEncoder(torch.nn.Module):
    """
    Uses a CNN to produce edge features for cluster GNN

    """
    def __init__(self, model_config):
        super(ClustSparseEdgeEncoder, self).__init__()

        # Initialize the CNN
        self.encoder = EncoderLayer(model_config)

        # Take the parameters from the config
        self.geo_features = model_config.get('add_geo', False)
        self.reduction = model_config.get('reduction', 'basic')
        
        self.geo_encoder = ClustGeoEdgeEncoder(model_config)

    def forward(self, data, clusts, edge_index):

        # Check if the graph is undirected, select the relevant part of the edge index
        half_idx = int(edge_index.shape[1]/2)
        undirected = (not edge_index.shape[1]%2 and [edge_index[1,0], edge_index[0,0]] == edge_index[:,half_idx].tolist())
        if undirected: edge_index = edge_index[:,:half_idx]

        # Use edge ID as a batch ID, pass through CNN
        device = data.device
        image_data = torch.empty((0,2), device=device, dtype=torch.float)
            
        for i, e in enumerate(edge_index.T):
            aux_image = torch.tensor([[1,1]], device=device, dtype=torch.float)
            image_data = torch.cat((image_data, aux_image))

        if self.geo_features:
            feats = self.geo_encoder(data, clusts, edge_index)
        else:
            feats = image_data

        # If the graph is undirected, duplicate features
        if undirected:
            feats = torch.cat([feats,feats])

        return feats