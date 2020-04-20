# CNN feature extractor for Cluster GNN
import torch
import numpy as np
from mlreco.models.layers.full_encoder_32 import EncoderLayer

def build_image(voxels, clust, values=None, size=32):
    # Build an empty image
    image = []
    # Find the max voxel range of voxels, hence the grid step (on a ]0,1] scale)
    mins      = np.min(voxels, axis=0)
    ranges    = np.ptp(voxels, axis=0)
    max_range = np.max(ranges)+1
    grid_step = 1./max_range
    # Rescale and center voxels
    grid_mins = (voxels-mins-ranges/2-0.5)*grid_step + 0.5
    # For each voxel in the input, divide up the energy into the output pixels it overlaps with
    new_step = 1./size
    for e, v in enumerate(grid_mins):
        # Find the new voxels it covers in x, y and z
        imins  = (v/new_step).astype(int)
        imaxs  = ((v-1e-9+grid_step)/new_step).astype(int)+1
        irange = imaxs-imins
        # Find the fractions that goes into each voxel for each axis
        for i in range(imins[0], imaxs[0]):
            for j in range(imins[1], imaxs[1]):
                for k in range(imins[2], imaxs[2]):
                    idx = np.array([i,j,k])
                    fracs = (np.min([v+grid_step, (idx+1)*new_step], axis=0)-np.max([v, idx*new_step], axis=0))/grid_step
                    image.append([i,j,k,len(voxels),np.prod(fracs)*values[e]])
    # Return image, scaling factor and offset
    return np.array(image), float(size)/max_range, mins+ranges/2



class ClustCNNNodeEncoder(torch.nn.Module):
    """
    Uses a CNN to produce node features for cluster GNN

    """
    def __init__(self, model_config):
        super(ClustCNNNodeEncoder, self).__init__()

        # Initialize the CNN
        self.encoder = EncoderLayer(model_config)

    def forward(self, data, clusts):

        # Use cluster ID as a batch ID, pass through CNN
        device = data.device
        cnn_data = torch.empty((0,5), device=device, dtype=torch.float)
        for i, c in enumerate(clusts):
            aux = torch.tensor(build_image(data[c,:3].detach().numpy(), i, values=data[c,4].detach().numpy())[0], device=device, dtype=torch.float)
            cnn_data = torch.cat((cnn_data, aux))
            cnn_data[-len(c):,3] = i*torch.ones(len(c)).to(device)

        return self.encoder(cnn_data)[0]

class ClustCNNEdgeEncoder(torch.nn.Module):
    """
    Uses a CNN to produce edge features for cluster GNN

    """
    def __init__(self, model_config):
        super(ClustCNNEdgeEncoder, self).__init__()

        # Initialize the CNN
        self.encoder = EncoderLayer(model_config)

    def forward(self, data, clusts, edge_index):

        # Check if the graph is undirected, select the relevant part of the edge index
        half_idx = int(edge_index.shape[1]/2)
        undirected = (not edge_index.shape[1]%2 and [edge_index[1,0], edge_index[0,0]] == edge_index[:,half_idx].tolist())
        if undirected: edge_index = edge_index[:,:half_idx]

        # Use edge ID as a batch ID, pass through CNN
        device = data.device
        cnn_data = torch.empty((0, 5), device=device, dtype=torch.float)
        for i, e in enumerate(edge_index.T):
            ci, cj = clusts[e[0]], clusts[e[1]]
            aux = torch.tensor(build_image(data[ci,:3].detach().numpy(), i, values=data[ci,4].detach().numpy())[0], device=device, dtype=torch.float)
            cnn_data = torch.cat((cnn_data, aux))
            aux = torch.tensor(build_image(data[cj,:3].detach().numpy(), i, values=data[cj,4].detach().numpy())[0], device=device, dtype=torch.float)
            cnn_data = torch.cat((cnn_data, aux))
            cnn_data[-len(ci)-len(cj):,3] = i*torch.ones(len(ci)+len(cj)).to(device)

        feats = self.encoder(cnn_data)[0]

        # If the graph is undirected, duplicate features
        if undirected:
            feats = torch.cat([feats,feats])

        return feats
