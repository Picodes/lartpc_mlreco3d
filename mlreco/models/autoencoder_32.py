# GNN that attempts to put clusters together into groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from .gnn import edge_model_construct, node_encoder_construct, edge_encoder_construct
from .layers.full_encoder import EncoderLayer
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_label, get_cluster_batch, get_cluster_group
from mlreco.utils.gnn.network import complete_graph, delaunay_graph, mst_graph, bipartite_graph, inter_cluster_distance, get_fragment_edges
from mlreco.utils.gnn.evaluation import edge_assignment, edge_assignment_from_graph
from mlreco.utils import local_cdist
import random as rd

def build_image(voxels, values=None, size=32):
    # Build an empty image
    image = np.zeros((size, size, size))
    # If values none, just make it 1 everywhere
    if values is None:
        values = np.ones(len(voxels))
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
                    image[i,j,k] += np.prod(fracs)*values[e]
    # Return image, scaling factor and offset
    return image, float(size)/max_range, mins+ranges/2

class EncoderModel(torch.nn.Module):
    def __init__(self, cfg):
        super(EncoderModel, self).__init__()
        
        # Choose what type of node to use
        self.node_type = cfg.get('node_type', 0)
        self.node_min_size = cfg.get('node_min_size', -1)
        self.node_encoder = node_encoder_construct(cfg)[0]
        
        layers_encoder = []
        layers_encoder.append(torch.nn.Conv3d(1, 2, 4, stride=2, padding = 1))
        layers_encoder.append(torch.nn.ReLU())
        layers_encoder.append(torch.nn.Conv3d(2, 4, 4, stride=2, padding = 1))
        layers_encoder.append(torch.nn.ReLU())
        layers_encoder.append(torch.nn.Conv3d(4, 8, 3, stride=2))
        
        self.encoder = torch.nn.Sequential(*layers_encoder)
        
        layers_decoder = []
        layers_decoder.append(torch.nn.ConvTranspose3d(8, 4, 4, stride=2))
        layers_decoder.append(torch.nn.ReLU())
        layers_decoder.append(torch.nn.ConvTranspose3d(4, 2, 4, stride=2, padding = 1))
        layers_decoder.append(torch.nn.ReLU())
        layers_decoder.append(torch.nn.ConvTranspose3d(2, 1, 4, stride=2, padding = 1))
        
        self.decoder = torch.nn.Sequential(*layers_decoder)
        

    def forward(self, data):
        # Find index of points that belong to the same clusters
        # If a specific semantic class is required, apply mask
        # Here the specified size selection is applied
        data = data[0]
        device = data.device
        
        if self.node_type > -1:
            mask = torch.nonzero(data[:,-1] == self.node_type).flatten()
            clusts = form_clusters(data[mask], self.node_min_size)
            clusts = [mask[c].cpu().numpy() for c in clusts]
        else:
            clusts = form_clusters(data, self.node_min_size)
            clusts = [c.cpu().numpy() for c in clusts]

        if not len(clusts):
            return {}

        # Get the batch id for each cluster
        batch_ids = get_cluster_batch(data, clusts)
        
        x = torch.empty((0,1,32,32,32), device=device, dtype=torch.float)
        
        # Use cluster ID as a batch ID, pass through CNN
        device = data.device
        for i, c in enumerate(clusts):
            image, ratio1, ratio2 = build_image(data[c,:3].detach().numpy(), data[c,4].detach().numpy())
            x = torch.cat((x,torch.tensor(np.resize(image.astype(float),(1,1,32,32,32)), device=device, dtype=torch.float)),0)
            #print(x.size())
            
        true_x = x
        
        print(x.size())
        
        x = self.encoder(x)
        
        print(x.size())
        
        embedding = x.view(x.size()[0],-1)
        
        x = self.decoder(x)
        
        print(x.size())
        
        return {'result': [x], 'embedding': [embedding], 'true': [true_x]}
    
class EncoderModelLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(EncoderModelLoss, self).__init__()
        self.loss = torch.nn.MSELoss()

    def forward(self, out, clusters):
        loss = self.loss(out['result'][0], out['true'][0])
      
        return {
            'accuracy': loss,
            'loss': loss
        }
        


