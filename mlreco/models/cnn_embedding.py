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

class EncoderModel(torch.nn.Module):
    def __init__(self, cfg):
        super(EncoderModel, self).__init__()
        
        # Choose what type of node to use
        self.node_type = cfg.get('node_type', 0)
        self.node_min_size = cfg.get('node_min_size', -1)
        self.node_encoder = node_encoder_construct(cfg)[0]
        
        layers = []
        layers.append(torch.nn.Linear(67, 32))
        layers.append(torch.nn.Linear(32, 16))
        layers.append(torch.nn.Linear(8, 2))
        
        self.MLP = torch.nn.Sequential(*layers)

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
        
       # Obtain node features
        x = self.node_encoder(data, clusts)
        
        #print(x.size())
        
        # Use cluster ID as a batch ID, pass through CNN
        device = data.device
        true_labels = torch.empty((0), device=device, dtype=torch.long)
        training_data = torch.empty((0,3), device=device, dtype=torch.float)
        for i, c in enumerate(clusts):
            true_voxels = data[c,:3]
            label = rd.randint(0,1)
            #print(label)
            label_tensor = torch.tensor([label], device=device, dtype=torch.long)
            true_labels = torch.cat((true_labels,label_tensor),0)
            #If we want a voxel in our fragment
            if (label == 1):
                ind = rd.randint(0,true_voxels.size()[0]-1)
                voxel = true_voxels[ind]
            else:
                voxel = torch.tensor([rd.randint(0,1024), rd.randint(0,1024), rd.randint(0,1024)], device=device, dtype=torch.float)
            voxel = voxel.float()
            sample = torch.empty((1,3), device=device, dtype=torch.float)
            sample[0] = voxel
                
            training_data = torch.cat((training_data, sample),0)
            
        
        x = torch.cat((x,training_data),1)
        
        x = self.MLP(x)
        #print(x[0])
        
        return {'result': [x], 'embedding': [x], 'true': [true_labels]}
    
class EncoderModelLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(EncoderModelLoss, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, out, clusters):
        """
        Applies the requested loss on the edge prediction.
        Args:
            out (dict):
                'edge_pred' (torch.tensor): (E,2) Two-channel edge predictions
                'clusts' ([np.ndarray])   : [(N_0), (N_1), ..., (N_C)] Cluster ids
                'edge_index' (np.ndarray) : (E,2) Incidence matrix
            clusters ([torch.tensor])     : (N,8) [x, y, z, batchid, value, id, groupid, shape]
            graph ([torch.tensor])        : (N,3) True edges
        Returns:
            double: loss, accuracy, clustering metrics
        """
        loss = self.loss(out['result'][0], out['true'][0])
        
        accuracy = 0
        size = out['true'][0].size()[0]
        for i in range(out['true'][0].size()[0]):
            if(out['true'][0][i] == 0):
                if (out['result'][0][i,0]>out['result'][0][i,1]):
                    accuracy +=1
            else :
                if (out['result'][0][i,0]<out['result'][0][i,1]):
                    accuracy +=1
        accuracy /= size
        #loss = self.loss(out['result'][0].view((-1)), out['true'][0])
        
        return {
            'accuracy': accuracy,
            'loss': loss
        }
        


