# GNN that attempts to put clusters together into groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from .layers.sparse_autoencoder import EncoderLayer
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_label, get_cluster_batch, get_cluster_group
from mlreco.utils.gnn.fragment_reducer import build_image_filled, build_image_no_filling, build_image_wth_tre
import sparseconvnet as scn



class EncoderModel(torch.nn.Module):
    def __init__(self, cfg):
        super(EncoderModel, self).__init__()
        
        # Choose what type of node to use
        self.node_type = cfg.get('node_type', 0)
        self.node_min_size = cfg.get('node_min_size', -1)
        
        self.reduction = cfg.get('reduction', 'basic')
        self.encoder = EncoderLayer(cfg['encoder'])
        print(self.reduction)

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
        
        # Use cluster ID as a batch ID, pass through CNN
        device = data.device
        cnn_data = torch.empty((0,5), device=device, dtype=torch.float)
        for i, c in enumerate(clusts):
            
            if (self.reduction == 'basic'):
                image, scaling, offset = build_image_no_filling(data[c,:3].detach().numpy(), i, values=data[c,4].detach().numpy())
            elif (self.reduction == 'fill'):
                image, scaling, offset = build_image_filled(data[c,:3].detach().numpy(), i, values=data[c,4].detach().numpy())
            elif (self.reduction == 'threshold'):
                image, scaling, offset = build_image_wth_tre(data[c,:3].detach().numpy(), i, values=data[c,4].detach().numpy())
            else :
                raise ValueError('Reduction mode not recognized')
                
            l = len(image[:,0])
            aux = torch.tensor(image, device=device, dtype=torch.float)
            cnn_data = torch.cat((cnn_data, aux))
            cnn_data[-l:,3] = i*torch.ones(l).to(device)
            cnn_data[-l:,4] = torch.ones(l).to(device)

        # Obtain node and edge features
        #x, target = self.encoder(data, clusts)
        encoding, image, original,encoder_x,decoder_x, dec_sparse_x = self.encoder(cnn_data)

        return {'encoding': [encoding], 
                'image': [image], 
                'original': [original], 
                'encoder_x': [encoder_x],
                'decoder_x': [decoder_x],
                'dec_sparse_x': [dec_sparse_x],
                'clusts': [clusts],
                'data': [data]}
    
    
# x original, y autoencoder result
def compare_sparse(x, y):
    cL,cR,L,R = x.metadata.compareSparseHelper(y.metadata, x.spatial_size)
    if x.features.is_cuda:
        cL=cL.cuda()
        cR=cR.cuda()
        L=L.cuda()
        R=R.cuda()
    e = 0
    if cR.numel():
        #was 3/4, 1/2 for try 13
        e += torch.max(1-y.features[cR,0],torch.zeros(y.features[cR,0].size())).pow(2).sum()/cR.numel()/2
    if R.numel():
        e += torch.max(1+y.features[R,0],torch.zeros(y.features[R,0].size())).pow(2).sum()/R.numel()/2
    return e

# x original, y autoencoder result
def metrics(x, y):
    cL,cR,L,R = x.metadata.compareSparseHelper(y.metadata, x.spatial_size)
    if x.features.is_cuda:
        cL=cL.cuda()
        cR=cR.cuda()
        L=L.cuda()
        R=R.cuda()
    e = 0
    print("Activated true   : ", cR.numel())
    print("Missing activated: ", L.numel())
    print("Activated false  : ", R.numel())

class EncoderModelLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(EncoderModelLoss, self).__init__()
        self.loss = torch.nn.MSELoss()

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
        total_loss, total_acc = 0., 0.
        device = clusters[0].device
        
        if not 'encoding' in out:
            return {
                'accuracy': 0.,
                'loss': torch.tensor(0., requires_grad=True, device=clusters[0].device)
            }
        
        encoding = out['encoding'][0]
        image = out['image'][0]
        original = out['original'][0]
        encoder_x = out['encoder_x'][0]
        decoder_x = out['decoder_x'][0]
        dec_sparse_x = out['dec_sparse_x'][0]
        
        # Handle the case where no cluster/edge were found
        if not encoding.size:
            return {
                'accuracy': 0.,
                'loss': torch.tensor(0., requires_grad=True, device=clusters[0].device)
            }
            
        nbatches = len(original.get_spatial_locations()[:,3].unique())
        print('nbatches: ',nbatches)
        
        # Handle the case where no cluster/edge were found
        if not nbatches:
            return {
                'accuracy': 0.,
                'loss': torch.tensor(0., requires_grad=True, device=clusters[0].device)
            }
            
        #for j in range(nbatches):
        #    print('j: ',j)
        
        #print("Before 1: ", scn.compare_sparse(image, original))
        
        #image.features = torch.zeros((len(image.features)),1).to(device)
        #print("After 0: ", scn.compare_sparse(image, original))       
        
        image_1 = image
        image_1.features = torch.ones((len(image_1.features)),1).to(device)
        
        original_1 = original
        original_1.features = torch.ones((len(original_1.features)),1).to(device)
        #print("After 1: ", scn.compare_sparse(image, original))
        
        #total_loss += scn.compare_sparse(image, original)
        total_acc += 1-scn.compare_sparse(image_1, original_1)
        
        for l in range(len(decoder_x)):
            print(encoder_x[l].spatial_size)
            print(compare_sparse(encoder_x[l], decoder_x[-1-l]))
            total_loss += compare_sparse(encoder_x[l], decoder_x[-1-l])
            metrics(encoder_x[l], dec_sparse_x[-1-l])

        
        print(total_loss)
        #print(total_acc/nbatches)
        
        return {
            'accuracy': (total_acc).detach(),
            'loss': total_loss
        }
        


