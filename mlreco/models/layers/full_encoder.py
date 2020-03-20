from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch_geometric.nn import MetaLayer, NNConv
import sparseconvnet as scn

class EncoderLayer(torch.nn.Module):

    def __init__(self, cfg):
        super(EncoderLayer, self).__init__()

        # Get the model input parameters
        model_config = cfg

        # Take the parameters from the config
        self._dimension = model_config.get('dimension', 3)
        self.num_strides = model_config.get('num_stride', 10)
        self.m =  model_config.get('feat_per_pixel', 4)
        self.nInputFeatures = model_config.get('input_feat_enc', 1)
        self.leakiness = model_config.get('leakiness_enc', 0)
        self.spatial_size = model_config.get('inp_spatial_size', 1024) #Must be a power of 2
        self.feat_aug_mode = model_config.get('feat_aug_mode', 'constant')
        self.use_linear_output = model_config.get('use_linear_output', False)
        self.num_output_feats = model_config.get('num_output_feats', 64)

        self.out_spatial_size = int(self.spatial_size/4**(self.num_strides-1))
        self.output = self.m*self.out_spatial_size**3
        

        nPlanes = [self.m for i in range(1, self.num_strides+1)]  # UNet number of features per level
        if self.feat_aug_mode == 'linear':
            nPlanes = [self.m * i for i in range(1, self.num_strides + 1)]
        elif self.feat_aug_mode == 'power':
            nPlanes = [self.m * pow(2, i) for i in range(self.num_strides)]
        elif self.feat_aug_mode != 'constant':
            raise ValueError('Feature augmentation mode not recognized')
        kernel_size = 2
        downsample = [kernel_size, 2]  # [filter size, filter stride]

        
        #Input for tpc voxels
        self.input = scn.Sequential().add(
           scn.InputLayer(self._dimension, self.spatial_size, mode=3)).add(
           scn.SubmanifoldConvolution(self._dimension, self.nInputFeatures, self.m, 3, False)) # Kernel size 3, no bias
        self.concat = scn.JoinTable()
    
        
        # Encoding TPC
        self.bn = scn.BatchNormLeakyReLU(nPlanes[0], leakiness=self.leakiness)
        self.encoding_conv = scn.Sequential()
        for i in range(self.num_strides):
            module2 = scn.Sequential()
            if i < self.num_strides-1:
                module2.add(
                    scn.BatchNormLeakyReLU(nPlanes[i], leakiness=self.leakiness)).add(
                    scn.Convolution(self._dimension, nPlanes[i], nPlanes[i+1],
                        downsample[0], downsample[1], False))

            self.encoding_conv.add(module2)
            
        self.to_dense = scn.Sequential().add(
           scn.SparseToDense(self._dimension,nPlanes[-1]))
        
        #Decoding TPC
        self.decoding_conv = scn.Sequential()
        for i in range(self.num_strides-2, -1, -1):
            module2 = scn.Sequential()
            if i < self.num_strides-1:
                module2.add(
                    scn.BatchNormLeakyReLU(nPlanes[i+1], leakiness=self.leakiness)).add(
                    scn.Deconvolution(self._dimension, nPlanes[i+1], nPlanes[i],
                    downsample[0], downsample[1], False))
                
            self.decoding_conv.add(module2)

        self.output = scn.Sequential().add(
           scn.BatchNormReLU(self.m)).add(
           scn.OutputLayer(self._dimension))
        
    def forward(self, data, clusts):
        # Use cluster ID as a batch ID, pass through CNN
        device = data.device
        cnn_data = torch.empty((0,5), device=device, dtype=torch.float)
        for i, c in enumerate(clusts):
            cnn_data = torch.cat((cnn_data, data[c,:5].float()))
            cnn_data[-len(c):,3] = i*torch.ones(len(c)).to(device)

            
        # We separate the coordinate tensor from the feature tensor
        coords = cnn_data[:, 0:self._dimension+1].float()
        batchs = cnn_data[:, self._dimension].float()
        features = cnn_data[:, self._dimension+1:].float()
        
        target = torch.cat((coords[:,:-1], features), 1)
    
        
        x = self.input((coords, features))
        
        initial_sparse = x
        
        #print("Initial size: ", x.spatial_size)

        # We send x through all the encoding layers
        feature_maps = [x]
        feature_ppn = [x]
        for i, layer in enumerate(self.encoding_conv):
            x = self.encoding_conv[i](x)
        #    print("Encoding: ", x.spatial_size)
        
        hidden_x = self.to_dense(x)
        hidden_x = hidden_x.view(-1,((hidden_x.size()[2]**3)*hidden_x.size()[1]))
        print("Hidden layer: ", hidden_x.size())
        print(hidden_x[0])
        #print(hidden_x)
        
        for i, layer in enumerate(self.decoding_conv):
            x = self.decoding_conv[i](x)
            #print("Decoding: ", x.spatial_size)
            #print(x)
        
        #print("Final size: ", x.spatial_size)
        
        return scn.compare_sparse(x, initial_sparse), hidden_x
        
        
        #x = self.output(x)
        #return x, target