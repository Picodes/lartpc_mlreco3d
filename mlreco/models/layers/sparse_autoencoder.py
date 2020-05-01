from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch_geometric.nn import MetaLayer, NNConv
import sparseconvnet as scn


from torch.autograd import Function, Variable
from torch.nn import Module
import sparseconvnet
from sparseconvnet.utils import *
from sparseconvnet.sparseConvNetTensor import SparseConvNetTensor
from sparseconvnet.metadata import Metadata
from sparseconvnet.sequential import Sequential
from sparseconvnet.activations import Sigmoid
from sparseconvnet.networkInNetwork import NetworkInNetwork


class SparsifyFCS(Module):
    """
    Sparsify by looking at the first feature channel's sign.
    """
    def __init__(self, dimension):
        Module.__init__(self)
        self.dimension = dimension
    def forward(self, input):
        if input.features.numel():
            output = SparseConvNetTensor()
            output.metadata = Metadata(self.dimension)
            output.spatial_size = input.spatial_size
            active = input.features[:,0]>0
            output.features=input.features[active]
            active=active.type('torch.LongTensor')
            input.metadata.sparsifyMetadata(
                output.metadata,
                input.spatial_size,
                active.byte(),
                active.cumsum(0))
            return output
        else:
            return input


class EncoderLayer(torch.nn.Module):

    def __init__(self, cfg):
        super(EncoderLayer, self).__init__()

        # Get the model input parameters
        model_config = cfg

        # Take the parameters from the config
        self._dimension = model_config.get('dimension', 3)
        self.m =  model_config.get('feat_per_pixel', 4)
        self.nInputFeatures = model_config.get('input_feat_enc', 1)
        self.leakiness = model_config.get('leakiness_enc', 0)
        self.spatial_size = model_config.get('inp_spatial_size', 32) #Must be a power of 2
        self.feat_aug_mode = model_config.get('feat_aug_mode', 'custom')
        self.use_linear_output = model_config.get('use_linear_output', False)
        self.num_output_feats = model_config.get('num_output_feats', 64)
        
        self.encoder_only = model_config.get('encoder_mode', False)
        
        self.num_strides = model_config.get('num_stride', 4) #Layers until the size is 4**d
        
        self.kernel_size = model_config.get('kernel_size', 2)
        
        self._freeze = model_config.get('freeze', False)
        
        self.out_spatial_size = int(self.spatial_size/4**(self.num_strides-1))
        self.output = self.m*self.out_spatial_size**3
        
        if self.feat_aug_mode == 'linear':
            nPlanes = [self.m * i for i in range(1, self.num_strides + 1)]
        elif self.feat_aug_mode == 'custom':
            nPlanes = [4,8,16,16,16]
        elif self.feat_aug_mode == 'power':
            nPlanes = [self.m * pow(2, i) for i in range(self.num_strides)]
        elif self.feat_aug_mode != 'constant':
            raise ValueError('Feature augmentation mode not recognized')
        
        downsample = [self.kernel_size, 2]  # [filter size, filter stride]
        
        #Input for tpc voxels
        self.input = scn.Sequential().add(
           scn.InputLayer(self._dimension, self.spatial_size, mode=3))
        
        self.prepare = scn.Sequential().add(
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
                    scn.SubmanifoldConvolution(self._dimension, nPlanes[i], nPlanes[i],
                        3, False)).add(
                    scn.Convolution(self._dimension, nPlanes[i], nPlanes[i+1],
                        downsample[0], downsample[1], False))

            self.encoding_conv.add(module2)
            
        self.to_dense = scn.Sequential().add(
           scn.SparseToDense(self._dimension,nPlanes[-1]))
        
        if not self.encoder_only :
            
            #Decoding TPC
            self.decoding_conv1 = scn.Sequential()
            self.decoding_conv2 = scn.Sequential()
            self.sparsify = scn.Sequential().add(
                        SparsifyFCS(3))
            for i in range(self.num_strides-2, -1, -1):
                module1 = scn.Sequential()
                module2 = scn.Sequential()
                if i < self.num_strides-1:
                    module1.add(
                        scn.BatchNormLeakyReLU(nPlanes[i+1], leakiness=self.leakiness)).add(
                        scn.FullConvolution(self._dimension, nPlanes[i+1], nPlanes[i],
                        downsample[0], downsample[1], False)).add(
                        scn.SubmanifoldConvolution(self._dimension, nPlanes[i], nPlanes[i],
                            3, False))
                    module2.add(
                        SparsifyFCS(3)).add(
                        scn.SubmanifoldConvolution(self._dimension, nPlanes[i], nPlanes[i],
                            4, False))

                self.decoding_conv1.add(module1)
                self.decoding_conv2.add(module2)

            self.output = scn.Sequential().add(
               scn.SubmanifoldConvolution(self._dimension, self.m, self.nInputFeatures, 3, False))

        if self._freeze:
            for param in self.parameters():
                param.requires_grad = False
        
    def forward(self, cnn_data):
        # We separate the coordinate tensor from the feature tensor
        coords = cnn_data[:, 0:self._dimension+1].float()
        batchs = cnn_data[:, self._dimension].float()
        features = cnn_data[:, self._dimension+1:].float()
        
        target = torch.cat((coords[:,:-1], features), 1)
    
        
        x = self.input((coords, features))
        
        initial_sparse = x
        
        x = self.prepare(x)
        
        #print("Initial size: ", x.spatial_size)
        
        # We send x through all the encoding layers
        feature_maps = []
        feature_ppn = []
        for i, layer in enumerate(self.encoding_conv):
            feature_maps.append(x)
            x = self.encoding_conv[i](x)
        
        hidden_x = self.to_dense(x)
        
        hidden_x = hidden_x.view(-1,((hidden_x.size()[2]**3)*hidden_x.size()[1]))
        print(hidden_x.size())
        
        if self.encoder_only:
            return hidden_x
        
        dec_feature_maps = []
        dec_feature_sparse = []
        for i, layer in enumerate(self.decoding_conv1):
            x = self.decoding_conv1[i](x)
            #dec_feature_maps.append(x)
            dec_feature_maps.append(x)
            x = self.sparsify(x)
            dec_feature_sparse.append(x)
            x = self.decoding_conv2[i](x)
            #print("Decoding: ", x.spatial_size)
            #print(x)
        
        x = self.output(x)
        #print("Final size: ", x.spatial_size)
        
        return hidden_x, x, self.input((coords, features)), feature_maps, dec_feature_maps, dec_feature_sparse
        
        
        #x = self.output(x)
        #return x, target