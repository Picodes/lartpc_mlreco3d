from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch_geometric.nn import MetaLayer, NNConv

class EncoderModel(torch.nn.Module):

    def __init__(self, cfg):
        super(EncoderModel, self).__init__()
        import sparseconvnet as scn

        # Get the model input parameters
        model_config = cfg

        # Take the parameters from the config
        self._dimension = model_config.get('dimension', 3)
        self.num_strides = model_config.get('num_stride', 4)
        self.m =  model_config.get('feat_per_pixel', 4)
        self.nInputFeatures = model_config.get('input_feat_enc', 4)
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
                        downsample[0], downsample[1], False)).add(
                    scn.AveragePooling(self._dimension, 2, 2))

            self.encoding_conv.add(module2)

        self.output = scn.Sequential().add(
           scn.SparseToDense(self._dimension,nPlanes[-1]))

        if self.use_linear_output:
            input_size = nPlanes[-1] * (self.out_spatial_size ** self._dimension)
            self.linear = torch.nn.Linear(input_size, self.num_output_feats)

    def forward(self, point_cloud):
        # We separate the coordinate tensor from the feature tensor
        coords = point_cloud[:, 0:self._dimension+1].float()
        features = torch.cat((point_cloud[:, 0:self._dimension].float(),point_cloud[:, self._dimension+1:].float()),1)

        x = self.input((coords, features))
        
        # We send x through all the encoding layers
        feature_maps = [x]
        feature_ppn = [x]
        for i, layer in enumerate(self.encoding_conv):
            x = self.encoding_conv[i](x)

        x = self.output(x)

        #Then we flatten the vector
        x = x.view(-1,(x.size()[2]*x.size()[2]*x.size()[2]*x.size()[1]))

        # Go through linear layer if necessary
        if self.use_linear_output:
            x = self.linear(x)
            x = x.view(-1, self.num_output_feats)

        return x
