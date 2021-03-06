import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sparseconvnet as scn

from .lovasz import mean, lovasz_hinge_flat, StableBCELoss, iou_binary
from .misc import FocalLoss, WeightedFocalLoss
from collections import defaultdict


class MaskBCELoss(nn.Module):
    '''
    Loss function for Sparse Spatial Embeddings Model, with fixed
    centroids and symmetric gaussian kernels. 
    '''
    def __init__(self, cfg, name='clustering_loss'):
        super(MaskBCELoss, self).__init__()
        self.loss_config = cfg['modules'][name]
        self.seediness_weight = self.loss_config.get('seediness_weight', 1.0)
        self.embedding_weight = self.loss_config.get('embedding_weight', 10.0)
        self.smoothing_weight = self.loss_config.get('smoothing_weight', 1.0)
        self.spatial_size = self.loss_config.get('spatial_size', 512)
        self.loss_scheme = self.loss_config.get('loss_scheme', 'BCE')

        if self.loss_scheme == 'BCE':
            self.mask_loss = StableBCELoss()
        elif self.loss_scheme == 'lovasz_hinge':
            self.mask_loss = lovasz_hinge_flat
        elif self.loss_scheme == 'focal':
            raise NotImplementedError
        else:
            raise ValueError(
            'Invalid loss scheme: {}'.format(self.loss_scheme))
        
        # BCELoss for Embedding Loss
        self.bceloss = StableBCELoss()
        # L2 Loss for Seediness and Smoothing
        self.l2loss = torch.nn.MSELoss(reduction='mean')

    def find_cluster_means(self, features, labels):
        '''
        For a given image, compute the centroids mu_c for each
        cluster label in the embedding space.
        Inputs:
            features (torch.Tensor): the pixel embeddings, shape=(N, d) where
            N is the number of pixels and d is the embedding space dimension.
            labels (torch.Tensor): ground-truth group labels, shape=(N, )
        Returns:
            cluster_means (torch.Tensor): (n_c, d) tensor where n_c is the number of
            distinct instances. Each row is a (1,d) vector corresponding to
            the coordinates of the i-th centroid.
        '''
        clabels = labels.unique(sorted=True)
        cluster_means = []
        for c in clabels:
            index = (labels == c)
            mu_c = features[index].mean(0)
            cluster_means.append(mu_c)
        cluster_means = torch.stack(cluster_means)
        return cluster_means

    def get_per_class_probabilities(self, embeddings, margins, labels, coords):
        '''
        Computes binary foreground/background loss.
        '''
        loss = 0.0
        smoothing_loss = 0.0
        centroids = self.find_cluster_means(coords, labels)
        n_clusters = len(centroids)
        cluster_labels = labels.unique(sorted=True)
        probs = torch.zeros(embeddings.shape[0]).float().cuda()
        acc = 0.0

        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            mask = torch.zeros(embeddings.shape[0]).cuda()
            mask[index] = 1.0
            mask[~index] = 0.0
            sigma = torch.mean(margins[index], dim=0)
            dists = torch.sum(torch.pow(embeddings - centroids[i], 2), dim=1)
            p = torch.clamp(torch.exp(-dists / (2 * torch.pow(sigma, 2))), min=0, max=1)
            probs[index] = p[index]
            loss += self.bceloss(p, mask)
            acc += iou_binary(p > 0.5, mask)
            sigma_detach = sigma.detach()
            smoothing_loss += torch.sum(torch.pow(margins[index] - sigma_detach, 2))

        loss /= n_clusters
        smoothing_loss /= n_clusters
        acc /= n_clusters

        return loss, smoothing_loss, probs, acc

    def combine_multiclass(self, embeddings, margins, seediness, slabels, clabels, coords):
        '''
        Wrapper function for combining different components of the loss, 
        in particular when clustering must be done PER SEMANTIC CLASS. 

        NOTE: When there are multiple semantic classes, we compute the DLoss
        by first masking out by each semantic segmentation (ground-truth/prediction)
        and then compute the clustering loss over each masked point cloud. 

        INPUTS: 
            features (torch.Tensor): pixel embeddings
            slabels (torch.Tensor): semantic labels
            clabels (torch.Tensor): group/instance/cluster labels

        OUTPUT:
            loss_segs (list): list of computed loss values for each semantic class. 
            loss[i] = computed DLoss for semantic class <i>. 
            acc_segs (list): list of computed clustering accuracy for each semantic class. 
        '''
        loss = defaultdict(list)
        accuracy = defaultdict(float)
        semantic_classes = slabels.unique()
        for sc in semantic_classes:
            index = (slabels == sc)
            mask_loss, smoothing_loss, probs, acc = self.get_per_class_probabilities(
                embeddings[index], margins[index], clabels[index], coords[index])
            prob_truth = probs.detach()
            seed_loss = self.l2loss(prob_truth, seediness[index].squeeze(1))
            total_loss = self.embedding_weight * mask_loss \
                       + self.seediness_weight * seed_loss \
                       + self.smoothing_weight * smoothing_loss
            loss['loss'].append(total_loss)
            loss['mask_loss'].append(float(self.embedding_weight * mask_loss))
            loss['seed_loss'].append(float(self.seediness_weight * seed_loss))
            loss['smoothing_loss'].append(float(self.smoothing_weight * smoothing_loss))
            loss['mask_loss_{}'.format(int(sc))].append(float(mask_loss))
            loss['seed_loss_{}'.format(int(sc))].append(float(seed_loss))
            accuracy['accuracy_{}'.format(int(sc))] = acc
            
        return loss, accuracy

    def forward(self, out, segment_label, group_label):

        num_gpus = len(segment_label)
        loss = defaultdict(list)
        accuracy = defaultdict(list)

        for i in range(num_gpus):
            slabels = segment_label[i][:, -1]
            coords = segment_label[i][:, :3].float()
            if torch.cuda.is_available():
                coords = coords.cuda()
            slabels = slabels.int()
            clabels = group_label[i][:, -2]
            batch_idx = segment_label[i][:, 3]
            embedding = out['embeddings'][i]
            seediness = out['seediness'][i]
            margins = out['margins'][i]
            nbatch = batch_idx.unique().shape[0]

            for bidx in batch_idx.unique(sorted=True):
                embedding_batch = embedding[batch_idx == bidx]
                slabels_batch = slabels[batch_idx == bidx]
                clabels_batch = clabels[batch_idx == bidx]
                seed_batch = seediness[batch_idx == bidx]
                margins_batch = margins[batch_idx == bidx]
                coords_batch = coords[batch_idx == bidx] / self.spatial_size

                loss_class, acc_class = self.combine_multiclass(
                    embedding_batch, margins_batch, 
                    seed_batch, slabels_batch, clabels_batch, coords_batch)
                for key, val in loss_class.items():
                    loss[key].append(sum(val) / len(val))
                for s, acc in acc_class.items():
                    accuracy[s].append(acc)
                acc = sum(acc_class.values()) / len(acc_class.values())
                accuracy['accuracy'].append(acc)

        loss_avg = {}
        acc_avg = defaultdict(float)

        for key, val in loss.items():
            loss_avg[key] = sum(val) / len(val)
        for key, val in accuracy.items():
            acc_avg[key] = sum(val) / len(val)

        res = {}
        res.update(loss_avg)
        res.update(acc_avg)

        return res


class MaskBCELoss2(MaskBCELoss):
    '''
    Spatial Embeddings Loss with trainable center of attention.
    '''
    def __init__(self, cfg, name='clustering_loss'):
        super(MaskBCELoss2, self).__init__(cfg)

    def get_per_class_probabilities(self, embeddings, margins, labels, coords):
        '''
        Computes binary foreground/background loss.
        '''
        loss = 0.0
        smoothing_loss = 0.0
        centroids = self.find_cluster_means(embeddings, labels)
        n_clusters = len(centroids)
        cluster_labels = labels.unique(sorted=True)
        probs = torch.zeros(embeddings.shape[0]).float().cuda()
        acc = 0.0

        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            mask = torch.zeros(embeddings.shape[0]).cuda()
            mask[index] = 1.0
            mask[~index] = 0.0
            sigma = torch.mean(margins[index], dim=0)
            dists = torch.sum(torch.pow(embeddings - centroids[i], 2), dim=1)
            p = torch.clamp(torch.exp(-dists / (2 * torch.pow(sigma, 2))), min=0, max=1)
            probs[index] = p[index]
            loss += self.bceloss(p, mask)
            acc += iou_binary(p > 0.5, mask, per_image=False)
            sigma_detach = sigma.detach()
            smoothing_loss += torch.sum(torch.pow(margins[index] - sigma_detach, 2))

        loss /= n_clusters
        smoothing_loss /= n_clusters
        acc /= n_clusters

        return loss, smoothing_loss, probs, acc


class MaskBCELossBivariate(MaskBCELoss):
    '''
    Spatial Embeddings Loss with trainable center of attraction and
    bivariate gaussian probability kernels. 
    '''
    def __init__(self, cfg, name='clustering_loss'):
        super(MaskBCELossBivariate, self).__init__(cfg)

    def get_per_class_probabilities(self, embeddings, margins, labels, coords):
        '''
        Computes binary foreground/background loss.
        '''
        loss = 0.0
        smoothing_loss = 0.0
        centroids = self.find_cluster_means(embeddings, labels)
        n_clusters = len(centroids)
        cluster_labels = labels.unique(sorted=True)
        probs = torch.zeros(embeddings.shape[0]).float().cuda()
        acc = 0.0

        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            mask = torch.zeros(embeddings.shape[0]).cuda()
            mask[index] = 1.0
            mask[~index] = 0.0
            sigma = torch.mean(margins[index], dim=0)
            dists = torch.pow(embeddings - centroids[i], 2)
            dists = dists / (2 * torch.pow(sigma, 2))
            p = torch.clamp(torch.exp(-torch.sum(dists, dim=1)), min=0, max=1)
            probs[index] = p[index]
            loss += self.bceloss(p, mask)
            sigma_detach = sigma.detach()
            smoothing_loss += torch.sum(torch.pow(margins[index] - sigma_detach, 2))

        loss /= n_clusters
        smoothing_loss /= n_clusters
        acc /= n_clusters

        return loss, smoothing_loss, probs, acc


class MaskLovaszHingeLoss(MaskBCELoss2):
    '''
    Spatial Embeddings Loss using Lovasz Hinge for foreground/background
    segmentation and trainable center of attention. 
    '''
    def __init__(self, cfg, name='clustering_loss'):
        super(MaskLovaszHingeLoss, self).__init__(cfg)

    def get_per_class_probabilities(self, embeddings, margins, labels, coords):
        '''
        Computes binary foreground/background loss.
        '''
        loss = 0.0
        smoothing_loss = 0.0
        centroids = self.find_cluster_means(embeddings, labels)
        n_clusters = len(centroids)
        cluster_labels = labels.unique(sorted=True)
        probs = torch.zeros(embeddings.shape[0]).float().cuda()
        acc = 0.0

        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            mask = torch.zeros(embeddings.shape[0]).cuda()
            mask[index] = 1
            mask[~index] = 0
            sigma = torch.mean(margins[index], dim=0)
            dists = torch.sum(torch.pow(embeddings - centroids[i], 2), dim=1)
            p = torch.exp(-dists / (2 * torch.pow(sigma, 2)))
            probs[index] = p[index]
            loss += lovasz_hinge_flat(2 * p - 1, mask)
            sigma_detach = sigma.detach()
            smoothing_loss += torch.sum(torch.pow(margins[index] - sigma_detach, 2))

        loss /= n_clusters
        smoothing_loss /= n_clusters
        acc /= n_clusters

        return loss, smoothing_loss, probs, acc


class MaskLovaszInterLoss(MaskLovaszHingeLoss):

    def __init__(self, cfg, name='clustering_loss'):
        super(MaskLovaszInterLoss, self).__init__(cfg)
        self.inter_weight = self.loss_config.get('inter_weight', 1.0)
        self.norm = 2


    def regularization(self, cluster_means):
        '''
        Implementation of regularization loss in Discriminative Loss
        Inputs:
            cluster_means (torch.Tensor): output from find_cluster_means
        Returns:
            reg_loss (float): computed regularization loss (see paper).
        '''
        reg_loss = 0.0
        n_clusters, _ = cluster_means.shape
        for i in range(n_clusters):
            reg_loss += torch.norm(cluster_means[i, :] + 1e-8, p=self.norm)
        reg_loss /= float(n_clusters)
        return reg_loss


    def inter_cluster_loss(self, cluster_means, margin=0.2):
        '''
        Implementation of distance loss in Discriminative Loss.
        Inputs:
            cluster_means (torch.Tensor): output from find_cluster_means
            margin (float/int): the magnitude of the margin delta_d in the paper.
            Think of it as the distance between each separate clusters in
            embedding space.
        Returns:
            inter_loss (float): computed cross-centroid distance loss (see paper).
            Factor of 2 is included for proper normalization.
        '''
        inter_loss = 0.0
        n_clusters = len(cluster_means)
        if n_clusters < 2:
            # Inter-cluster loss is zero if there only one instance exists for
            # a semantic label.
            return 0.0
        else:
            for i, c1 in enumerate(cluster_means):
                for j, c2 in enumerate(cluster_means):
                    if i != j:
                        dist = torch.norm(c1 - c2 + 1e-8, p=self.norm)
                        hinge = torch.clamp(2.0 * margin - dist, min=0)
                        inter_loss += torch.pow(hinge, 2)
            inter_loss /= float((n_clusters - 1) * n_clusters)
            return inter_loss


    def get_per_class_probabilities(self, embeddings, margins, labels, coords):
        '''
        Computes binary foreground/background loss.
        '''
        loss = 0.0
        smoothing_loss = 0.0
        centroids = self.find_cluster_means(embeddings, labels)
        inter_loss = self.inter_cluster_loss(centroids)
        reg_loss = self.regularization(centroids)
        n_clusters = len(centroids)
        cluster_labels = labels.unique(sorted=True)
        probs = torch.zeros(embeddings.shape[0]).float().cuda()
        accuracy = 0.0

        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            mask = torch.zeros(embeddings.shape[0]).cuda()
            mask[index] = 1
            mask[~index] = 0
            sigma = torch.mean(margins[index], dim=0)
            dists = torch.sum(torch.pow(embeddings - centroids[i], 2), dim=1)
            p = torch.exp(-dists / (2 * torch.pow(sigma, 2)))
            probs[index] = p[index]
            loss += lovasz_hinge_flat(2 * p - 1, mask)
            accuracy += iou_binary(p > 0.5, mask, per_image=False)
            sigma_detach = sigma.detach()
            smoothing_loss += torch.sum(torch.pow(margins[index] - sigma_detach, 2))

        loss /= n_clusters
        smoothing_loss /= n_clusters
        accuracy /= n_clusters
        loss += inter_loss
        loss += reg_loss / n_clusters

        return loss, smoothing_loss, inter_loss, probs, accuracy


    def combine_multiclass(self, embeddings, margins, seediness, slabels, clabels, coords):
        '''
        Wrapper function for combining different components of the loss, 
        in particular when clustering must be done PER SEMANTIC CLASS. 

        NOTE: When there are multiple semantic classes, we compute the DLoss
        by first masking out by each semantic segmentation (ground-truth/prediction)
        and then compute the clustering loss over each masked point cloud. 

        INPUTS: 
            features (torch.Tensor): pixel embeddings
            slabels (torch.Tensor): semantic labels
            clabels (torch.Tensor): group/instance/cluster labels

        OUTPUT:
            loss_segs (list): list of computed loss values for each semantic class. 
            loss[i] = computed DLoss for semantic class <i>. 
            acc_segs (list): list of computed clustering accuracy for each semantic class. 
        '''
        loss = defaultdict(list)
        accuracy = defaultdict(float)
        semantic_classes = slabels.unique()
        for sc in semantic_classes:
            index = (slabels == sc)
            mask_loss, smoothing_loss, inter_loss, probs, acc = \
                self.get_per_class_probabilities(
                embeddings[index], margins[index], 
                clabels[index], coords[index])
            prob_truth = probs.detach()
            seed_loss = self.l2loss(prob_truth, seediness[index].squeeze(1))
            total_loss = self.embedding_weight * mask_loss \
                       + self.seediness_weight * seed_loss \
                       + self.smoothing_weight * smoothing_loss
            loss['loss'].append(total_loss)
            loss['mask_loss'].append(
                float(self.embedding_weight * mask_loss))
            loss['seed_loss'].append(
                float(self.seediness_weight * seed_loss))
            loss['smoothing_loss'].append(
                float(self.smoothing_weight * smoothing_loss))
            loss['inter_loss'].append(
                float(self.inter_weight * inter_loss))
            loss['mask_loss_{}'.format(int(sc))].append(float(mask_loss))
            loss['seed_loss_{}'.format(int(sc))].append(float(seed_loss))
            accuracy['accuracy_{}'.format(int(sc))] = acc
            
        return loss, accuracy


class MaskFocalLoss(MaskBCELoss2):
    '''
    Spatial Embeddings Loss with trainable center of attention.
    '''
    def __init__(self, cfg, name='clustering_loss'):
        super(MaskFocalLoss, self).__init__(cfg)
        self.bceloss = FocalLoss(logits=False)

    def get_per_class_probabilities(self, embeddings, margins, labels, coords):
        '''
        Computes binary foreground/background loss.
        '''
        loss = 0.0
        smoothing_loss = 0.0
        centroids = self.find_cluster_means(embeddings, labels)
        n_clusters = len(centroids)
        cluster_labels = labels.unique(sorted=True)
        probs = torch.zeros(embeddings.shape[0]).float().cuda()
        acc = 0.0

        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            mask = torch.zeros(embeddings.shape[0]).cuda()
            mask[index] = 1.0
            mask[~index] = 0.0
            sigma = torch.mean(margins[index], dim=0)
            dists = torch.sum(torch.pow(embeddings - centroids[i], 2), dim=1)
            logits = dists / (2 * torch.pow(sigma, 2))
            p = torch.clamp(torch.exp(-dists / (2 * torch.pow(sigma, 2))), min=0, max=1)
            probs[index] = p[index]
            loss += self.bceloss(p, mask)
            acc += iou_binary(p > 0.5, mask, per_image=False)
            sigma_detach = sigma.detach()
            smoothing_loss += torch.sum(torch.pow(margins[index] - sigma_detach, 2))

        loss /= n_clusters
        smoothing_loss /= n_clusters
        acc /= n_clusters

        return loss, smoothing_loss, probs, acc


class MaskWeightedFocalLoss(MaskFocalLoss):

    def __init__(self, cfg, name='clustering_loss'):
        super(MaskWeightedFocalLoss, self).__init__(cfg)
        self.bceloss = WeightedFocalLoss(logits=False)


class EllipsoidalKernelLoss(MaskLovaszInterLoss):

    def __init__(self, cfg, name='clustering_loss'):
        super(EllipsoidalKernelLoss, self).__init__(cfg)
        self.quaternion_weight = self.loss_config.get('quaternion_weight', 10.0)

    @staticmethod
    def geodesic_distance_S3(qArr, qmean):
        '''
        Computes geodesic distance of q1, q2 in SO(3) group. 
        '''
        qmeans = qmean.expand_as(qArr)
        dist = torch.abs(torch.sum(qArr * qmeans, dim=1))
        dist = 1 - dist
        return dist

    @staticmethod
    def average_quaternions(q, num_samples=100):
        '''
        Correct averaging scheme for quaternion orientation estimation.

        INPUTS:
            - q: (N x 4) Tensor of unit quaternions
        '''
        try:
            qtq = torch.mm(q.t(), q)
            qAvg = torch.symeig(qtq, eigenvectors=True)[1][-1]
        except:
            qAvg = q[0]
        return qAvg

    @staticmethod
    def generate_rotation_matrix(q):
        U = torch.zeros((3,3)).cuda()
        U[0,0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
        U[0,1] = 2*q[1]*q[2] - 2*q[0]*q[3]
        U[0,2] = 2*q[1]*q[3] + 2*q[0]*q[2]
        U[1,0] = 2*q[1]*q[2] + 2*q[0]*q[3]
        U[1,1] = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
        U[1,2] = 2*q[2]*q[3] - 2*q[0]*q[1]
        U[2,0] = 2*q[1]*q[3] - 2*q[0]*q[2]
        U[2,1] = 2*q[2]*q[3] + 2*q[0]*q[1]
        U[2,2] = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
        return U

    @staticmethod
    def generate_rotation_matrix_batch(qArr):
        UArr = torch.zeros((qArr.shape[0], 3, 3))
        U[:, 0, 0] = q[:, 0]**2 + q[:, 1]**2 - q[:, 2]**2 - q[:, 3]**2
        U[:, 0, 1] = 2*q[:, 1]*q[:, 2] - 2*q[:, 0]*q[:, 3]
        U[:, 0, 2] = 2*q[:, 1]*q[:, 3] + 2*q[:, 0]*q[:, 2]
        U[:, 1, 0] = 2*q[:, 1]*q[:, 2] + 2*q[:, 0]*q[:, 3]
        U[:, 1, 1] = q[:, 0]**2 - q[:, 1]**2 + q[:, 2]**2 - q[:, 3]**2
        U[:, 1, 2] = 2*q[:, 2]*q[:, 3] - 2*q[:, 0]*q[:, 1]
        U[:, 2, 0] = 2*q[:, 1]*q[:, 3] - 2*q[:, 0]*q[:, 2]
        U[:, 2, 1] = 2*q[:, 2]*q[:, 3] + 2*q[:, 0]*q[:, 1]
        U[:, 2, 2] = q[:, 0]**2 - q[:, 1]**2 - q[:, 2]**2 + q[:, 3]**2
        return UArr

    @staticmethod
    def generate_covariance_matrix(sigma, q, eps=1e-8):
        '''
        Generate covariance matrix from principal components <sigma> and
        unit quarternion q.
        '''
        var = torch.pow(sigma, 2)
        U = self.generate_rotation_matrix(q)
        D = torch.diag(1.0 / (var + eps))
        cov = torch.chain_matmul(U, D, U.t())
        return cov
        
    
    def get_per_class_probabilities(self, embeddings, margins, labels, coords):
        '''
        Computes binary foreground/background loss.
        '''
        loss = 0.0
        smoothing_loss = 0.0
        centroids = self.find_cluster_means(embeddings, labels)
        inter_loss = self.inter_cluster_loss(centroids)
        reg_loss = self.regularization(centroids)
        n_clusters = len(centroids)
        cluster_labels = labels.unique(sorted=True)
        probs = torch.zeros(embeddings.shape[0]).float().cuda()
        accuracy = 0.0

        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            # Generate Ground Truth Mask
            mask = torch.zeros(embeddings.shape[0]).cuda()
            mask[index] = 1.0
            mask[~index] = 0.0
            # Obtain Axes and rotation from network
            sigma = torch.mean(margins[index][:, :3], dim=0)
            quat = margins[index][:, 3:]
            quat = quat / torch.norm(quat, dim=1, keepdim=True)
            # Average quaternions using metric defined on SO(3)
            q = self.average_quaternions(quat)
            covInverse = self.generate_covariance_matrix(sigma, q)
            offset = embeddings - centroids[i]
            # Compute argument inside exponential
            dists = torch.mm(covInverse, offset.t()).t()
            dists = torch.bmm(
                offset.view(offset.shape[0], 1, -1), 
                dists.view(dists.shape[0], -1, 1))
            dists = dists.squeeze(2)
            p = torch.clamp(torch.exp(-dists), min=0, max=1).squeeze(1)
            probs[index] = p[index]
            loss += lovasz_hinge_flat(2.0 * p - 1.0, mask)
            accuracy += iou_binary(p > 0.5, mask, per_image=False)
            sigma_detach = sigma.detach()
            smoothing_loss += torch.sum(torch.pow(
                margins[index][:, :3] - sigma_detach, 2))
            qdetach = q.detach()
            smoothing_loss += self.quaternion_weight * torch.mean(
                self.geodesic_distance_S3(quat, qdetach))
            

        loss /= n_clusters
        smoothing_loss /= n_clusters
        accuracy /= n_clusters
        loss += inter_loss
        loss += reg_loss / n_clusters

        return loss, smoothing_loss, inter_loss, probs, accuracy