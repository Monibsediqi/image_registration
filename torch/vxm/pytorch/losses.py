"""
*Preliminary* pytorch implementation.
Losses for VoxelMorph
Revised by Monib Sediqi @ MedicalIP Inc
Data: 18 / Feb / 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class Gradient:

    def __init__(self, penalty='l1', loss_mult=None):

        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, flow):
        """
        @Desc: Smoothing the deformation field obtained from the voxelmorph model
        flow: deformation field

        """
        dy = torch.abs(flow[:, :, 1:, :, :]) - (flow[:, :, :-1, :, :])
        dx = torch.abs(flow[:, :, :, 1:, :]) - (flow[:, :, :, :-1, :])
        dz = torch.abs(flow[:, :, :, :, 1:]) - (flow[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        grad = (torch.mean(dy) + torch.mean(dx) + torch.mean(dz)) / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class MSE:

    def loss(self, y_true, y_pred):
        """
        Mean Squared Loss
        @ Desc: Use this loss when intensity distribution and local contrast of images are similar
        x: Fixed image volume
        y: Moved image volume
        """
        return torch.mean((y_true - y_pred) ** 2)


class Exp2NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        ndims = 3
        # set window size
        win = [9, 9, 9]
        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(y_pred.device)

        stride = (1, 1, 1)
        padding = (4, 4, 4)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        TT = y_true * y_true
        PP = y_pred * y_pred
        TP = y_true * y_pred

        T_sum = conv_fn(y_true, sum_filt, stride=stride, padding=padding)
        P_sum = conv_fn(y_pred, sum_filt, stride=stride, padding=padding)
        TT_sum = conv_fn(TT, sum_filt, stride=stride, padding=padding)
        PP_sum = conv_fn(PP, sum_filt, stride=stride, padding=padding)
        TP_sum = conv_fn(TP, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        T_hat = T_sum / win_size
        P_hat = P_sum / win_size

        cross = TP_sum - P_hat * T_sum - T_hat * P_sum + T_hat * P_hat * win_size
        T_var = TT_sum - 2 * T_hat * T_sum + T_hat * T_hat * win_size
        P_var = PP_sum - 2 * P_hat * P_sum + P_hat * P_hat * win_size

        cc = cross * cross / (T_var * P_var + 1e-5)

        return -torch.mean(cc)

class Dice:

    def loss(self, y_true, y_pred):

        ndims = 3
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)

        return -dice

def flow_jacdet(flow):

    vol_size = flow.shape[:-1]
    grid = np.stack(pynd.ndutils.volsize2ndgrid(vol_size), len(vol_size))
    J = np.gradient(flow + grid)

    dx = J[0]
    dy = J[1]
    dz = J[2]

    Jdet0 = dx[:,:,:,0] * (dy[:,:,:,1] * dz[:,:,:,2] - dy[:,:,:,2] * dz[:,:,:,1])
    Jdet1 = dx[:,:,:,1] * (dy[:,:,:,0] * dz[:,:,:,2] - dy[:,:,:,2] * dz[:,:,:,0])
    Jdet2 = dx[:,:,:,2] * (dy[:,:,:,0] * dz[:,:,:,1] - dy[:,:,:,1] * dz[:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet


import SimpleITK as sitk


class JacLoss():
    def __init__(self, weight=None, size_average=True):
        super(JacLoss, self).__init__()

    def loss(self, flow):
        flow = flow.cpu().detach().numpy()[0, ...]
        img_size = flow.shape
        print("Jac: ")

        xx = np.arange(img_size[2])
        yy = np.arange(img_size[1])
        zz = np.arange(img_size[3])

        grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)
        grid = np.transpose(grid, (3, 0, 1, 2))

        # Get vector map
        vector_map = flow + grid
        vector_map = np.stack((vector_map[1, :, :, :],
                               vector_map[0, :, :, :],
                               vector_map[2, :, :, :]), 3)

        flow = sitk.GetImageFromArray(vector_map, isVector=True)

        jacobian_det_volume = sitk.DisplacementFieldJacobianDeterminant(flow)
        jacobian_det_np_arr = sitk.GetArrayViewFromImage(jacobian_det_volume)

        volume_folding = jacobian_det_np_arr[jacobian_det_np_arr < 0]

        jac_loss = len(volume_folding) / (img_size[2] * img_size[1] * img_size[0])  # Volume folding ratio

        return jac_loss

# ---------- ------------ EXPERIMENTAL PART --------------------
"""
taken from https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942
"""
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

"""
 Coding rule: 
 add a exp_ or EXP prefix to indicate that its experimental  
"""

def mutualInformation(bin_centers,
                      sigma_ratio=0.5,  # sigma for soft MI. If not provided, it will be half of a bin length
                      max_clip=1,
                      crop_background=False,  # crop_background should never be true if local_mi is True
                      local_mi=False,
                      patch_size=1):
    """
    mutual information for image-image pairs.
    Author: Courtney Guo. See thesis https://dspace.mit.edu/handle/1721.1/123142
    """
    print("vxm:mutual information loss is experimental.", file=sts.stderr)

    if local_mi:
        return localMutualInformation(bin_centers, sigma_ratio, max_clip, patch_size)

    else:
        return globalMutualInformation(bin_centers, sigma_ratio, max_clip, crop_background)

def globalMutualInformation(bin_centers,
                            sigma_ratio=0.5,
                            max_clip=1,
                            crop_background=False):
    """
    Mutual Information for image-image pairs
    Building from neuron.losses.MutualInformationSegmentation()
    This function assumes that y_true and y_pred are both (batch_size x height x width x depth x nb_chanels)
    Author: Courtney Guo. See thesis at https://dspace.mit.edu/handle/1721.1/123142
    """
    print("vxm:mutual information loss is experimental.", file=sts.stderr)

    """ prepare MI. """
    vol_bin_centers = K.variable(bin_centers)
    num_bins = len(bin_centers)
    sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

    preterm = K.variable(1 / (2 * np.square(sigma)))

    def mi(y_true, y_pred):
        """ soft mutual info """
        y_pred = K.clip(y_pred, 0, max_clip)
        y_true = K.clip(y_true, 0, max_clip)

        if crop_background:
            # does not support variable batch size
            thresh = 0.0001
            padding_size = 20
            filt = tf.ones([padding_size, padding_size, padding_size, 1, 1])

            smooth = tf.nn.conv3d(y_true, filt, [1, 1, 1, 1, 1], "SAME")
            mask = smooth > thresh
            # mask = K.any(K.stack([y_true > thresh, y_pred > thresh], axis=0), axis=0)
            y_pred = tf.boolean_mask(y_pred, mask)
            y_true = tf.boolean_mask(y_true, mask)
            y_pred = K.expand_dims(K.expand_dims(y_pred, 0), 2)
            y_true = K.expand_dims(K.expand_dims(y_true, 0), 2)

        else:
            # reshape: flatten images into shape (batch_size, heightxwidthxdepthxchan, 1)
            y_true = K.reshape(y_true, (-1, K.prod(K.shape(y_true)[1:])))
            y_true = K.expand_dims(y_true, 2)
            y_pred = K.reshape(y_pred, (-1, K.prod(K.shape(y_pred)[1:])))
            y_pred = K.expand_dims(y_pred, 2)

        nb_voxels = tf.cast(K.shape(y_pred)[1], tf.float32)

        # reshape bin centers to be (1, 1, B)
        o = [1, 1, np.prod(vol_bin_centers.get_shape().as_list())]
        vbc = K.reshape(vol_bin_centers, o)

        # compute image terms
        I_a = K.exp(- preterm * K.square(y_true - vbc))
        I_a /= K.sum(I_a, -1, keepdims=True)

        I_b = K.exp(- preterm * K.square(y_pred - vbc))
        I_b /= K.sum(I_b, -1, keepdims=True)

        # compute probabilities
        I_a_permute = K.permute_dimensions(I_a, (0, 2, 1))
        pab = K.batch_dot(I_a_permute, I_b)  # should be the right size now, nb_labels x nb_bins
        pab /= nb_voxels
        pa = tf.reduce_mean(I_a, 1, keep_dims=True)
        pb = tf.reduce_mean(I_b, 1, keep_dims=True)

        papb = K.batch_dot(K.permute_dimensions(pa, (0, 2, 1)), pb) + K.epsilon()
        mi = K.sum(K.sum(pab * K.log(pab / papb + K.epsilon()), 1), 1)

        return mi

    def loss(y_true, y_pred):
        return -mi(y_true, y_pred)

    return loss

def localMutualInformation(bin_centers,
                           vol_size,
                           sigma_ratio=0.5,
                           max_clip=1,
                           patch_size=1):
    """
    Local Mutual Information for image-image pairs
    # vol_size is something like (160, 192, 224)
    This function assumes that y_true and y_pred are both (batch_sizexheightxwidthxdepthxchan)
    Author: Courtney Guo. See thesis at https://dspace.mit.edu/handle/1721.1/123142
    """
    print("vxm:mutual information loss is experimental.", file=sts.stderr)

    """ prepare MI. """
    vol_bin_centers = K.variable(bin_centers)
    num_bins = len(bin_centers)
    sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

    preterm = K.variable(1 / (2 * np.square(sigma)))

    def local_mi(y_true, y_pred):
        y_pred = K.clip(y_pred, 0, max_clip)
        y_true = K.clip(y_true, 0, max_clip)

        # reshape bin centers to be (1, 1, B)
        o = [1, 1, 1, 1, num_bins]
        vbc = K.reshape(vol_bin_centers, o)

        # compute padding sizes
        x, y, z = vol_size
        x_r = -x % patch_size
        y_r = -y % patch_size
        z_r = -z % patch_size
        pad_dims = [[0, 0]]
        pad_dims.append([x_r // 2, x_r - x_r // 2])
        pad_dims.append([y_r // 2, y_r - y_r // 2])
        pad_dims.append([z_r // 2, z_r - z_r // 2])
        pad_dims.append([0, 0])
        padding = tf.constant(pad_dims)

        # compute image terms
        # num channels of y_true and y_pred must be 1
        I_a = K.exp(- preterm * K.square(tf.pad(y_true, padding, 'CONSTANT') - vbc))
        I_a /= K.sum(I_a, -1, keepdims=True)

        I_b = K.exp(- preterm * K.square(tf.pad(y_pred, padding, 'CONSTANT') - vbc))
        I_b /= K.sum(I_b, -1, keepdims=True)

        I_a_patch = tf.reshape(I_a, [(x + x_r) // patch_size, patch_size, (y + y_r) // patch_size, patch_size,
                                     (z + z_r) // patch_size, patch_size, num_bins])
        I_a_patch = tf.transpose(I_a_patch, [0, 2, 4, 1, 3, 5, 6])
        I_a_patch = tf.reshape(I_a_patch, [-1, patch_size ** 3, num_bins])

        I_b_patch = tf.reshape(I_b, [(x + x_r) // patch_size, patch_size, (y + y_r) // patch_size, patch_size,
                                     (z + z_r) // patch_size, patch_size, num_bins])
        I_b_patch = tf.transpose(I_b_patch, [0, 2, 4, 1, 3, 5, 6])
        I_b_patch = tf.reshape(I_b_patch, [-1, patch_size ** 3, num_bins])

        # compute probabilities
        I_a_permute = K.permute_dimensions(I_a_patch, (0, 2, 1))
        pab = K.batch_dot(I_a_permute, I_b_patch)  # should be the right size now, nb_labels x nb_bins
        pab /= patch_size ** 3
        pa = tf.reduce_mean(I_a_patch, 1, keep_dims=True)
        pb = tf.reduce_mean(I_b_patch, 1, keep_dims=True)

        papb = K.batch_dot(K.permute_dimensions(pa, (0, 2, 1)), pb) + K.epsilon()
        mi = K.mean(K.sum(K.sum(pab * K.log(pab / papb + K.epsilon()), 1), 1))

        return mi

    def loss(y_true, y_pred):
        return -local_mi(y_true, y_pred)

    return loss