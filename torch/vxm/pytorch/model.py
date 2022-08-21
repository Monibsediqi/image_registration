"""
*Preliminary* pytorch implementation.

Networks for voxelmorph model

In general, these are fairly specific architectures that were designed for the presented papers.
However, the VoxelMorph concepts are not tied to a very particular architecture, and we 
encourage you to explore architectures that fit your needs. 
see e.g. more powerful unet function in https://github.com/adalca/neuron/blob/master/neuron/models.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

    encoder: [16, 32, 32, 32]
    decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, dim, enc_nf, dec_nf, full_size=True):
        """
        Instiatiate UNet model
            :param dim: dimension of the image passed into the net # 3
            :param enc_nf: the number of features maps in each layer of encoding stage #
            :param dec_nf: the number of features maps in each layer of decoding stage #
            :param full_size: boolean value representing whether full amount of decoding
                            layers

            dim = 2 for 2d imags, 3 for 3d images

        """
        super(Unet, self).__init__()

        self.full_size = full_size

        # -------------------- Start Encoder formation ------------------------
        self.encoder = nn.ModuleList()

        for i in range(len(enc_nf)):  # i = 0, 1, 2, 3 | final exp_output of encoder is 32 channels
            prev_nf = 2 if i == 0 else enc_nf[i - 1]
            self.encoder.append(ConvBlock(dim, prev_nf, enc_nf[i], 2))
        # --------------------- Encoder formation End -----------------------

        # Decoder formation
        self.decoder = nn.ModuleList()
        self.decoder.append(ConvBlock(dim, enc_nf[-1], dec_nf[0]))              # 1
        self.decoder.append(ConvBlock(dim, dec_nf[0] * 2, dec_nf[1]))           # 2
        self.decoder.append(ConvBlock(dim, dec_nf[1] * 2, dec_nf[2]))           # 3
        self.decoder.append(ConvBlock(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))   # 4
        self.decoder.append(ConvBlock(dim, dec_nf[3], dec_nf[4]))               # 5

        if self.full_size:
            self.decoder.append(ConvBlock(dim, dec_nf[4] + 2, dec_nf[5], 1))

        """
        IMPORTANT:  nn.Upsample() The input data is assumed to be of the form
        `minibatch x channels x [optional depth] x [optional height] x width`.
        Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.
        """
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # 2x upsampling (could use transpose)

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and aligned_liver image
        """
        # Get encoder activations
        x_enc = [x]
        for layer in self.encoder:
            x_enc.append(layer(x_enc[-1]))

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            y = self.decoder[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)   # skip connection

        # Two convs at full_size/2 res
        y = self.decoder[3](y)
        y = self.decoder[4](y)

        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.decoder[5](y)
        return y


class cvpr2018_net(nn.Module):
    """
    [cvpr2018_net] is a class representing the specific implementation for
    the 2018 implementation of voxelmorph.
    """

    def __init__(self, dim, enc_nf, dec_nf, full_size=True):
        """
        Instiatiate 2018 model
            :param dim: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
        """
        super(cvpr2018_net, self).__init__()

        self.unet_model = Unet(dim, enc_nf, dec_nf, full_size)

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)

        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # self.spatial_transform = SpatialTransformer(dim)

    def forward(self, src, tgt):
        """
        Pass input x through forward once
            :param src: aligned_liver image that we want to shift
            :param tgt: fixed image that we want to shift to
        """
        x = torch.cat([src, tgt], dim=1)
        x = self.unet_model(x)
        flow = self.flow(x)
        # y = self.spatial_transform(src, flow)

        return flow


class SpatialTransformer(nn.Module):

    def __init__(self, dim, mode='bilinear'):
        """
        Instiatiate the block
            :param input_shape: size of input to the spatial transformer block (h,w,d)
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()
        self.dim = dim
        self.grid = None
        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original aligned_liver image
            :param flow: the exp_output from the U-Net
        """
        if self.dim == 3:
            batch, channel, height, width, slices = flow.size()
            size = [height, width, slices]
        else:
            batch, channel, height, width = flow.size()
            size = [height, width]
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)
        new_locs = grid + flow
        shape = flow.shape[2:]
        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners =True, mode=self.mode)


class ConvBlock(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and exp_output
    channels and then preforms a Leaky Relu with parameter 0.2.

    [updated 24 Feb 2022] by Monib Sediqi
    ------------
    1. Add instance Normalization layer

    """

    def __init__(self, dim, in_channels, out_channels, stride=1):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of exp_output channels
            :param stride: stride of the convolution
        """
        super(ConvBlock, self).__init__()

        conv_fn = getattr(nn, f"Conv{dim}d")
        instance_norm = getattr(nn, f"InstanceNorm{dim}d")

        kernel_size = 3
        self.Conv = conv_fn(in_channels, out_channels, kernel_size, stride, 1)
        self.Norm = instance_norm(out_channels)
        self.Activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Pass the input through the conv_block
        """
        out = self.Conv(x)
        out = self.Norm(out)
        out = self.Activation(out)
        return out


# --------------------------------- Experimental part --------------------------

class ExpUnet(nn.Module):

    """
    A unet architecture. Layer architecture can be specified as a list of encoder and decoder or as a single integer along with a number of unet levels.
    The default layer features (when no option are specified) are:

    ecnocer: [16, 32,32,32]
    decoder: [32,32,32,32,32,16,16]
    """

    def __init__(self, dim, enc_nf, dec_nf, full_size = True):

        """
        params:
        input_shape: Input shape e.g., (512,512,112)
        input_features: Number of input features
        np_features: Unet's convolutional features. Can be specified via a list of lists with the form
        [[encoder_feats], [decoder_feats]]
        nb_levels: Number of levels in Unet. Only use when number of features is an integer
            Default: None
        feat_mult: Per-level feature multiplier. Only used when number of feature is an integer
            Default: 1
        nb_conv_per_level: Number of convs per unet level. Default is 1.
        half_res: Skip the last decoder upsampling. Default is false
        """
        super(ExpUnet, self).__init__()

        # Ensure correct dimensionality
        ndims = dim

        assert ndims in [1,2,3], "ndims must be one of 1, 2, or 3. found %d" %ndims

        # cache some params
        self.full_size = full_size

        if enc_nf is not None:
            self.enc_nf = enc_nf
        else:
            self.enc_nf = [16, 32, 32, 32]

        if dec_nf is not None:
            self.dec_nf = dec_nf
        else:
            self.dec_nf= [32,32,32,32,32,16,16]

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder
        nf_prev = 2
        self.encoder = nn.ModuleList()
        for nf in self.enc_nf:
            self.encoder.append(ExpConvBlock(ndims, nf, nf_prev, stride=2))
            nf_prev = nf


        # configure decoder
        self.decoder = nn.ModuleList()
        enc_history = list(reversed(self.enc_nf))
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = nf_prev + enc_history[i] if i > 0 else nf_prev
            self.decoder.append(ExpConvBlock(ndims, channels, nf, stride=1))
            nf_prev = nf

        # configure extra decoder convolutions (no up-sampling)
        nf_prev += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, nf_prev, nf, stride=1))
            nf_prev = nf

    def forward(self, x):

        x_enc = [x]
        for layer in self.encoder:
            x_enc.append(layer(x_enc[-1]))

            # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x


class ExpConvBlock(nn.Module):

    def __init__(self, ndims, in_channels, out_channels, stride = 1):
        super(ExpConvBlock, self).__init__()

        conv_fun = getattr(nn, 'Conv%dd' % ndims)
        instance_norm = getattr(nn, f"InstanceNorm{ndims}d")

        kernel = 3
        self.Conv = conv_fun(in_channels, out_channels, kernel, stride, 1)
        self.Norm = instance_norm(out_channels)
        self.Activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.Conv(x)
        out = self.Norm(out)      # added new
        out = self.Activation(out)
        return out

