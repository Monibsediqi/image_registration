
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class Unet(nn.Module):
    """

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
        # encoder formation
        self.encoder = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i - 1]
            self.encoder.append(ConvBlock(dim, prev_nf, enc_nf[i], 2))


        # Decoder formation
        self.decoder = nn.ModuleList()
        self.decoder.append(ConvBlock(dim, enc_nf[-1], dec_nf[0]))
        self.decoder.append(ConvBlock(dim, dec_nf[0] * 2, dec_nf[1]))
        self.decoder.append(ConvBlock(dim, dec_nf[1] * 2, dec_nf[2]))
        self.decoder.append(ConvBlock(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))
        self.decoder.append(ConvBlock(dim, dec_nf[3], dec_nf[4]))

        if self.full_size:
            self.decoder.append(ConvBlock(dim, dec_nf[4] + 2, dec_nf[5], 1))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and aligned_liver image
        """
        # Get encoder activations
        x_enc = [x]
        for layer in self.encoder:
            x_enc.append(layer(x_enc[-1]))

        y = x_enc[-1]
        for i in range(3):
            y = self.decoder[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)

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

    def __init__(self, vol_size, enc_nf, dec_nf, full_size=True):
        """
        Instiatiate 2018 model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
        """
        super(cvpr2018_net, self).__init__()

        dim = len(vol_size)

        self.unet_model = Unet(dim, enc_nf, dec_nf, full_size)

        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.spatial_transform = SpatialTransformer(vol_size)

    def forward(self, src, tgt):
        """
        Pass input x through forward once
            :param src: aligned_liver image that we want to shift
            :param tgt: fixed image that we want to shift to
        """
        x = torch.cat([src, tgt], dim=1)
        x = self.unet_model(x)
        flow = self.flow(x)
        y = self.spatial_transform(src, flow)

        return y, flow


class SpatialTransformer(nn.Module):

    def __init__(self, input_shape, mode='bilinear'):
        """
        Instiatiate the block
            :param input_shape: size of input to the spatial transformer block (h,w,d)
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()
        self.dim = len(input_shape)

        self.grid = None
        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original aligned_liver image
            :param flow: the output from the U-Net
        """
        if self.dim == 3:
            batch, channel, height, width, slices= flow.size()
            size = [height, width, slices]
        else:
            batch, channel, height, width = flow.size()
            size = [height, width]
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)
        new_locs = grid + flow
        shape = flow.shape[2:]
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
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.

    [updated 24 Feb 2022] by Monib Sediqi
    """

    def __init__(self, dim, in_channels, out_channels, stride=1):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
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


