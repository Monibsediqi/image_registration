# local imports
from pytorch.model import cvpr2018_net as VoxelMorph
from pytorch.model import SpatialTransformer as STN

# built-in imports

# lib
import torch


class BuildModel(torch.nn.Module):

    def __init__(self, args):
        super(BuildModel, self).__init__()

        self.VoxelMorph = VoxelMorph(dim=args.dim,
                                     enc_nf=args.enc_nf,
                                     dec_nf=args.dec_nf,
                                     full_size=args.full_size)
        self.VoxelMorph = self.VoxelMorph.to(args.device)

    def forward(self, src, tgt):  # forward pass through the model
        return self.VoxelMorph(src, tgt)



class BuildSTN(torch.nn.Module):
    def __init__(self, args):
        super(BuildSTN, self).__init__()
        self.STN = STN(args.dim, args.interpn)
        self.STN = self.STN.to(args.device)

    def forward(self, src, flow):
        """
        # returns moved image
        """
        return self.STN(src, flow)