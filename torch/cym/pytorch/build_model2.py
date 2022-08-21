# local imports
from pytorch.model2 import cvpr2018_net as VoxelMorph

# lib
import torch


class BuildModel(torch.nn.Module):

    def __init__(self, args):
        super(BuildModel, self).__init__()

        self.VoxelMorph = VoxelMorph(vol_size=args.inshape,
                                     enc_nf=args.enc_nf,
                                     dec_nf=args.dec_nf,
                                     full_size=args.full_size)
        self.VoxelMorph = self.VoxelMorph.to(args.device)

    def forward(self, src, tgt):  # forward pass through the model
        return self.VoxelMorph(src, tgt)
