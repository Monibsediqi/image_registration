

import argparse
import pathlib

class Args(argparse.ArgumentParser):

    """
    Defines global default parameters
    """

    def __init__(self, **overrides):

        """Args:
        **overrides (dict, optional): Keyword arguments used to override default argument values
        """
        super().__init__(formatter_class= argparse.ArgumentDefaultsHelpFormatter)

        self.add_argument('--interpn', type=str, default='nearest', help="STN interpolation mode")
        self.add_argument('--log_file_name', type=str, default="log_v1", help="log file name")
        self.add_argument('--use_mask', action='store_true',
                          help='use the segmentation mask to compute the dice in each epoch')
        self.add_argument('--sf', type=float, default=1,
                          help="Scale factor. Uses the same scale factor for down sample and upsample")
        self.add_argument('--patch_size', type=int, default=64, help="An integer defining the size of a patch")

        # Setting dataset params
        self.add_argument('--train_moving_data', type=pathlib.Path, help= "Path to the train aligned_liver data(.npz)")
        self.add_argument('--train_fixed_data', type=pathlib.Path, help="Path to the train fixed data (.npz)")

        self.add_argument('--val_moving_data', type=pathlib.Path, help="Path to the val aligned_liver data")
        self.add_argument('--val_fixed_data', type=pathlib.Path, help="Path to the val fixed data")

        self.add_argument('--moved_save_path_A', type=pathlib.Path, help="Path to save the aligned data")
        self.add_argument('--moved_save_path_B', type=pathlib.Path, help="Path to save the aligned data")
        self.add_argument('--flow_save_path_A', type=pathlib.Path, help="Path to save the flow (for debugging purpose) ")
        self.add_argument('--flow_save_path_B', type=pathlib.Path, help="Path to save the flow (for debugging purpose) ")

        # specify data format (dicom or nii)
        self.add_argument('--data_type', type=str, choices= ['dicom', 'nifti'], default="dicom", help = 'specify the type of input data. The output type is infered from the input type')

        # preprocess
        self.add_argument('--norm_method', type=str, default='z-score', help = "min-max or z-score")

        # setting random seed
        self.add_argument('--seed', default=42, type=int, help='Seed for random number generators')

        # setting debug param
        self.add_argument('--debug', default=0, type=int,
                          help='If turned on, prints out debug information on terminal')

        # setting up network structure
        self.add_argument('--inshape', type=list, default=[512, 512, 112], help='3 in case of 3d, 2 in case of 2d image registration')
        self.add_argument('--enc_nf', type=list, default=[32, 32, 32, 32], help='===to be inserted===')
        self.add_argument('--dec_nf', type=list, default=[32, 32, 32, 32, 32, 16], help='=== to be inserted ===')
        self.add_argument("--lambda", type=float, dest="reg_param", default=1.0,  # recommend 1.0 for ncc, 0.01 for mse
                            help="regularization parameter")
        self.add_argument('--full_size', type=bool, default=True, help='=== to be inserted ===')
        self.add_argument('--drop_prob', type=float, default=0.0, help='Dropout probability')
        self.add_argument('--switch_residualpath', type=float, default=0, help='U-Net Residual Path Option')

        # setting hypter-parameters
        self.add_argument('--batch_size', default=1, type=int, help='Mini batch size')
        self.add_argument('--epochs', type=int, default=1500, help='Number of training epochs')
        self.add_argument('--optim', type=str, default="Adam", help='Optimizer type [adam, sgd]')
        self.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        self.add_argument("--data_loss", type=str, dest="data_loss", default='ncc', help="data_loss: mse or ncc")

        # Params for reporting
        self.add_argument('--report_interval', type=int, default=5, help='Period of loss reporting')
        self.add_argument('--report_interval_epoch', type=int, default=5000, help='Period of loss reporting')

        # Hardware setup
        self.add_argument('--device', type=str, default='cuda',
                          help='Which device to train on. Set to "cuda" to use the GPU')
        self.add_argument('--op_sys', choices=['windows', 'mac', 'linux'], default='linux', help='additional')
        self.add_argument('--accel_method', type=str, default='gpu',
                          help='Which device to train on. Set to "cuda" to use the GPU')
        self.add_argument('--data_parallel', action='store_true',
                          help='If set, use multiple GPUs for data parallelism')

        # Setup path
        self.add_argument('--export_dir', type=pathlib.Path, help='Path to save model')
        self.add_argument('--exp_name', type=pathlib.Path, default='exp', help='Path where model and results should be saved')
        self.add_argument('--resume', action='store_true', help='If set, resume the training from a previous model checkpoint. ''"--checkpoint" should be set with this')
        self.add_argument('--checkpoint', type=str, help='Path to an existing checkpoint. Used along with "--resume"')

        self.set_defaults(**overrides)
