"""
# *Preliminary* pytorch implementation.
# VoxelMorph training.

initiated by JM Kim, Ph.D., MedicalIP, Inc.
===initial: 27-August-2021

=== added VoxelMorph by Monib Sediqi
=== VoxelMorph:  02-February-2022
"""

# local imports
from pytorch.build_model import BuildModel, BuildSTN
from data_preparation.create_data_3d import create_infer_data_loaders_3d
from data_preparation.utils import read_dicom_files, resize_data_volume
from ndutils import get_quiver_plot
from args import Args

# built-in imports
import os, warnings, logging, datetime, random
import gc
import glob

# library
import numpy as np
import torch
from torch.autograd import Variable
import nibabel

import matplotlib.pyplot as plt

def get_2d_quiver(flow_2d, sp_factor = 20):
    """
    flow_2d: Flow filed in 2d+3 format. Example (512,512,3)
    sp_factor = sparsity factor.
    """
    spatial_flow = flow_2d[:, :, 0:2]
    meshg = meshgridnd_like(spatial_flow[::sp_factor, ::sp_factor, 0])
    mesh = np.asarray(meshg)
    mesh_mv = np.moveaxis(mesh, 0, -1)
    meshX = mesh_mv[:, :, 0]
    meshY = mesh_mv[:, :, 1]


    flowX_2d = flow_2d[::sp_factor, ::sp_factor, 0]
    flowY_2d = flow_2d[::sp_factor, ::sp_factor, 1]
    flowZ_2d = flow_2d[::sp_factor, ::sp_factor, 2]

    fig, ax = plt.subplots(figsize=(10,10))

    ax.quiver(meshX, meshY, flowX_2d, flowY_2d, flowZ_2d )

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_aspect('equal')

    return fig

def meshgridnd_like(in_img, rng_func=range):
    new_shape = list(in_img.shape)
    all_range = [rng_func(i_len) for i_len in new_shape]
    return tuple([x_arr.swapaxes(0, 1) for x_arr in np.meshgrid(*all_range)])

def build_optim(args, params):
    """
        Build Optimizer, default = Adam
    """
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.SGD(params, lr=args.lr)
    return optimizer

def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = BuildModel(args)

    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(checkpoint['model'])
    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint, model, optimizer


def extract_and_save(args, model, stn, display_loader):
    model_VM = model
    model_VM.eval()
    STN = stn

    current_time = datetime.datetime.now().strftime('%Y_%m_%d')
    print("current time", current_time)

    Tensor = args.Tensor

    with torch.no_grad():
        for iter, (
        scan_A_torch, scan_B_torch, rescale_slope, rescale_intercept, scan_path_A, scan_path_B, folder_name_A,
        folder_name_B) in enumerate(display_loader):

            moving_scan = Variable(scan_A_torch.type(Tensor))
            fixed_scan = Variable(scan_B_torch.type(Tensor))

            flow = model_VM(moving_scan, fixed_scan)
            moved_scan = STN(moving_scan, flow)

            moved_scan = torch.squeeze(moved_scan).cpu().detach().numpy()
            flow_np = torch.squeeze(flow).cpu().detach().numpy()

            # denormalize
            sf = args.sf
            denorm_moved_scan = moved_scan * (rescale_slope.cpu().numpy()) + rescale_intercept.cpu().numpy()

            denorm_moved_scan = resize_data_volume(denorm_moved_scan, [int(denorm_moved_scan.shape[0] / sf),
                                                                       int(denorm_moved_scan.shape[1] / sf),
                                                                       int(denorm_moved_scan.shape[2] / sf)])

            flow_np = flow_np * (rescale_slope.cpu().numpy()) + rescale_intercept.cpu().numpy()

            print("denorm_moved_scan shape: ", denorm_moved_scan.shape)
            print("denorm_moved_scan slices: ", denorm_moved_scan.shape[2])

            flow_path = "flow_npy"
            if args.save_npy_flow == 1:
                flow_cp = np.copy(np.swapaxes(flow_np, 2, 1))
                os.makedirs(flow_path, exist_ok=True)
                np.save(flow_path + f"/flow_{folder_name_A[0]}.npy", flow_cp)

            save_path_m = os.path.join(args.moved_save_path, current_time, "[Moved]_" + folder_name_A[0])
            print(f"save path m: {save_path_m}")
            print(f"extract_and_save, scan_path_A: {scan_path_A}")
            print(f"extract_and_save, scan_path_B: {scan_path_B}")
            os.makedirs(save_path_m, exist_ok=True)

            save_path_f = os.path.join(args.flow_save_path, current_time, "[Flow]_" + folder_name_A[0])
            os.makedirs(save_path_f, exist_ok=True)

            figure = get_quiver_plot(flow_np)
            figure.savefig(f'{save_path_f}/01_3D_quiver_fig.png')
            print("quiver figure saved")

            if args.data_type == "dicom":
                print('Input Data: Dicom')
                stacked_dicom = read_dicom_files(scan_path_B[0])
                print('stacked dicom len', len(stacked_dicom))

                for i in range(len(stacked_dicom)):
                    dicom_slice = stacked_dicom[i]
                    moved_temp = denorm_moved_scan[:, :, i]
                    moved_temp[moved_temp < 0] = 0
                    moved_slice = moved_temp.astype(str(dicom_slice.pixel_array.dtype))
                    dicom_slice.PixelData = moved_slice.tobytes()

                    save_slices_path = os.path.join(save_path_m, f"MV_0000{i:03}.dcm")

                    dicom_slice.SeriesDescription = "[Research & Science] - Generated Data"
                    dicom_slice.save_as(save_slices_path)

                for fl in range(flow_np.shape[3]):
                    single_flow = flow_np[:, :, :, fl]
                    single_flow = np.moveaxis(single_flow, 0, -1)

                    single_flow_min = np.min(single_flow)
                    single_flow_max = np.max(single_flow)

                    single_flow = ((single_flow - single_flow_min) / (single_flow_max - single_flow_min))

                    save_flow_path = os.path.join(save_path_f, f"FL_0000{fl:03}.png")

                    plt.imsave(save_flow_path, single_flow, cmap='brg')

            else:  # the data type is nifti
                print('Input Data: nifti')
                nii_file_B = glob.glob(os.path.join(scan_path_B[0], "*.nii"))
                nifti_file_B = nibabel.load(nii_file_B[0])  # [0] -> expecting a single 3D nii file

                nifti_img_B = nibabel.Nifti1Image(denorm_moved_scan, nifti_file_B.affine, nifti_file_B.header)
                save_nifti_path = os.path.join(save_path_m, f"MV_00001.nii")
                nibabel.save(nifti_img_B, save_nifti_path)

                number_of_slices = denorm_moved_scan.shape[2]

                flow_np = np.swapaxes(flow_np, 1, 2)  # <- solves the swapped h, w of the flow
                for i in range(number_of_slices):
                    single_flow = flow_np[:, :, :, i]
                    single_flow = np.moveaxis(single_flow, 0, -1)

                    single_flow_min = np.min(single_flow)
                    single_flow_max = np.max(single_flow)

                    single_flow = ((single_flow - single_flow_min) / (single_flow_max - single_flow_min))

                    save_flow_path = os.path.join(save_path_f, f"FL_0000{i:03}.png")

                    plt.imsave(save_flow_path, single_flow, cmap='brg')

            print('Output saved! ')

def main(args):
    gc.collect()
    torch.cuda.empty_cache()
    print('Starting inference a previous checkpoint...')
    checkpoint, model_VM, optimizer_VM = load_model(args.checkpoint)
    STN = BuildSTN(args)
    args = checkpoint['args']
    display_loader = create_infer_data_loaders_3d(args)
    extract_and_save(args, model_VM, STN, display_loader)
    print(f"Inference completed! ")


def create_arg_parser():
    parser = Args()
    print('Finish === create_arg_parser ===')
    return parser


if __name__ =="__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    args = create_arg_parser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.device = 'cuda'
    args.Tensor = torch.cuda.FloatTensor

    if torch.cuda.is_available():
        print("GPU is working.")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
    else:
        print("GPU is not working. Running on CPU")

    logging.basicConfig(level=logging.INFO, filename=f"infer_log.txt")
    logger = logging.getLogger(__name__)

    main(args)
