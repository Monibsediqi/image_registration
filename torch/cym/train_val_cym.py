"""
# *Preliminary* pytorch implementation.
# VoxelMorph training.

initiated by JM Kim, Ph.D., MedicalIP, Inc.
===initial: 27-August-2021

=== added VoxelMorph by Monib Sediqi
=== VoxelMorph:  02-March-2022
"""
# # local imports
from pytorch.build_model import BuildModel, BuildSTN
from pytorch.losses import crossCorrelation3D, gradientLoss
from pytorch.losses import MSE

from data_preparation.create_data_3d import create_data_loaders_3d
from data_preparation.utils import read_dicom_files
from ndutils import get_quiver_plot, uniform_sampler_v2
from args import Args

# built-in imports
import os, warnings, logging, shutil, datetime, random
import gc, pathlib, time
from pathlib import Path
import glob

# library
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import nibabel

import itertools

import matplotlib.pyplot as plt


def build_optim(args, params):
    """
        Build Optimizer, default = Adam
    """
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.SGD(params, lr=args.lr)
    return optimizer


def save_model(args, export_dir, epoch, model_A, model_B, optimizer, best_value):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model_A': model_A.state_dict(),
            'model_B': model_B.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_value,
            'export_dir': export_dir
        },
        f=export_dir / 'model.pt'
    )

    shutil.copyfile(export_dir / 'model.pt', export_dir / 'best_model.pt')
    print(f"INFO: Model saved at: {export_dir}")


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model_A = BuildModel(args)
    model_B = BuildModel(args)

    if args.data_parallel:
        model_A = torch.nn.DataParallel(checkpoint['model_A'])
        model_B = torch.nn.DataParallel(checkpoint['model_B'])

    model_A.load_state_dict(checkpoint['model_A'])
    model_B.load_state_dict(checkpoint['model_B'])
    optimizer = build_optim(args, itertools.chain(model_A.parameters(), model_B.parameters()))
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint, model_A, model_B, optimizer


def train(args, epoch, model_A, model_B, STN, train_loader, optimizer, writer):

    print("Training...")
    Tensor = args.Tensor
    start_epoch_time = time.perf_counter()
    now = datetime.datetime.now()
    curr_time = now.strftime("%Y-%m-%d %H:%M:%S")

    model_A.train()
    model_B.train()

    T_losses = []

    criterionL2 = gradientLoss('l2')
    criterionCC = crossCorrelation3D(1, kernel=(9,9,9)) if args.data_loss == "ncc" else MSE().loss
    criterionCy = torch.nn.L1Loss()
    criterionId = crossCorrelation3D(1, kernel=(9,9,9)) if args.data_loss == "ncc" else MSE().loss

    for iter2, (scan_A_torch, scan_B_torch, _, _, _, _, _, _, mask_A_torch, mask_B_torch) in enumerate(train_loader):
        """
        scan_A_torch -> aligned_liver scan
        scan_B_torch -> fixed scan 
        """
        if args.use_patch:
            print("using patch!")

            image_A_patches, _ = uniform_sampler_v2(torch.squeeze(torch.squeeze(scan_A_torch)),
                                                    patch_size=args.patch_size)
            image_B_patches, _ = uniform_sampler_v2(torch.squeeze(torch.squeeze(scan_B_torch)),
                                                    patch_size=args.patch_size)

            print("Number of patches:", len(image_A_patches))

            for idx in range(len(image_A_patches)):
                moving_variable = Variable(image_A_patches[idx].type(Tensor))
                fixed_variable = Variable(image_B_patches[idx].type(Tensor))

                moving_patch = torch.unsqueeze(torch.unsqueeze(moving_variable, 0), 0)
                fixed_patch = torch.unsqueeze(torch.unsqueeze(fixed_variable, 0), 0)


                flow_A = model_A(moving_patch, fixed_patch)
                flow_B = model_B(fixed_patch, moving_patch)

                moved_patch_A = STN(moving_patch, flow_A)
                moved_patch_B = STN(fixed_patch, flow_B)

                lambda_ = 0.01 # NCC = 1.0, MSE = 0.01
                alpha   = 0.1
                beta    = 0.5
                #registration losses

                lossA_RC = criterionCC(moved_patch_A, fixed_patch)
                lossA_RL = criterionL2(flow_A) * lambda_

                lossB_RC = criterionCC(moved_patch_B, moving_patch)
                lossB_RL = criterionL2(flow_B) * lambda_

                # Cycle loss
                back_A, bflow_A = model_B(moved_patch_A, moved_patch_A)
                lossA_CY = criterionCy(back_A, moving_patch) * alpha

                back_B, bflow_B = model_A(moved_patch_B, moved_patch_A)
                lossB_CY = criterionCy(back_B, fixed_patch) * alpha

                # Identity loss
                idt_A, iflow_A = model_A(fixed_patch, fixed_patch)
                lossA_ID = criterionId(idt_A, fixed_patch) * beta

                idt_B, iflow_B = model_B(moving_patch, moving_patch)
                lossB_ID = criterionId(idt_B, moving_patch) * beta

                loss = lossA_RC + lossA_RL + lossB_RC + lossB_RL + lossA_CY + lossB_CY + lossA_ID + lossB_ID

                loss.backward()
                optimizer.step()

                writer.add_scalar("TrainLoss", loss, epoch)

                T_losses.append(loss.item())
        else:
            pass

    return np.mean(T_losses), time.perf_counter() - start_epoch_time


def validate(args, epoch,  model_A, model_B, STN, val_loader, writer):
    print("Validating ...")

    start_epoch_time = time.perf_counter()
    now = datetime.datetime.now()
    curr_time = now.strftime("%Y-%m-%d %H:%M:%S")

    model_A.eval()
    model_B.eval()

    Tensor = args.Tensor

    V_losses = []

    criterionL2 = gradientLoss('l2')
    criterionCC = crossCorrelation3D(1, kernel=(9, 9, 9)) if args.data_loss == "ncc" else MSE().loss
    criterionCy = torch.nn.L1Loss()
    criterionId = crossCorrelation3D(1, kernel=(9, 9, 9)) if args.data_loss == "ncc" else MSE().loss

    with torch.no_grad():

        for iter2, (scan_A_torch, scan_B_torch, _, _, _, _, _, _, mask_A_torch, mask_B_torch) in enumerate(
                val_loader):
            """
            scan_A_torch -> aligned_liver scan
            scan_B_torch -> fixed scan 
            """
            if args.use_patch:
                print("using patch!")

                image_A_patches, _ = uniform_sampler_v2(torch.squeeze(torch.squeeze(scan_A_torch)),
                                                        patch_size=args.patch_size)
                image_B_patches, _ = uniform_sampler_v2(torch.squeeze(torch.squeeze(scan_B_torch)),
                                                        patch_size=args.patch_size)

                print("Number of patches:", len(image_A_patches))

                for idx in range(len(image_A_patches)):
                    moving_variable = Variable(image_A_patches[idx].type(Tensor))
                    fixed_variable = Variable(image_B_patches[idx].type(Tensor))

                    moving_patch = torch.unsqueeze(torch.unsqueeze(moving_variable, 0), 0)
                    fixed_patch = torch.unsqueeze(torch.unsqueeze(fixed_variable, 0), 0)

                    flow_A = model_A(moving_patch, fixed_patch)
                    flow_B = model_B(fixed_patch, moving_patch)

                    moved_patch_B = STN(moving_patch, flow_A)
                    moved_patch_A = STN(moving_patch, flow_B)

                    lambda_ = 0.01  # NCC = 1.0, MSE = 0.01
                    alpha = 0.1
                    beta = 0.5
                    # registration losses

                    lossA_RC = criterionCC(moved_patch_B, fixed_patch)
                    lossA_RL = criterionL2(flow_A) * lambda_

                    lossB_RC = criterionCC(moved_patch_A, moving_patch)
                    lossB_RL = criterionL2(flow_B) * lambda_

                    # Cycle loss
                    back_A, bflow_A = model_B(moved_patch_B, moved_patch_A)
                    lossA_CY = criterionCy(back_A, moving_patch) * alpha

                    back_B, bflow_B = model_A(moved_patch_A, moved_patch_B)
                    lossB_CY = criterionCy(back_B, fixed_patch) * alpha

                    # Identity loss
                    idt_A, iflow_A = model_A(fixed_patch, fixed_patch)
                    lossA_ID = criterionId(idt_A, fixed_patch) * beta

                    idt_B, iflow_B = model_B(moving_patch, moving_patch)
                    lossB_ID = criterionId(idt_B, moving_patch) * beta

                    loss = lossA_RC + lossA_RL + lossB_RC + lossB_RL + lossA_CY + lossB_CY + lossA_ID + lossB_ID

                    writer.add_scalar("TrainLoss", loss, epoch)

                    V_losses.append(loss.item())
            else:
                pass

    return np.mean(V_losses), time.perf_counter() - start_epoch_time


def extract_and_save(args, model_A, model_B, STN, display_loader):
    print("extract and saving...")
    scale_factor = args.sf
    current_time = datetime.datetime.now().strftime('%Y_%m_%d')

    Tensor = args.Tensor

    model_A.eval()
    model_B.eval()

    with torch.no_grad():
        for iter, (scan_A_torch, scan_B_torch, rescale_slope, rescale_intercept, scan_path_A, scan_path_B, folder_name_A, folder_name_B, mask_A_torch, mask_B_torch) in enumerate(display_loader):

            moved_A = np.zeros(torch.squeeze(torch.squeeze(scan_A_torch)).cpu().detach().numpy().shape)
            flow_A = np.zeros((3, 512, 512, torch.squeeze(scan_A_torch).cpu().detach().numpy().shape[-1]))

            moved_B = np.zeros(torch.squeeze(torch.squeeze(scan_A_torch)).cpu().detach().numpy().shape)
            flow_B = np.zeros((3, 512, 512, torch.squeeze(scan_A_torch).cpu().detach().numpy().shape[-1]))

            if not args.use_patch:
                moving_variable = Variable(scan_A_torch.type(Tensor))
                fixed_variable = Variable(scan_B_torch.type(Tensor))

                moving_scan = torch.unsqueeze(torch.unsqueeze(moving_variable, 0), 0)
                fixed_scan = torch.unsqueeze(torch.unsqueeze(fixed_variable, 0), 0)

                flow_A = model_A(moving_scan, fixed_scan)
                flow_B = model_B(fixed_scan, moving_scan)

                moved_A = STN(moving_scan, flow_A)
                moved_B = STN(fixed_scan, flow_B)

            else:
                image_A_patches, locations_A = uniform_sampler_v2(torch.squeeze(torch.squeeze(scan_A_torch)),
                                                        patch_size=args.patch_size)
                image_B_patches, locations_B = uniform_sampler_v2(torch.squeeze(torch.squeeze(scan_B_torch)),
                                                        patch_size=args.patch_size)

                moved_patches_A = []
                flow_patches_A = []

                moved_patches_B = []
                flow_patches_B = []
                for idx in range(len(image_A_patches)):
                    moving_variable = Variable(image_A_patches[idx].type(Tensor))
                    fixed_variable = Variable(image_B_patches[idx].type(Tensor))

                    moving_patch = torch.unsqueeze(torch.unsqueeze(moving_variable, 0), 0)
                    fixed_patch = torch.unsqueeze(torch.unsqueeze(fixed_variable, 0), 0)

                    flow_patch_A = model_A(moving_patch, fixed_patch)
                    flow_patch_B = model_B(fixed_patch, moving_patch)

                    moved_patch_A = STN(moving_patch, flow_patch_A)
                    moved_patch_B = STN(fixed_patch, flow_patch_B)

                    moved_patches_A.append(torch.squeeze(moved_patch_A).cpu().detach().numpy())
                    flow_patches_A.append(torch.squeeze(flow_patch_A).cpu().detach().numpy())

                    moved_patches_B.append(torch.squeeze(moved_patch_B).cpu().detach().numpy())
                    flow_patches_B.append(torch.squeeze(flow_patch_B).cpu().detach().numpy())

                for loc_idx in range(len(locations_A)):
                    a_coor_A, b_coor_A = locations_A[loc_idx]
                    a_coor_B, b_coor_B = locations_B[loc_idx]
                    moved_A[a_coor_A:a_coor_A + args.patch_size, b_coor_A:b_coor_A + args.patch_size, ...] = \
                    image_A_patches[loc_idx]
                    flow_A[:, a_coor_A:a_coor_A + args.patch_size, b_coor_A:b_coor_A + args.patch_size, ...] = \
                    flow_patches_A[loc_idx]

                    moved_B[a_coor_B:a_coor_B + args.patch_size, b_coor_B:b_coor_B + args.patch_size, ...] = \
                        image_B_patches[loc_idx]
                    flow_B[:, a_coor_B:a_coor_B + args.patch_size, b_coor_B:b_coor_B + args.patch_size, ...] = \
                        flow_patches_B[loc_idx]

            """
            Denormalize moved scan and flow
            """
            denorm_moved_scan_A = moved_A * (rescale_slope.cpu().numpy()) + rescale_intercept.cpu().numpy()
            denorm_flow_np_A = flow_A * (rescale_slope.cpu().numpy()) + rescale_intercept.cpu().numpy()

            denorm_moved_scan_B = moved_B * (rescale_slope.cpu().numpy()) + rescale_intercept.cpu().numpy()
            denorm_flow_np_B = flow_B * (rescale_slope.cpu().numpy()) + rescale_intercept.cpu().numpy()

            print("denorm_moved_scan A shape: ", denorm_moved_scan_A.shape)
            print("flow_np_A shape: ", denorm_flow_np_A.shape)

            save_path_m_A = os.path.join(args.moved_save_path_A, current_time, "[Moved]_" + folder_name_A[0])
            save_path_m_B = os.path.join(args.moved_save_path_B, current_time, "[Moved]_" + folder_name_B[0])

            os.makedirs(save_path_m_A, exist_ok=True)
            os.makedirs(save_path_m_B, exist_ok=True)

            save_path_f_A = os.path.join(args.flow_save_path_A, current_time, "[Flow]_" + folder_name_A[0])
            save_path_f_B = os.path.join(args.flow_save_path_B, current_time, "[Flow]_" + folder_name_B[0])
            os.makedirs(save_path_f_A, exist_ok=True)
            os.makedirs(save_path_f_B, exist_ok=True)


            figure_A = get_quiver_plot(denorm_flow_np_A)
            figure_B = get_quiver_plot(denorm_flow_np_B)
            figure_A.savefig(f'{save_path_f_A}/01_3D_quiver_fig.png')
            figure_B.savefig(f'{save_path_f_B}/01_3D_quiver_fig.png')
            print("quiver figure saved")

            if args.data_type == "dicom":
                stacked_dicom_A = read_dicom_files(scan_path_B[0])
                stacked_dicom_B = read_dicom_files(scan_path_A[0])

                for i in range(len(stacked_dicom_A)):
                    dicom_slice_A = stacked_dicom_A[i]
                    dicom_slice_B = stacked_dicom_B[i]

                    moved_slice_A = denorm_moved_scan_A[:, :, i].astype(str(dicom_slice_A.pixel_array.dtype))
                    moved_slice_B = denorm_moved_scan_B[:, :, i].astype(str(dicom_slice_B.pixel_array.dtype))

                    dicom_slice_A.PixelData = moved_slice_A.tobytes()
                    dicom_slice_B.PixelData = moved_slice_B.tobytes()

                    save_slices_path_A = os.path.join(save_path_m_A, f"MV_0000{i:03}.dcm")
                    save_slices_path_B = os.path.join(save_path_m_B, f"MV_0000{i:03}.dcm")

                    dicom_slice_A.SeriesDescription = "[Research & Science] - Generated Data"
                    dicom_slice_B.SeriesDescription = "[Research & Science] - Generated Data"

                    dicom_slice_A.save_as(save_slices_path_A)
                    dicom_slice_B.save_as(save_slices_path_B)

                    single_flow_A = denorm_flow_np_A[:, :, :, i]
                    single_flow_B = denorm_flow_np_B[:, :, :, i]
                    single_flow_A = np.moveaxis(single_flow_A, 0, -1)
                    single_flow_B = np.moveaxis(single_flow_B, 0, -1)

                    single_flow_min_A = np.min(single_flow_A)
                    single_flow_max_A = np.max(single_flow_A)

                    single_flow_min_B = np.min(single_flow_B)
                    single_flow_max_B = np.max(single_flow_B)

                    single_flow_A = ((single_flow_A - single_flow_min_A) / (single_flow_max_A - single_flow_min_A))
                    single_flow_B = ((single_flow_B - single_flow_min_B) / (single_flow_max_B - single_flow_min_B))

                    save_flow_path_A = os.path.join(save_path_f_A, f"FL_0000{i:03}.png")
                    save_flow_path_B = os.path.join(save_path_f_B, f"FL_0000{i:03}.png")

                    plt.imsave(save_flow_path_A, single_flow_A, cmap ='brg')
                    plt.imsave(save_flow_path_B, single_flow_B, cmap ='brg')
                    plt.close()

            else: # the data type is nifti

                nii_file_B = glob.glob(os.path.join(scan_path_B[0], "*.nii"))
                nifti_file_B = nibabel.load(nii_file_B[0])

                nifti_img_B = nibabel.Nifti1Image(denorm_moved_scan_A, nifti_file_B.affine, nifti_file_B.header)
                save_nifti_path = os.path.join(save_path_m_A, f"MV_00001.nii")
                nibabel.save(nifti_img_B, save_nifti_path)

                number_of_slices = denorm_moved_scan_A.shape[2]
                for i in range(number_of_slices):
                    single_flow = denorm_flow_np_A[:, :, :, i]
                    single_flow = np.moveaxis(single_flow, 0, -1)

                    single_flow_min = np.min(single_flow)
                    single_flow_max = np.max(single_flow)

                    single_flow = ((single_flow - single_flow_min) / (single_flow_max - single_flow_min))

                    save_flow_path = os.path.join(save_path_f_A, f"FL_0000{i:03}.png")

                    plt.imsave(save_flow_path, single_flow, cmap='brg')
                    plt.close()

            print('Output saved! ')


def main(args):
    gc.collect()
    torch.cuda.empty_cache()

    args.export_dir.mkdir(parents=True, exist_ok=True)

    if args.op_sys == 'linux':
        tmp_export_dir = pathlib.PosixPath(
            os.path.join(args.export_dir, datetime.datetime.now().strftime('%Y_%m_%d')))

    if args.op_sys == 'windows':
        tmp_export_dir = Path(args.export_dir) / Path(args.exp_name) / datetime.datetime.now().strftime(
            '%Y_%m_%d')

    # tensorboard log dir
    os.makedirs(tmp_export_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tmp_export_dir / 'summary')

    if args.resume:
        print('Resuming from a previous checkpoint...')
        checkpoint, model_A, model_B, optimizer_VM = load_model(args.checkpoint)
        STN = BuildSTN(args)

        args = checkpoint['args']
        best_val_loss = checkpoint['epoch']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model_A = BuildModel(args)
        model_B = BuildModel(args)
        STN = BuildSTN(args)

        if args.data_parallel:
            print("INFO: Using Data parallel!")
            model_A = torch.nn.DataParallel(model_A).to(args.device)
            model_B = torch.nn.DataParallel(model_B).to(args.device)

        optimizer_VM = build_optim(args, itertools.chain(model_A.parameters(), model_B.parameters()))

        best_val_loss = 1e13
        start_epoch = 0

    logging.info(args)
    logging.info(model_A)
    logging.info(model_B)

    train_loader, val_loader, display_loader = create_data_loaders_3d(args)
    for epoch in range(start_epoch, args.epochs):

        train_loss, train_time = train(args, epoch, model_A, model_B, STN, train_loader, optimizer_VM, writer)

        val_loss, validation_time = validate(args, epoch, model_A, model_B, STN, val_loader, writer)

        is_new_best = val_loss < best_val_loss

        best_val_loss = min(best_val_loss, val_loss)

        if is_new_best:
            print(f"is new best: {is_new_best}")
            save_model(args, tmp_export_dir, epoch, model_A, model_B, optimizer_VM, best_val_loss)
            extract_and_save(args, model_A, model_B, STN, display_loader)

        logging.info(
            f" Epoch = [{epoch:4d}/{args.epochs}], Train Loss = {train_loss:.5g},"
            f" Val Loss = {val_loss: .5g}, Train duration = {train_time: .4f}s, Val duration = {validation_time:.4f}s"
        )
    print(f"Training completed! ")
    print(f"check the log file for details.")
    writer.close()


def create_arg_parser():
    parser = Args()
    print('Finish === create_arg_parser ===')
    return parser


if __name__ == "__main__":
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

    logging.basicConfig(level=logging.INFO, filename=args.log_file_name)
    logger = logging.getLogger(__name__)

    main(args)
