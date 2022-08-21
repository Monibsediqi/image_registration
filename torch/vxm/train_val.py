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
from pytorch.losses import Dice, Gradient, MSE, Exp2NCC, Dice
from data_preparation.create_data_3d import create_data_loaders_3d
from data_preparation.utils import read_dicom_files
from ndutils import get_quiver_plot, uniform_sampler_v2, save2dicom, save_flow
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

import matplotlib.pyplot as plt


def flowmap_patch(flowmap , patch_size):
    """
    flowmap: a 3D flow image
    patch_size: a tuple of int indicating the patch size. ex (128,128,128
    """
    stride = int(patch_size*6 / 8)

    flowmap_patches = []
    locations = []

    for i in range(0, flowmap.shape[2], stride):
        for j in range(0, flowmap.shape[3], stride):
            patch_img = flowmap[:,:,i:i + patch_size, j:j + patch_size, ...]
            flowmap_patches.append(patch_img)
            locations.append((i, j))

    return flowmap_patches, locations

def build_optim(args, params):
    """
        Build Optimizer, default = Adam
    """
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.SGD(params, lr=args.lr)
    return optimizer


def save_model(args, export_dir, epoch, model, optimizer, best_value):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
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
    model = BuildModel(args)

    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(checkpoint['model'])
    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint, model, optimizer


def train(args, epoch, model, STN, train_loader, optimizer, writer):
    print("Training...")
    Tensor = args.Tensor
    start_epoch_time = time.perf_counter()
    now = datetime.datetime.now()
    curr_time = now.strftime("%Y-%m-%d %H:%M:%S")
    model.train()

    T_combined = []
    T_data_loss = []
    T_flow_loss = []
    T_dice_loss = []

    # set losses
    sim_loss_fn = Exp2NCC().loss if args.data_loss == "ncc" else MSE().loss
    grad_loss_fn = Gradient().loss
    for iter2, (scan_A_torch, scan_B_torch, _, _, _, _, _, _, mask_A_torch, mask_B_torch) in enumerate(train_loader):
        """
        scan_A_torch -> moving scan
        scan_B_torch -> fixed scan 
        """
        if args.use_patch:
            print("using patch!")

            image_A_patches, _ = uniform_sampler_v2(torch.squeeze(torch.squeeze(scan_A_torch)), patch_size=args.patch_size)
            image_B_patches, _ = uniform_sampler_v2(torch.squeeze(torch.squeeze(scan_B_torch)), patch_size=args.patch_size)

            print("Number of patches:", len(image_A_patches))

            for idx in range(len(image_A_patches)):
                moving_patch = Variable(image_A_patches[idx].type(Tensor))
                fixed_patch = Variable(image_B_patches[idx].type(Tensor))

                moving_patch = torch.unsqueeze(torch.unsqueeze(moving_patch, 0), 0)
                fixed_patch = torch.unsqueeze(torch.unsqueeze(fixed_patch, 0), 0)

                flow = model(moving_patch, moving_patch)
                moved_patch = STN(moving_patch, flow)

                # Calculate loss
                recon_loss = sim_loss_fn(moved_patch, moved_patch)
                grad_loss = grad_loss_fn(flow) * args.reg_param

                combined_loss = recon_loss + grad_loss

                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()

                T_combined.append(combined_loss.detach().item())
                T_data_loss.append(recon_loss.detach().item())
                T_flow_loss.append(grad_loss.detach().item())

                # detach loss from graph after update
                combined_loss.detach()
        else:
            moving_scan = Variable(scan_A_torch.type(Tensor))
            fixed_scan = Variable(scan_B_torch.type(Tensor))

            flow = model(moving_scan, fixed_scan)
            moved_scan = STN(moving_scan, flow)

            dice_loss = 0
            if args.use_mask:
                # --------------------------- Mask part --------------------------
                moving_mask = Variable(mask_A_torch.type(Tensor))
                fixed_mask = Variable(mask_B_torch.type(Tensor))
                moved_mask = STN(moving_mask, flow)
                dice_loss = Dice().loss(fixed_mask, moved_mask)
                T_dice_loss.append(dice_loss.detach().item())

            recon_loss = sim_loss_fn(fixed_scan, moved_scan)
            grad_loss = grad_loss_fn(flow) * args.reg_param

            combined_loss = recon_loss + grad_loss + dice_loss

            optimizer.zero_grad(set_to_none=True)
            combined_loss.backward()
            optimizer.step()

            T_combined.append(combined_loss.detach().item())
            T_data_loss.append(recon_loss.detach().item())
            T_flow_loss.append(grad_loss.detach().item())

            # detach loss from graph after update
            combined_loss.detach()

        # # Logging
        # if iter2 % args.report_interval == 0:
        #     logging.info(f" Curr Time = [{curr_time}] Train Iter = {iter2}/{len(train_loader)}, Combined Loss: {np.mean(T_combined):.3g}, Data Loss: {np.mean(T_data_loss):.3g}, Flow Loss: {np.mean(T_flow_loss):.3g}")

    writer.add_scalar("TrainLoss", np.mean(T_combined), epoch)

    T_losses = {'combined_loss': np.mean(T_combined), 'data_loss': np.mean(T_data_loss), 'flow_loss': np.mean(T_flow_loss), "dice_loss": np.mean(T_dice_loss)}
    duration = time.perf_counter() - start_epoch_time
    return T_losses, duration


def validate(args, epoch, model, STN, val_loader, writer):
    print("validating...")
    model.eval()

    now = datetime.datetime.now()
    curr_time = now.strftime("%Y-%m-%d %H:%M:%S")

    Tensor = args.Tensor

    V_combined = []
    V_data_loss = []
    V_flow_loss = []
    V_dice_loss = []

    start_time = time.perf_counter()

    sim_loss_fn = Exp2NCC().loss if args.data_loss == "ncc" else MSE().loss
    grad_loss_fn = Gradient().loss

    with torch.no_grad():
        for iter2, (scan_A_torch, scan_B_torch, _, _, _, _, _,_, mask_A_torch, mask_B_torch) in enumerate(val_loader):

            if args.use_patch:
                print("using patch!")
                image_A_patches, _ = uniform_sampler_v2(torch.squeeze(torch.squeeze(scan_A_torch)),
                                                        patch_size=args.patch_size)
                image_B_patches, _ = uniform_sampler_v2(torch.squeeze(torch.squeeze(scan_B_torch)),
                                                        patch_size=args.patch_size)

                print("Number of patches:", len(image_A_patches))

                for idx in range(len(image_A_patches)):
                    moving_patch = Variable(image_A_patches[idx].type(Tensor))
                    fixed_patch = Variable(image_B_patches[idx].type(Tensor))

                    moving_patch = torch.unsqueeze(torch.unsqueeze(moving_patch, 0), 0)
                    fixed_patch = torch.unsqueeze(torch.unsqueeze(fixed_patch, 0), 0)

                    flow = model(moving_patch, moving_patch)
                    moved_patch = STN(moving_patch, flow)

                    # Calculate loss
                    recon_loss = sim_loss_fn(moving_patch, moved_patch)
                    grad_loss = grad_loss_fn(flow) * args.reg_param

                    combined_loss = recon_loss + grad_loss

                    V_combined.append(combined_loss.detach().item())
                    V_data_loss.append(recon_loss.detach().item())
                    V_flow_loss.append(grad_loss.detach().item())

                    # detach loss from graph after update
                    combined_loss.detach()
            else:
                moving_scan = Variable(scan_A_torch.type(Tensor))
                fixed_scan = Variable(scan_B_torch.type(Tensor))

                flow = model(moving_scan, fixed_scan)
                moved_scan = STN(moving_scan, flow)

                dice_loss = 0
                if args.use_mask:
                    # --------------------------- Mask part --------------------------
                    moving_mask = Variable(mask_A_torch.type(Tensor))
                    fixed_mask = Variable(mask_B_torch.type(Tensor))

                    moved_mask = STN(moving_mask, flow)

                    dice_loss = Dice().loss(fixed_mask, moved_mask)
                    V_dice_loss.append(dice_loss.detach().item())
                #     print("val dice value [moved mask and fixed mask]", dice_metric(moved_mask.cpu().detach().numpy(), fixed_mask.cpu().detach().numpy()))
                # print("val dice loss", dice_loss)

                recon_loss = sim_loss_fn(fixed_scan, moved_scan)
                grad_loss = grad_loss_fn(flow) * args.reg_param

                combined_loss = recon_loss + grad_loss + dice_loss

                V_combined.append(combined_loss.detach().item())
                V_data_loss.append(recon_loss.detach().item())
                V_flow_loss.append(grad_loss.detach().item())

                # detach loss from graph after update
                combined_loss.detach()

            # # Logging
            # if iter2 % args.report_interval == 0:
            #     logging.info(
            #         f" Curr Time = [{curr_time}] Val Iter = [{iter2} /{len(val_loader)}], Val Combined Loss: {np.mean(V_combined):.3g}")
            #     logging.info(f"Val Data Loss: {np.mean(V_data_loss):.3g}, Val Flow Loss: {np.mean(V_flow_loss):.3g}, Val Dice Loss {np.mean(dice_loss)}")
    writer.add_scalar('ValLoss', np.mean(V_combined), epoch)

    V_losses = {'combined_loss': np.mean(V_combined), "data_loss": np.mean(V_data_loss), "flow_loss": np.mean(V_flow_loss), "dice_loss": np.mean(V_dice_loss)}
    duration = time.perf_counter() - start_time
    return V_losses, duration


def extract_and_save(args, model, STN, display_loader):

    print("extract and saving...")
    scale_factor = args.sf
    current_time = datetime.datetime.now().strftime('%Y_%m_%d')

    global_flow_path_dir = "resize_global_flow_npy"

    Tensor = args.Tensor
    model.eval()
    with torch.no_grad():
        for iter, (scan_A_torch, scan_B_torch, rescale_slope, rescale_intercept, scan_path_A, scan_path_B, folder_name_A, folder_name_B, mask_A_torch, mask_B_torch) in enumerate(display_loader):

            # # ---------------------------------- Global + local part --------------------------------
            #
            #
            # global_flow_path = os.path.join(global_flow_path_dir, f"flow_{folder_name_A[0]}.npy")
            # global_flow = np.load(global_flow_path)
            #
            # padding_global_flow = np.zeros((1, 3, 528, 528, scan_A_torch.shape[-1]))
            # padding_global_flow[padding_global_flow == 0] = -1024 / 10000
            # padding_global_flow[:, :, 8:520, 8:520, :global_flow.shape[-1]] = global_flow
            # padding_global_flow = torch.from_numpy(padding_global_flow)
            # flowmap_patches, locations_C = flowmap_patch(padding_global_flow, patch_size=args.patch_size)
            #
            # #             print("flowmap_patches shape")
            # #             print(flowmap_patches[0].shape)
            # #             print(len(flowmap_patches))
            #
            # tmp_A = np.zeros((528, 528, scan_A_torch.shape[-1]))
            # tmp_A[tmp_A == 0] = -1024 / 10000
            # tmp_A[8:520, 8:520, ...] = scan_A_torch
            # scan_A_torch = tmp_A
            # tmp_B = np.zeros((528, 528, scan_B_torch.shape[-1]))
            # tmp_B[tmp_B == 0] = -1024 / 10000
            # tmp_B[8:520, 8:520, ...] = scan_B_torch
            # scan_B_torch = tmp_B
            # # print(scan_B_torch.shape)

            # scan_A_torch = torch.from_numpy(scan_A_torch)
            # scan_B_torch = torch.from_numpy(scan_B_torch)
            # ------------------------------------------ End ---------------------------------------

            moved_scan = np.zeros(torch.squeeze(torch.squeeze(scan_A_torch)).cpu().detach().numpy().shape)
            flow = np.zeros((3, 512, 512, torch.squeeze(scan_A_torch).cpu().detach().numpy().shape[-1]))

            if args.use_patch:
                print("using patch!")

                image_A_patches, locations_A = uniform_sampler_v2(torch.squeeze(torch.squeeze(scan_A_torch)),
                                                               patch_size=args.patch_size)
                image_B_patches, locations_B = uniform_sampler_v2(torch.squeeze(torch.squeeze(scan_B_torch)),
                                                               patch_size=args.patch_size)

                moved_patches = []
                flow_patches = []
                for idx in range(len(image_A_patches)):
                    moving_patch = Variable(image_A_patches[idx].type(Tensor))
                    fixed_patch = Variable(image_B_patches[idx].type(Tensor))

                    moving_patch = torch.unsqueeze(torch.unsqueeze(moving_patch, 0), 0)
                    fixed_patch = torch.unsqueeze(torch.unsqueeze(fixed_patch, 0), 0)

                    flow_A = model(moving_patch, moving_patch)
                    moved_patch = STN(moving_patch, flow_A)

                    moved_patches.append(torch.squeeze(moved_patch).cpu().detach().numpy())
                    flow_patches.append(torch.squeeze(flow_A).cpu().detach().numpy())

                # recon_moved = np.zeros(torch.squeeze(torch.squeeze(scan_A_torch)).cpu().detach().numpy().shape)
                # moved_scan[moved_scan == 0] = -1000
                # recon_flow = np.zeros((3, 512, 512, torch.squeeze(scan_A_torch).cpu().detach().numpy().shape[-1]))

                # Reconstruct the moved image from patches
                for loc_idx in range(len(locations_A)):
                    a_coor_A, b_coor_A = locations_A[loc_idx]
                    a_coor_B, b_coor_B = locations_B[loc_idx]
                    moved_scan[a_coor_A:a_coor_A + args.patch_size, b_coor_A:b_coor_A + args.patch_size, ...] = moved_patches[loc_idx]
                    flow[:, a_coor_A:a_coor_A + args.patch_size, b_coor_A:b_coor_A + args.patch_size, ...] = flow_patches[loc_idx]

            else:
                moving_scan = Variable(scan_A_torch.type(Tensor))
                fixed_scan = Variable(scan_B_torch.type(Tensor))

                # moved_scan, flow = model_VM(moving_scan, fixed_scan)
                flow = model(moving_scan, fixed_scan)
                moved_scan = STN(moving_scan, flow)

                moved_scan = torch.squeeze(moved_scan).cpu().detach().numpy()
                flow = torch.squeeze(flow).cpu().detach().numpy()

            # denormalize
            denorm_moved_scan = moved_scan * (rescale_slope.cpu().numpy()) + rescale_intercept.cpu().numpy()
            denorm_flow_np = flow * (rescale_slope.cpu().numpy()) + rescale_intercept.cpu().numpy()

            # denorm_moved_scan = resize_data_volume(denorm_moved_scan, [int(denorm_moved_scan.shape[0] / sf),
            #                                                            int(denorm_moved_scan.shape[1] / sf),
            #                                                            int(denorm_moved_scan.shape[2] / sf)])


            print("denorm_moved_scan shape: ", denorm_moved_scan.shape)
            print("flow_np shape: ", denorm_flow_np.shape)

            # flow_path = "flow_npy"
            # if args.save_npy_flow == 1:
            #     flow_cp = np.copy(np.swapaxes(flow_np, 2, 1 ))
            #     os.makedirs(flow_path, exist_ok=True)
            #     np.save(flow_path + f"/flow_{folder_name_A[0]}.npy", flow_cp)


            save_path_m = os.path.join(args.moved_save_path, current_time, "[Moved]_" + folder_name_A[0])
            print(f"save path moved: {save_path_m}")
            os.makedirs(save_path_m, exist_ok=True)
            
            save_path_f = os.path.join(args.flow_save_path, current_time, "[Flow]_" + folder_name_A[0])
            os.makedirs(save_path_f, exist_ok=True)


            figure = get_quiver_plot(denorm_flow_np)
            figure.savefig(f'{save_path_f}/01_3D_quiver_fig.png')
            print("quiver figure saved")

            if args.data_type == "dicom":
                print('[INFO]_Input Data: Dicom')
                stacked_dicom = read_dicom_files(scan_path_A[0])
                slices = denorm_moved_scan.shape[2] if denorm_moved_scan.shape[2] < len(stacked_dicom) else len(stacked_dicom)

                # Save to Dicom
                save2dicom(stacked_dicom, denorm_moved_scan, save_path_m)
                save_flow(denorm_flow_np, save_path_f)

            else: # the data type is nifti
                print('Input Data: nifti')
                nii_file_B = glob.glob(os.path.join(scan_path_B[0], "*.nii"))
                nifti_file_B = nibabel.load(nii_file_B[0])  # [0] -> expecting a single 3D nii file

                nifti_img_B = nibabel.Nifti1Image(denorm_moved_scan, nifti_file_B.affine, nifti_file_B.header)
                save_nifti_path = os.path.join(save_path_m, f"MV_00001.nii")
                nibabel.save(nifti_img_B, save_nifti_path)

                number_of_slices = denorm_moved_scan.shape[2]

                flow_np = np.swapaxes(flow_np, 1, 2) # <- solves the swapped h, w of the flow
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
        checkpoint, model_VM, optimizer_VM = load_model(args.checkpoint)
        STN = BuildSTN(args)

        args = checkpoint['args']
        best_val_loss = checkpoint['best_value']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model_VM = BuildModel(args)
        STN = BuildSTN(args)

        if args.data_parallel:
            print("INFO: Using Data parallel!")
            model_VM = torch.nn.DataParallel(model_VM).to(args.device)

        optimizer_VM = build_optim(args, model_VM.parameters())

        best_val_loss = 1e13
        start_epoch = 0

    logging.info(args)
    logging.info(model_VM)

    train_loader, val_loader, display_loader = create_data_loaders_3d(args)
    for epoch in range(start_epoch, args.epochs):

        train_loss, train_time = train(args, epoch, model_VM, STN, train_loader, optimizer_VM, writer)

        val_loss, validation_time = validate(args, epoch, model_VM, STN, val_loader, writer)

        is_new_best = val_loss['combined_loss'] < best_val_loss  # check this part for minus lost

        best_val_loss = min(best_val_loss, val_loss['combined_loss'])

        if is_new_best:
            print(f"is new best: {is_new_best}")
            save_model(args, tmp_export_dir, epoch, model_VM, optimizer_VM, best_val_loss)
            extract_and_save(args, model_VM, STN, display_loader)

        logging.info(f"Epoch = [{epoch:4d}/{args.epochs}]")
        logging.info(f" Train -> Combined Loss = {train_loss['combined_loss']:.3g}, Data Loss = {train_loss['data_loss']:3g}, Flow Loss = {train_loss['flow_loss']:3g}, Dice loss = {train_loss['dice_loss']:3g}")
        logging.info(f" Val -> Combined Loss = {val_loss['combined_loss']: .3g}, Data Loss = {val_loss['data_loss']: .3g}, Flow Loss = {val_loss['flow_loss']:.3g}, Dice loss = {val_loss['dice_loss']:.3g}")
        logging.info(f" Train duration = {train_time: .3f}s, Val duration = {validation_time:.3f}s")

    print(f"Training completed! ")
    print(f"check the log file for details.")
    writer.close()


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

    logging.basicConfig(level=logging.INFO, filename=f"../logs/{args.log_file_name}.txt")
    logger = logging.getLogger(__name__)

    main(args)