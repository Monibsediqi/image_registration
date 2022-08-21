"""
initiated by JM Kim, Ph.D., MedicalIP, Inc.
===initial: 27-August-2021
--------------------------------------------------------------------------
=== Modified by Monib Sediqi : 04-02-2022
    1. Added 3D preprocessing
    2. Ensure multiples of 16 - utils
    3. Volume resize - utils
    4. Fast Affine Alignment algorithm
"""
import pathlib
import numpy as np
from torch.utils.data import Dataset
import torch
from data_preparation.preprocess_3d import Preprocess_3D
import nibabel
import os
from data_preparation.utils import read_dicom_files
import glob

class Image_3D_Data(Dataset):
    """
    Custom Pytorch Dataset
    __init__    : Define data path -> path to folder that contains dicom files
    __len__     : Define total number of dataset (in 3d total number of cases)
    __getitem__ : Get 3D dicom image -> return 3D normalized  pytorch  data
    """

    def __init__(self, root_A, root_B, m_root_A = None, m_root_B = None, preprocess_3d = None, data_type = None, scale_factor=None):
        """
        root_A : path to data A
        root_B : path to data B

        m_root_A = path to mask of data A
        m_root_B = path to mask of data B

        preprocess : pre-processing, see data.DataProcessing
        data_type : dicom or nifti file
        """
        self.preprocess_3d = preprocess_3d
        self.data_type = data_type

        # moving
        self.folders_A = list(pathlib.Path(root_A).iterdir())
        self.folders_A = sorted(self.folders_A)
        self.folder_A_names = os.listdir(pathlib.Path(root_A))
        self.folder_A_names = sorted(self.folder_A_names)

        # fixed
        self.folders_B = list(pathlib.Path(root_B).iterdir())
        self.folders_B = sorted(self.folders_B)

        self.folder_B_names = os.listdir(pathlib.Path(root_B))
        self.folder_B_names = sorted(self.folder_B_names)
        self.use_mask = False

        if m_root_A is not None and m_root_B is not None:
            print('--------------------------- using mask -------------------------')

            # moving mask
            self.m_folders_A = list(pathlib.Path(m_root_A).iterdir())
            self.m_folders_A = sorted(self.m_folders_A)
            self.m_folder_A_names = os.listdir(pathlib.Path(m_root_A))
            self.m_folder_A_names = sorted(self.m_folder_A_names)

            # fixed mask
            self.m_folders_B = list(pathlib.Path(m_root_B).iterdir())
            self.m_folders_B = sorted(self.m_folders_B)
            self.m_folder_B_names = os.listdir(pathlib.Path(m_root_B))
            self.m_folder_B_names = sorted(self.m_folder_B_names)

            self.use_mask = True

        for p in range(len(self.folders_A)):
            print(self.folders_A[p], self.folders_B[p])

            if self.use_mask:
                print(self.m_folders_A[p], self.m_folders_B[p])

    def __len__(self):
        return len(self.folders_A)

    def __getitem__(self, i):
        # return 3D data
        scan_path_A = str(self.folders_A[i])
        scan_path_B = str(self.folders_B[i])


        if self.data_type =='dicom':
            stacked_dicom_A = read_dicom_files(scan_path_A)  # stack of dicom files in a list

            image_shape_A = list(stacked_dicom_A[0].pixel_array.shape)
            image_shape_A.append(len(stacked_dicom_A))

            image_3d_A = np.zeros(image_shape_A)

            stacked_dicom_B = read_dicom_files(scan_path_B)
            image_shape_B = list(stacked_dicom_B[0].pixel_array.shape)
            image_shape_B.append(len(stacked_dicom_B))

            image_3d_B = np.zeros(image_shape_B)

            for j in range(len(stacked_dicom_A)):
                image_3d_A[:, :, j] = stacked_dicom_A[j].pixel_array
                image_3d_B[:, :, j] = stacked_dicom_B[j].pixel_array

            # -------------------------- Mask ------------------------
            if self.use_mask:
                stacked_dicom_A_mask = read_dicom_files(self.m_folders_A[i])  # stack of dicom files in a list

                mask_shape_A = list(stacked_dicom_A_mask[0].pixel_array.shape)
                mask_shape_A.append(len(stacked_dicom_A_mask))

                mask_3d_A = np.zeros(mask_shape_A)

                stacked_dicom_B_mask = read_dicom_files(self.m_folders_B[i])
                mask_shape_B = list(stacked_dicom_B_mask[0].pixel_array.shape)
                mask_shape_B.append(len(stacked_dicom_B_mask))

                mask_3d_B = np.zeros(mask_shape_B)

                for j in range(len(stacked_dicom_A_mask)):
                    mask_3d_A[:, :, j] = stacked_dicom_A_mask[j].pixel_array
                    mask_3d_B[:, :, j] = stacked_dicom_B_mask[j].pixel_array

        if self.data_type == "nifti":

            nii_file_A = glob.glob(os.path.join(scan_path_A, "*.nii"))
            nii_file_B = glob.glob(os.path.join(scan_path_B, "*.nii"))

            nifti_file_A = nibabel.load(nii_file_A[0])  # [0] -> expecting a single 3D nii file
            nifti_file_B = nibabel.load(nii_file_B[0])

            nifti_array_A = nifti_file_A.get_fdata()
            nifti_array_B = nifti_file_B.get_fdata()

            image_3d_A = nifti_array_A
            image_3d_B = nifti_array_B

        folder_name_A = self.folder_A_names[i]
        folder_name_B = self.folder_B_names[i]

        if self.use_mask:
            return self.preprocess_3d(image_3d_A, image_3d_B, scan_path_A, scan_path_B, folder_name_A, folder_name_B, mask_3d_A, mask_3d_B)
        else:
            return self.preprocess_3d(image_3d_A, image_3d_B, scan_path_A, scan_path_B, folder_name_A, folder_name_B, None, None)

if __name__ == "__main__":

    # train_A_data_path = '/media/monib/External Disk/work2022/voxelmorph_nets/voxelmorph_v02/input_data_liver/train/example_A'
    # train_B_data_path = '/media/monib/External Disk/work2022/voxelmorph_nets/voxelmorph_v02/input_data_liver/train/example_B'

    train_moving_data = '/media/monib/ext1/work2022/Base_Dataset/test/input/train/example_A'
    train_fixed_data = '/media/monib/ext1/work2022/Base_Dataset/test/input/train/example_B'
    train_moving_mask = '/media/monib/ext1/work2022/Base_Dataset/test/input/train/example_A_mask'
    train_fixed_mask = '/media/monib/ext1/work2022/Base_Dataset/test/input/train/example_B_mask'

    img_3d = Image_3D_Data(train_moving_data, train_fixed_data, train_moving_mask, train_fixed_mask, Preprocess_3D(method='z-score'), data_type='dicom')

    scan_A_torch, scan_B_torch, rescale_slope, rescale_intercept, scan_path_A, scan_path_B, folder_name_A, folder_name_B, mask_A_torch, mask_B_torch = img_3d[0]

    print(f"scan_A_torch: {scan_A_torch.shape}")
    print(f"scan_A_torch: {mask_A_torch.shape}")
    print(f"scan_A_torch min: {torch.min(scan_A_torch)}, max: {torch.max(scan_A_torch)}")
    print(f"mask_A_torch min: {torch.min(mask_A_torch)}, max: {torch.max(mask_A_torch)}")
    print(f"scan_B_torch: {scan_B_torch.shape}")
    print(f"rescale_slope val: {type(rescale_slope)}")
    print(f"rescale_intercept val: {rescale_intercept}")
    print(f"scan_path_A: {scan_path_A}") # path to each individual dicom slice
    print(f"scan_path_B: {scan_path_B}") # # path to each individual dicom slice
    print(f"folder_name_A: {folder_name_A}") # name of the folder (subject). e.g., ['HCC_1109_Pre']
    print(f"folder_name_B: {folder_name_B}")




















