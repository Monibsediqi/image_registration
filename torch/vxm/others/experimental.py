#import packages

import os
import numpy as np
import pydicom
import nibabel
import dicom2nifti
from tqdm import tqdm
from data_preparation.utils import read_dicom_files, ensure_multiples_of_16_v2
from ants import from_numpy, registration
from ndutils import crop_non_zero_3d

def nifti2dicom(arr_data, path_dicom, save_dir, index=0):
    """
    parameter:

        `arr_data`: numpy array that represents only one slice.
        `save_path`: The directory to save the slices
        `index`: the index of the slice, so this parameter will be used to put
        the name of each slice while using a for loop to convert all the slices
    """

    dicom_file = pydicom.dcmread(path_dicom)
    arr = arr_data.astype('uint16')
    dicom_file.Rows = arr.shape[0]
    dicom_file.Columns = arr.shape[1]
    dicom_file.PhotometricInterpretation = "MONOCHROME2"
    dicom_file.SamplesPerPixel = 1
    dicom_file.BitsStored = 16
    dicom_file.BitsAllocated = 16
    dicom_file.HighBit = 15
    dicom_file.PixelRepresentation = 1
    dicom_file.PixelData = arr.tobytes()
    dicom_file.save_as(os.path.join(save_dir, f'slice_00{index:04}.dcm'))


def nifti2dicom_v2(root_path, nifti_dir_path, ref_dicom_dir_path, debug= False):
    """
    params:
    nifti_dir_path: path to the dir where nifti files are stored
    ref_dicom_dir_path: path to reference dicom dir <- reference dicom dir is the dicom files that nifti files are made from
    save_dicom_path: path to save the converted nifti to dicom files

    Note: For each nifti file, there is one dicom dir.
    Note2: Read the corresponding meta data from each ref dicom data and store the nifti array to that dicom file

    """

    path_nifti_files = [os.path.join(nifti_dir_path, file_name) for file_name in os.listdir(nifti_dir_path)]

    # reference dicom files for each nifti files
    dicom_dir_paths = [os.path.join(ref_dicom_dir_path, single_dicom_dir) for single_dicom_dir in
                       (os.listdir(ref_dicom_dir_path))]
    path_dicom_files = [os.path.join(dicom_dir, os.listdir(dicom_dir)[0]) for dicom_dir in dicom_dir_paths]


    for i in tqdm(range(len(path_nifti_files))):
        if debug:
            print(f"{path_nifti_files[i]}, {path_dicom_files[i]}")

        save_path = path_nifti_files[i].strip('.nii').split('/')[-1]
        save_path = os.path.join(root_path, f"nii2dicom/{save_path}")

        os.makedirs(save_path, exist_ok=True)

        # read nifti file
        nifti_file = nibabel.load(path_nifti_files[i])
        nifti_array = nifti_file.get_fdata()
        nifti_array = np.swapaxes(nifti_array, 0, 1)

        nifti_slices = nifti_array.shape[2]

        for slice in range(nifti_slices):
            nifty_2d_array = nifti_array[:, :, slice]

            dicom_file = pydicom.dcmread(path_dicom_files[i])
            arr = nifty_2d_array.astype('uint16')
            dicom_file.Rows = arr.shape[0]
            dicom_file.Columns = arr.shape[1]
            dicom_file.PhotometricInterpretation = "MONOCHROME2"
            dicom_file.SamplesPerPixel = 1
            dicom_file.BitsStored = 16
            dicom_file.BitsAllocated = 16
            dicom_file.HighBit = 15
            dicom_file.PixelRepresentation = 1
            dicom_file.SeriesDescription = "R&S Data"
            dicom_file.PixelData = arr.tobytes()
            dicom_file.save_as(os.path.join(save_path, f'slice_00{slice:04}.dcm'))

    print('nifti2dicom conversion completed!')

def affine_registration(moving, fixed, reg_type = "AffineFast"):
    """
    Affine registration - rigid (rotation and translation) + scale - registration of a aligned_liver image to a fixed image

    """
    affine_reg = registration(fixed=fixed, moving=moving, type_of_transform=reg_type)
    return affine_reg


if __name__ == "__main__":


    # crop and pad
    dicom_dirs_path = "../exp_input_data/dump_data"
    save_path = '../exp_output'

    # name_of_dirs = sorted(os.listdir(dicom_dirs_path))
    # print(name_of_dirs)

    multiples_of_16 = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320,
                       336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624,
                       640, 656, 672, 688, 704, 720, 736, 752, 768, 784]

    dicom_stacks = read_dicom_files(dicom_dirs_path)

    img_shape = list(dicom_stacks[0].pixel_array.shape)

    img_shape.append(len(dicom_stacks))

    img_3d = np.zeros(img_shape)

    for j in range(len(dicom_stacks)):
        img_3d[:, :, j] = dicom_stacks[j].pixel_array

    save_dir_path = os.path.join(save_path, "dump_data")
    os.makedirs(save_dir_path, exist_ok=True)

    img_padded_3d = ensure_multiples_of_16_v2(img_3d)

    for j in range(img_padded_3d.shape[2]):
        if j < len(dicom_stacks):
            dicom_file = dicom_stacks[j]
            cropped_slice = img_padded_3d[:, :, j].astype('uint16')
            dicom_file.Rows = cropped_slice.shape[0]
            dicom_file.Columns = cropped_slice.shape[1]
            dicom_file.PixelData = cropped_slice.tobytes()
            dicom_file.SeriesDescription = "[Research & Science]"
            dicom_file.InstanceNumber = j
        else:
            dicom_file = dicom_stacks[len(dicom_stacks)-1] # Gets the meta data of the last slice
            cropped_slice = img_padded_3d[:, :, j].astype('uint16')
            dicom_file.Rows = cropped_slice.shape[0]
            dicom_file.Columns = cropped_slice.shape[1]
            dicom_file.PixelData = cropped_slice.tobytes()
            dicom_file.SeriesDescription = "[Research & Science]"
            dicom_file.InstanceNumber = j

        save_slices_path = os.path.join(save_dir_path, f"Slice_0000{j:03}.dcm")
        dicom_file.save_as(save_slices_path)
