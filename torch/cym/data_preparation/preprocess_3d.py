"""
initiated by JM Kim, Ph.D., MedicalIP, Inc.
===initial: 27-August-2021
===modified: 03-September-2021
===modified by HJ Chung: 05-November-2021
--------------------------------------------------------------------------
=== Modified by Monib Sediqi : 04-02-2022
    1. Added 3D preprocessing
    2. Ensure multiples of 16 - utils
    3. Volume resize - utils
    4. Fast Affine Alignment algorithm
"""

import torch
from torch.nn.functional import interpolate
import numpy as np
from data_preparation.utils import ensure_multiples_of_16_v2, resize_data_volume, fast_affine

class Preprocess_3D():

    def __init__(self, method='min-max', scale_factor = 1):
        """
            method : Normalization method
                - 'min-max' : min-max normalization
                - 'z-score' : z-score standardization
                - ' div10000' : Divide by 10,000 for NCC loss convergence
        """
        self.method = method
        self.scale_factor = scale_factor

    def __call__(self, scan_A_3D, scan_B_3D, scan_path_A, scan_path_B, folder_name_A, folder_name_B, mask_A_3D = None, mask_B_3D= None,):

        """
        scan_A: stack of dicom files - aligned scan
        scan_B: stack of dicom files - fixed scan
        """

        # print(f'before affine image A shape: {scan_A_3D.shape}, B shape: {scan_B_3D.shape}')
        #
        # affined, image_x, flag = fast_affine(scan_A_3D, scan_B_3D, verbose=False, reg_type="AffineFast")
        #
        # if flag == 0:
        #     scan_A_3D = affined
        #     scan_B_3D = image_x
        # elif flag == 1:
        #     scan_A_3D = image_x
        #     scan_B_3D = affined
        # print(f"affined A shape: {scan_A_3D.shape}, B shape: {scan_B_3D.shape}")

        sf = self.scale_factor # 0.5
        # scan_A_3D = interpolate(scan_A_3D, scale_factor=sf)
        # scan_B_3D = interpolate(scan_B_3D, scale_factor=sf)
        # scan_A_3D = resize_data_volume(scan_A_3D, [int(scan_A_3D.shape[0] * sf), int(scan_A_3D.shape[1] * sf),
        #                                               int(scan_A_3D.shape[2] * sf)])
        # scan_B_3D = resize_data_volume(scan_B_3D, [int(scan_B_3D.shape[0] * sf), int(scan_B_3D.shape[1] * sf),
        #                                               int(scan_B_3D.shape[2] * sf)])

        scan_A_3D_m16 = ensure_multiples_of_16_v2(image_3d=scan_A_3D)
        scan_B_3D_m16 = ensure_multiples_of_16_v2(image_3d=scan_B_3D)
        scan_A_3D = scan_A_3D_m16[np.newaxis, ...]
        scan_B_3D = scan_B_3D_m16[np.newaxis, ...]

        # ---------------------- Mask Part -------------------------
        if mask_A_3D is not None:
            mask_A_3D_m16 = ensure_multiples_of_16_v2(image_3d=mask_A_3D, is_mask=True)
            mask_B_3D_m16 = ensure_multiples_of_16_v2(image_3d=mask_B_3D, is_mask=True)

            mask_A_3D = mask_A_3D_m16[np.newaxis, ...]
            mask_B_3D = mask_B_3D_m16[np.newaxis, ...]

        if self.method == 'min-max':
            max_val = np.max(scan_A_3D)
            min_val = np.min(scan_A_3D)

            rescale_slope = max_val - min_val + 1e-13
            rescale_intercept = min_val

            image_3d_A = (scan_A_3D - rescale_intercept) / rescale_slope

            max_val = np.max(scan_B_3D)
            min_val = np.min(scan_B_3D)

            rescale_slope = max_val - min_val + 1e-13

            rescale_intercept = min_val

            image_3d_B = (scan_B_3D - rescale_intercept) / rescale_slope

        elif self.method == 'z-score':
            mean_val = np.mean(scan_A_3D)
            std_val = np.std(scan_A_3D)

            rescale_slope = std_val + 1e-13
            rescale_intercept = mean_val

            image_3d_A = (scan_A_3D - rescale_intercept) / rescale_slope

            mean_val = np.mean(scan_B_3D)
            std_val = np.std(scan_B_3D)

            rescale_slope = std_val + 1e-13
            rescale_intercept = mean_val

            image_3d_B = (scan_B_3D - rescale_intercept) / rescale_slope

        elif self.method == 'div10000':
            rescale_slope = 3000
            rescale_intercept = 0

            image_3d_A = (scan_A_3D - rescale_intercept) / rescale_slope
            image_3d_B = (scan_B_3D - rescale_intercept) / rescale_slope

        else:
            rescale_slope = 1
            rescale_intercept = 0

            image_3d_A = (scan_A_3D - rescale_intercept) / rescale_slope
            image_3d_B = (scan_B_3D - rescale_intercept) / rescale_slope

        scan_A_torch, scan_B_torch = torch.from_numpy(image_3d_A), torch.from_numpy(image_3d_B)

        if mask_A_3D is not None:
            mask_A_torch, mask_B_torch = torch.from_numpy(mask_A_3D), torch.from_numpy(mask_B_3D)
        else:
            dumpy_data = np.zeros([1,1,1])
            mask_A_torch, mask_B_torch = torch.from_numpy(dumpy_data), torch.from_numpy(dumpy_data)

        # returns image_B's rescale_slope and rescale_intercept
        return scan_A_torch, scan_B_torch, rescale_slope, rescale_intercept, scan_path_A, scan_path_B, folder_name_A, folder_name_B, mask_A_torch, mask_B_torch