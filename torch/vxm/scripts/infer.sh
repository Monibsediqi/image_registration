#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python '/media/monib/ext1/work2022/voxelmorph_nets/voxelmorph_v02/infer.py' \
--val_moving_data '/media/monib/ext1/work2022/Base_Dataset/vm_data_affine_p2a/input/val/example_A' \
--val_fixed_data '/media/monib/ext1/work2022/Base_Dataset/vm_data_affine_p2a/input/val/example_B' \
--moved_save_path '/media/monib/ext1/work2022/Base_Dataset/vm_data_affine_p2a/inferred/moved_data/' \
--flow_save_path '/media/monib/ext1/work2022/Base_Dataset/vm_data_affine_p2a/inferred/flows/' \
--data_type 'dicom' \
--save_npy_flow 0 \
--checkpoint "/media/monib/ext1/work2022/voxelmorph_nets/DGX_ckps/voxelmorph_v02_docker/ckp_dgx_all_data_p2a/2022_04_15/best_model.pt"
