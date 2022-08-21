#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python '/media/monib/ext1/work2022/voxelmorph_nets/voxelmorph_v02/train_val.py' \
--train_moving_data '/media/monib/ext1/work2022/Base_Dataset/vm_data_affine_d2p/train/example_A' \
--train_fixed_data '/media/monib/ext1/work2022/Base_Dataset/vm_data_affine_d2p/train/example_B' \
--val_moving_data '/media/monib/ext1/work2022/Base_Dataset/vm_data_affine_d2p/val/example_A' \
--val_fixed_data '/media/monib/ext1/work2022/Base_Dataset/vm_data_affine_d2p/val/example_B' \
--moved_save_path '/media/monib/ext1/work2022/Base_Dataset/vm_data_affine_d2p/output/moved' \
--flow_save_path '/media/monib/ext1/work2022/Base_Dataset/vm_data_affine_d2p/output/flows' \
--train_moving_mask '/media/monib/ext1/work2022/Base_Dataset/vm_data_affine_d2p/train/example_A_mask' \
--train_fixed_mask '/media/monib/ext1/work2022/Base_Dataset/vm_data_affine_d2p/train/example_B_mask' \
--val_moving_mask '/media/monib/ext1/work2022/Base_Dataset/vm_data_affine_d2p/val/example_A_mask' \
--val_fixed_mask '/media/monib/ext1/work2022/Base_Dataset/vm_data_affine_d2p/val/example_B_mask' \
--export_dir '/media/monib/ext1/work2022/voxelmorph_nets/voxelmorph_v02/ckp_d2p_all_ncc' \
--exp_name "ckp_d2p_all_ncc" \
--norm_method 'div10000' \
--lr 0.001 \
--epochs 500 \
--batch_size 1 \
--debug 0 \
--lambda 1.0 \
--data_loss ncc \
--data_parallel \
--data_type 'dicom' \
--save_npy_flow 0 \
--log_file_name "log_d2p_all_ncc" \
--report_interval 10 \
--sf 0.5



# recommend 1.0 for ncc, 0.01 for mse