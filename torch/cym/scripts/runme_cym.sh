#!/bin/bash
CUDA_VISIBLE_DEVICES=1,2 python '/media/monib/ext1/work2022/voxelmorph_nets/voxelmorph_v02/train_val_cym.py' \
--train_moving_data '/media/monib/ext1/work2022/Base_Dataset/test/input/train/example_A' \
--train_fixed_data '/media/monib/ext1/work2022/Base_Dataset/test/input/train/example_B' \
--val_moving_data '/media/monib/ext1/work2022/Base_Dataset/test/input/val/example_A' \
--val_fixed_data '/media/monib/ext1/work2022/Base_Dataset/test/input/val/example_B' \
--moved_save_path_A '/media/monib/ext1/work2022/Base_Dataset/test/output_cy_morph/moved_A' \
--moved_save_path_B '/media/monib/ext1/work2022/Base_Dataset/test/output_cy_morph/moved_B' \
--flow_save_path_A '/media/monib/ext1/work2022/Base_Dataset/test/output_cy_morph/flows_A' \
--flow_save_path_B '/media/monib/ext1/work2022/Base_Dataset/test/output_cy_morph/flows_B' \
--export_dir '/media/monib/ext1/work2022/voxelmorph_nets/checkpoints/cyclemorph_patch' \
--exp_name "patch_based_training" \
--lr 0.001 \
--epochs 10 \
--batch_size 1 \
--debug 0 \
--lambda 0.01 \
--data_loss mse \
--data_parallel \
--data_type 'dicom' \
--patch_size 128 \
--log_file_name "log_cycle_patch_test" \


# recommend 1.0 for ncc, 0.01 for mse