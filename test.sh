#!/usr/bin/env bash

# Paris StreetView
# test baseline(center)
python main.py --mode test \
               --test_deepFillV1 \
               --batch_image /media/linx/dataset/Paris_StreetView_Dataset/test \
               --output_path /home/linx/new_disk/result/baseline \

# test baseline(random)
python main.py --mode test \
               --test_deepFillV1 \
               --batch_image /media/linx/dataset/Paris_StreetView_Dataset/test \
               --output_path /home/linx/new_disk/result/4-3/paris/baseline \
               --mask_mode random

# test baseline+GB
python main.py  --mode test \
                --test_deepFillV1_gradient_branch \
                --batch_image /media/linx/dataset/Paris_StreetView_Dataset/test \
                --output_path /home/linx/new_disk/result/baseline+GB \
                --checkpoint_dir_GB generative_inpainting_gradient_branch/model_logs/20201210220500246444_linx_paris_NORMAL_wgan_gp_full_model_paris_256

# test baseline+GB(random)
python main.py  --mode test \
                --test_deepFillV1_gradient_branch \
                --batch_image /media/linx/dataset/Paris_StreetView_Dataset/test \
                --output_path /home/linx/new_disk/result/4-3/paris/baseline+GB \
                --checkpoint_dir_GB generative_inpainting_gradient_branch/model_logs/20201210220500246444_linx_paris_NORMAL_wgan_gp_full_model_paris_256 \
                --mask_mode random

# compare_baseline_to_GB
python main.py  --mode concanate \
                --file1 /home/linx/new_disk/result/Paris/center/baseline \
                --file2 /home/linx/new_disk/result/Paris/center/baseline+GB \
                --final_file /home/linx/new_disk/result/Paris/center/compare_baseline_to_GB \
                --reverse

# compare_baseline_to_GB(random)
python main.py  --mode concanate \
                --file1 /home/linx/new_disk/result/4-3/paris/baseline \
                --file2 /home/linx/new_disk/result/4-3/paris/baseline+GB \
                --final_file /home/linx/new_disk/result/4-3/paris/compare_baseline_to_GB \
                --reverse


# ===================================================================================================================
# CelebA
python main.py --mode test \
               --test_deepFillV1 \
               --batch_image /media/linx/dataset/Paris_StreetView_Dataset/test \
               --output_path /home/linx/new_disk/result/baseline \
               --checkpoint_dir_CA /home/linx/new_disk/checkpoints/CelebA/baseline/20201204123003055400_4512-pc_celeba_NORMAL_wgan_gp_full_model_celeba_256 \
               --output_path /home/linx/new_disk/result/CelebA/128/baseline \

# test baseline+GB
python main.py  --mode test \
                --test_deepFillV1_gradient_branch \
                --batch_image /media/linx/dataset/Paris_StreetView_Dataset/test \
                --output_path /home/linx/new_disk/result/baseline+GB \
                --checkpoint_dir_GB generative_inpainting_gradient_branch/model_logs/20201210220500246444_linx_paris_NORMAL_wgan_gp_full_model_paris_256

# compare_baseline_to_GB
python main.py  --mode concanate \
                --file1 /home/linx/new_disk/result/baseline \
                --file2 /home/linx/new_disk/result/baseline+GB \
                --final_file /home/linx/new_disk/result/compare_baseline_to_GB \
                --reverse