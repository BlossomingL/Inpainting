# -*- coding: utf-8 -*-
# @Time    : 2020/12/19 上午6:37
# @Author  : LinX
# @File    : main.py

import argparse
from test import test_batch, concanate


def main():
    parser = argparse.ArgumentParser(description='For inpainting')

    parser.add_argument('--mode', default='concanate', choices=['train', 'test', 'concanate'], help='选择程序运行模式')

    # 如果选择测试
    parser.add_argument('--mask_mode', default='center', choices=['center', 'random'])
    parser.add_argument('--test_deepFillV1', default=False, action='store_true', help='是否测试deepFillV1')
    parser.add_argument('--test_deepFillV1_gradient_branch', default=False, action='store_true', help='是否测试deepFillV1+梯度分支网络')
    parser.add_argument('--test_deepFillV1_gradient_branch_SENet', default=False, action='store_true', help='是否测试deepFillV1+梯度分支网络+SENet')
    parser.add_argument('--test_PM', default=False, help='是否测试PM算法')
    parser.add_argument('--test_CE', default=False, help='是否测试CE算法')
    parser.add_argument('--image', default='/media/linx/dataset/Paris_StreetView_Dataset/test/097_im.png', type=str,
                        help='The filename of image to be completed.')
    parser.add_argument('--batch_image', default='/media/linx/dataset/Paris_StreetView_Dataset/test', help='批量测试')
    parser.add_argument('--rand_crop', default=True, help='')
    parser.add_argument('--mask', default='generative_inpainting-1.0.0/examples/center_mask_256.png', type=str,
                        help='The filename of mask, value 255 indicates mask.')
    parser.add_argument('--output', default='output.png', type=str,
                        help='Where to write output.')
    parser.add_argument('--output_path', default='/home/linx/new_disk/result/baseline+GB', help='')
    parser.add_argument('--checkpoint_dir_CA', default='generative_inpainting-1.0.0/model_logs/20201210070306295692_linx_paris_NORMAL_wgan_gp_full_model_paris_256', type=str,
                        help='The directory of tensorflow checkpoint.')
    parser.add_argument('--checkpoint_dir_GB', default='generative_inpainting_gradient_branch/model_logs/20201210220500246444_linx_paris_NORMAL_wgan_gp_full_model_paris_256', type=str,
                        help='The directory of tensorflow checkpoint.')
    parser.add_argument('--checkpoint_dir_SE', default='', type=str,
                        help='The directory of tensorflow checkpoint.')

    # 如果选择拼接图片
    parser.add_argument('--file1', default='/home/linx/new_disk/result/baseline', help='')
    parser.add_argument('--file2', default='/home/linx/new_disk/result/baseline+GB', help='')
    parser.add_argument('--final_file', default='/home/linx/new_disk/result/compare_baseline_to_GB', help='')
    parser.add_argument('--reverse', action='store_true', help='是否翻转')

    args = parser.parse_args()

    if args.mode == 'test':
        test_batch(args)

    elif args.mode == 'concanate':
        concanate(args)


if __name__ == '__main__':
    main()
