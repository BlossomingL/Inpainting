import math
import numpy as np
from skimage import io
from skimage import measure
import matplotlib.pyplot as plt
import cv2

def create_center_mask(w, h, hole_rate):
    mask_three = np.ones([w, h, 3])
    hole_w, hole_h = int(w * hole_rate), int(h * hole_rate)
    hole_start_i, hole_start_j = math.ceil((h - hole_h) / 2), math.ceil((w - hole_w) // 2)
    for i in range(hole_start_i - 1, hole_start_i - 1 + hole_h):
        for j in range(hole_start_j - 1, hole_start_j - 1 + hole_w):
            for k in range(3):
                mask_three[i][j][k] = 0
    return mask_three


def main():
    # im = cv2.imread('5454/ours3/54-2.jpg')
    # im = cv2.resize(im, (167, 167))
    # cv2.imwrite('54-2.jpg', im)

    postfix = '3.jpg'
    ori_path = 'four_chapter/real/' + postfix
    best_inpaint_path = 'four_chapter/loss/' + postfix
    no_loss_path = 'four_chapter/noloss/' + postfix

    ori_img = io.imread(ori_path)
    best_img = io.imread(best_inpaint_path)
    no_loss_img = io.imread(no_loss_path)

    my_psnr = measure.compare_psnr(ori_img, best_img, 255)
    print('我的修复的psnr:', my_psnr)
    no_loss_psnr = measure.compare_psnr(ori_img, no_loss_img, 255)
    print('no loss修复的psnr:', no_loss_psnr)
    print('\n')


    my_ssim = measure.compare_ssim(ori_img, best_img, data_range=255, multichannel=True)
    print('我的修复的ssim:', my_ssim)
    no_loss_ssim = measure.compare_ssim(ori_img, no_loss_img, data_range=255, multichannel=True)
    print('no loss修复的ssim:', no_loss_ssim)
    print('\n')

    my_mse = measure.compare_nrmse(ori_img, best_img)
    print('我的修复的mse:', my_mse)
    no_loss_mse = measure.compare_nrmse(ori_img, no_loss_img)
    print('no loss修复的mse:', no_loss_mse)
    print('\n')


if __name__ == '__main__':
    main()
