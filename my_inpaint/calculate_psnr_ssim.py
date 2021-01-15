# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author:XLin
# datetime:2020/3/15 0015 20:28
# software: PyCharm
import cv2
import os
from skimage.measure import compare_ssim
import numpy as np
import math

sr_dir = os.listdir('./SR')
hr_dir = os.listdir('./HR')

psnr_value = 0.0
ssim = 0.0
n = 0


def to_grey(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


for hr_image in hr_dir:
    for sr_image in sr_dir:
        if sr_image == hr_image:
            if (sr_image[-3:]) != 'png':
                continue
            sr = cv2.imread('./SR/' + sr_image)
            h, w, _ = sr.shape
            hr = cv2.imread('./HR/' + hr_image)
            # hr.resize((128, 128, 3))
            # sr.resize((128, 128, 3))
            print(sr.shape, hr.shape)
            compute_psnr = cv2.PSNR(sr, hr)
            compute_ssim = compare_ssim(to_grey(sr), to_grey(hr))
            psnr_value += compute_psnr
            ssim += compute_ssim
            n += 1
            if n % 100 == 0:
                print("finish compute [%d/%d]" % (n, len(hr_dir)))
print(n)
psnr_value = psnr_value / n
ssim = ssim / n
print("average psnr = ", psnr_value)
print("average ssim = ", ssim)
