# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author:XLin
# datetime:2020/3/25 0025 14:04
# software: PyCharm
import cv2


def split(img):
    k = 0
    for i in range(2):
        for j in range(8):
            sub_img = img[2 * (i + 1) + 128 * i : 2 * (i + 1) + 128 * (i + 1), 2 * (j + 1) + 128 * j : 2 * (j + 1) + 128 * (j + 1), : ]
            print(sub_img.shape)
            cv2.imwrite('split_img/' + str(k) + '.jpg', sub_img)
            k += 1

def split_4(img):
    for i in range(4):
        sub_img = img[2:2+160, 2*(i+1)+160*i : 2*(i+1)+160*(i + 1), :]

        cv2.imwrite('split_img/' + str(i + 1) + '.jpg', sub_img)

def main():
    img_path = '60a17f2c5ade450587b1fe262c694cf.jpg'
    img = cv2.imread(img_path)
    split_4(img)

if __name__ == '__main__':
    main()
