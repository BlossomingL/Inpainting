# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author:XLin
# datetime:2020/3/15 0015 15:17
# software: PyCharm
import numpy as np
import cv2
import math
import os


def get_image_concancte(img_path1, img_path2):
    img_names1 = os.listdir(img_path1)
    img_names2 = os.listdir(img_path2)
    for name1, name2 in zip(img_names1, img_names2):
        img = np.zeros((164, ))
        path1 = os.path.join(img_path1, name1)
        path2 = os.path.join(img_path2, name2)
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)


def get_image_cut(img_path):
    img_names = os.listdir(img_path)
    for i, name in enumerate(img_names):
        if name.find('jpg') != -1:
            path = os.path.join(img_path, name)
            img = cv2.imread(path)
            # img = img[2:162, 326: 486, :]

            # img = img[2:162, 2:2+160, :]
            # img = img[2:162, 2+160+2:2+160+2+160, :]
            # img = img[2:162, 2+160+2+160+2:2+160+2+160+2+160, :]
            img = img[2:162, 2+160+2+160+2+160+2:2+160+2+160+2+160+2+160, :]

            # img = img[2:162, 164: 324, :]
            cv2.imwrite(os.path.join(img_path, 'image_0000' + str(i) + '.png'), img)


def get_mask(img_path, base, bias, mask_name):
    """
    提取mask
    :param img_path:
    :return:
    """
    img = cv2.imread(img_path)
    w, h, c = img.shape
    tmp = np.zeros([w, h])
    for i in range(w):
        for j in range(h):
            if base[0] - bias[0] <= img[i][j][0] <= base[0] + bias[0] and base[1] - bias[1] <= img[i][j][1] <= base[1] + bias[1] and base[2] - bias[2] <= img[i][j][2] <= base[2] + bias[2]:
                tmp[i][j] = 255
    cv2.imwrite(mask_name, tmp)
    return tmp


def create_center_mask(w, h, hole_rate, mask_path):
    tmp = [[0] * h for _ in range(w)]
    mask_two = np.asarray(tmp)

    hole_w, hole_h = math.ceil(w * hole_rate), math.ceil(h * hole_rate)
    hole_start_i, hole_start_j = math.ceil((h - hole_h) // 2 - 1), math.ceil((w - hole_w) // 2 - 1)

    for i in range(hole_start_i, hole_start_i + hole_h):
        for j in range(hole_start_j, hole_start_j + hole_w):
            mask_two[i][j] = 255
    cv2.imwrite(mask_path, mask_two)


def FMM(input_path, mask_path, hole_rate, my_mask=None, center=True):
    """
    FMM算法
    :param input_path: 未缺失图片
    :param output_shape: 输出大小
    :param mask_path: mask路径
    :return: 修复后图片
    """

    img = cv2.imread(input_path)
    w, h, _ = img.shape
    # img = cv2.resize(img, (h, w))
    # 如果选择了中心缺失模式
    if center:
        hole_w, hole_h = int(w * hole_rate), int(h * hole_rate)
        hole_start_i, hole_start_j = math.ceil((h - hole_h) / 2), math.ceil((w - hole_w) // 2)

        create_center_mask(w, h, hole_rate, mask_path)
        mask_three = np.ones([w, h, 3])
        incom = img[:]

        for i in range(hole_start_i - 1, hole_start_i - 1 + hole_h):
            for j in range(hole_start_j - 1, hole_start_j - 1 + hole_w):
                for k in range(3):
                    mask_three[i][j][k] = 0
                    incom[i][j][k] = 255
    else:
        cv2.imwrite(mask_path, my_mask)
        mask_three = np.ones([w, h, 3])
        for i in range(w):
            for j in range(h):
                if my_mask[i][j] == 255:
                    for k in range(3):
                        mask_three[i][j][k] = 0

    # tmp = [[0] * 128 for _ in range(128)]
    # mask_two = np.asarray(tmp)
    # for i in range(31, 31 + 64):
    #     for j in range(31, 31 + 64):
    #         mask_two[i][j] = 255
    # x_input = img * mask_three
    #
    cv2.imwrite('incomplete.jpg', incom)

    tmp = cv2.imread('7.jpg')
    cv2.imwrite('merge.jpg', img * mask_three + tmp * (1 - mask_three))

    # cv2.imwrite('test.jpg', x_input)

    mask_two = cv2.imread(mask_path, 0)
    x_input = cv2.imread('incomplete.jpg')

    # plt.subplot(221), plt.imshow(mask_two)
    # plt.title('degraded image')
    # x_input = cv2.imread('incomplete.jpg')
    if center:
        res = cv2.inpaint(x_input, mask_two, 10, cv2.INPAINT_TELEA)
    else:
        res = cv2.inpaint(x_input, mask_two, 50, cv2.INPAINT_NS)
    cv2.imwrite('res.jpg', res)
    return res


def main():
    # input_path = '3.png'
    # FMM(input_path, [128, 128], 'mask.jpg', 0.3)

    # for i in range(7):
    #     img = os.path.join('/home/linx/new_disk/Program/DeepLearning/Inpainting/my_inpaint/experiment_3_5', str(i) + '.jpg')
    #     mask_name = 'mask_0000' + str(i) + '.jpg'
    #     mask = get_mask(img, [98, 109, 129], [6, 6, 6], mask_name)
    #     origin = cv2.imread('origin.jpg')
    #     for m in range(160):
    #         for n in range(160):
    #             if mask[m][n] == 255:
    #                 for k in range(3):
    #                     origin[m][n][k] = 255
    #     cv2.imwrite('origin_white_'+ str(i) + '.jpg', origin)

    # FMM(img, 'mask.jpg', 0, mask, False)
    # img = 'origin.jpg'
    # FMM(img, 'mask_' + img, 0.5)

    get_image_cut('/home/linx/new_disk/Program/DeepLearning/Inpainting/my_inpaint/experiment_3_5')
    # create_center_mask(160, 160, 0.5, '1.png')


if __name__ == '__main__':
    # args = parser.parse_args()
    main()

