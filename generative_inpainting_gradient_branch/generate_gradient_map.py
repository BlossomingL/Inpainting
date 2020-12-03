# -*- coding: utf-8 -*-
# @Time    : 2020/12/1 下午3:25
# @Author  : LinX
# @File    : generate_gradient_map.py
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import tensorflow as tf


class Get_gradient_tf:
    def __init__(self):
        super(Get_gradient_tf, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]

        kernel_v = tf.convert_to_tensor(kernel_v, tf.float32)
        self.weight_v = tf.expand_dims(tf.expand_dims(kernel_v, 2), 3)

        kernel_h = tf.convert_to_tensor(kernel_h, tf.float32)
        self.weight_h = tf.expand_dims(tf.expand_dims(kernel_h, 2), 3)

    def get_gradient_tf(self, x):
        x0 = x[:, :, :, 0]
        x1 = x[:, :, :, 1]
        x2 = x[:, :, :, 2]
        # print(x.shape)
        # print(x0.shape)
        # print(x1.shape)
        # print(x2.shape)

        x0_v = tf.nn.conv2d(tf.expand_dims(x0, 3), self.weight_v, strides=[1, 1, 1, 1], padding='SAME')
        x0_h = tf.nn.conv2d(tf.expand_dims(x0, 3), self.weight_h, strides=[1, 1, 1, 1], padding='SAME')

        x1_v = tf.nn.conv2d(tf.expand_dims(x1, 3), self.weight_v, strides=[1, 1, 1, 1], padding='SAME')
        x1_h = tf.nn.conv2d(tf.expand_dims(x1, 3), self.weight_h, strides=[1, 1, 1, 1], padding='SAME')

        x2_v = tf.nn.conv2d(tf.expand_dims(x2, 3), self.weight_v, strides=[1, 1, 1, 1], padding='SAME')
        x2_h = tf.nn.conv2d(tf.expand_dims(x2, 3), self.weight_h, strides=[1, 1, 1, 1], padding='SAME')

        x0 = tf.sqrt(tf.square(x0_v) + tf.square(x0_h) + 1e-6)
        x1 = tf.sqrt(tf.square(x1_v) + tf.square(x1_h) + 1e-6)
        x2 = tf.sqrt(tf.square(x2_v) + tf.square(x2_h) + 1e-6)

        x = tf.concat([x0, x1, x2], 3)
        return x


class Get_gradient_tf_nopadding:
    def __init__(self):
        super(Get_gradient_tf_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]

        kernel_v = tf.convert_to_tensor(kernel_v, tf.float32)
        self.weight_v = tf.expand_dims(tf.expand_dims(kernel_v, 2), 3)

        kernel_h = tf.convert_to_tensor(kernel_h, tf.float32)
        self.weight_h = tf.expand_dims(tf.expand_dims(kernel_h, 2), 3)

    def get_gradient_tf(self, x):
        x0 = x[:, :, :, 0]
        x1 = x[:, :, :, 1]
        x2 = x[:, :, :, 2]
        # print(x.shape)
        # print(x0.shape)
        # print(x1.shape)
        # print(x2.shape)

        x0_v = tf.nn.conv2d(tf.expand_dims(x0, 3), self.weight_v, strides=[1, 1, 1, 1], padding='VALID')
        x0_h = tf.nn.conv2d(tf.expand_dims(x0, 3), self.weight_h, strides=[1, 1, 1, 1], padding='VALID')

        x1_v = tf.nn.conv2d(tf.expand_dims(x1, 3), self.weight_v, strides=[1, 1, 1, 1], padding='VALID')
        x1_h = tf.nn.conv2d(tf.expand_dims(x1, 3), self.weight_h, strides=[1, 1, 1, 1], padding='VALID')

        x2_v = tf.nn.conv2d(tf.expand_dims(x2, 3), self.weight_v, strides=[1, 1, 1, 1], padding='VALID')
        x2_h = tf.nn.conv2d(tf.expand_dims(x2, 3), self.weight_h, strides=[1, 1, 1, 1], padding='VALID')

        x0 = tf.sqrt(tf.square(x0_v) + tf.square(x0_h) + 1e-6)
        x1 = tf.sqrt(tf.square(x1_v) + tf.square(x1_h) + 1e-6)
        x2 = tf.sqrt(tf.square(x2_v) + tf.square(x2_h) + 1e-6)

        x = tf.concat([x0, x1, x2], 3)
        return x


class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        # self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()
        # self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad = False).cuda()
        # self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad = False).cuda()
        # self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        print(x.shape)
        print(x0.shape)
        print(x1.shape)
        print(x2.shape)
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        # n_img = len(tensor)
        # img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        # img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        pass
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)


def generate_gradient_map_pytorch(LR_path):
    img_LR = cv2.imread(LR_path, cv2.IMREAD_UNCHANGED)
    img_LR = img_LR.astype(np.float32) / 255.

    # BGR to RGB, HWC to CHW, numpy to tensor
    # if img_LR.shape[2] == 3:
    #     img_LR = img_LR[:, :, [2, 1, 0]]

    img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()
    img_LR = torch.unsqueeze(img_LR, 0)
    print(img_LR.shape)

    get_grad = Get_gradient()
    gradient_img = get_grad(img_LR.cuda())

    print(gradient_img.shape)

    out = tensor2img(gradient_img)
    save_img(out, 'gradient_img.png')


def generate_gradient_map_tensorflow(LR_path):
    img_LR = cv2.imread(LR_path, cv2.IMREAD_UNCHANGED)
    img_LR = img_LR.astype(np.float32) / 255.

    # BGR to RGB, HWC to CHW, numpy to tensor
    # if img_LR.shape[2] == 3:
    #     img_LR = img_LR[:, :, [2, 1, 0]]

    print(img_LR.shape)
    img_LR = tf.convert_to_tensor(np.ascontiguousarray(img_LR), tf.float32)
    img_LR = tf.expand_dims(img_LR, 0)

    print(img_LR.shape)
    get_grad = Get_gradient_tf()
    gradient_img = get_grad.get_gradient_tf(img_LR)

    print(gradient_img.shape)
    sess = tf.InteractiveSession()
    gradient_img = tf.squeeze(gradient_img, 0).eval()

    min_max = (0, 1)
    gradient_img = np.clip(gradient_img, 0, 1)  # clamp

    gradient_img = (gradient_img - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    gradient_img = (gradient_img * 255.0).round().astype(np.uint8)
    gradient_img = np.mean(gradient_img, 2)
    print(gradient_img.shape)
    save_img(gradient_img, 'gradient_img.png')


if __name__ == '__main__':
    LR_path = '/home/linx/桌面/2020-12-01 19-19-37屏幕截图.png'
    generate_gradient_map_tensorflow(LR_path)



