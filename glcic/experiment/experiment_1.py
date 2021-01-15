# -*- coding: utf-8 -*-
# @Time    : 2021/1/4 下午11:03
# @Author  : LinX
# @File    : experiment_1.py

import os
import argparse
import torch
import json
import torchvision.transforms as transforms
from torchvision.utils import save_image
from glcic.models import CompletionNetwork,CompletionNetwork_SE
from PIL import Image
from glcic.utils import poisson_blend, gen_input_mask
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('config')
parser.add_argument('input_img')
parser.add_argument('output_img')
parser.add_argument('--step', choices=['snap1', 'snap2'], default='snap1')
parser.add_argument('--max_holes', type=int, default=5)
parser.add_argument('--img_size', type=int, default=160)
parser.add_argument('--hole_min_w', type=int, default=24)
parser.add_argument('--hole_max_w', type=int, default=48)
parser.add_argument('--hole_min_h', type=int, default=24)
parser.add_argument('--hole_max_h', type=int, default=48)


def main(args):

    args.model = os.path.expanduser(args.model)
    args.config = os.path.expanduser(args.config)
    args.input_img = os.path.expanduser(args.input_img)
    args.output_img = os.path.expanduser(args.output_img)


    # =============================================
    # Load model
    # =============================================
    with open(args.config, 'r') as f:
        config = json.load(f)
    mpv = torch.tensor(config['mpv']).view(1, 3, 1, 1)
    # model = CompletionNetwork()
    # model.load_state_dict(torch.load(args.model, map_location='cpu'))
    if args.step == 'snap1':
        model_list = os.listdir(args.model)[:8]
        model_list.sort()
        model_list = [model_list[0], model_list[2], model_list[4], model_list[-1]]
        model_list = [os.path.join(args.model, name) for name in model_list]
    elif args.step == 'snap2':
        model_list = ['/home/linx/new_disk/checkpoints/SE+Contextual/senet/phase_3/model_cn_step100000',
                      '/home/linx/new_disk/checkpoints/SE+Contextual/senet/phase_3/model_cn_step200000',
                      '/home/linx/new_disk/checkpoints/SE+Contextual/senet/results/phase_3/model_cn_step300000',
                      '/home/linx/new_disk/checkpoints/SE+Contextual/senet/results/phase_3/model_cn_step400000']

        # model_list = ['/home/linx/new_disk/checkpoints/SE+Contextual/ori/phase_3/model_cn_step100000',
        #               '/home/linx/new_disk/checkpoints/SE+Contextual/ori/phase_3/model_cn_step200000',
        #               '/home/linx/new_disk/checkpoints/SE+Contextual/ori/phase_3/model_cn_step300000',
        #               '/home/linx/new_disk/checkpoints/SE+Contextual/ori/phase_3/model_cn_step400000']

    print(model_list)
    # =============================================
    # Predict
    # =============================================
    # convert img to tensor
    for path in os.listdir(args.input_img):
        new_path = path
        path = os.path.join(args.input_img, path)
        img = Image.open(path)
        img = transforms.Resize(args.img_size)(img)
        img = transforms.RandomCrop((args.img_size, args.img_size))(img)
        x = transforms.ToTensor()(img)
        x = torch.unsqueeze(x, dim=0)

        # create mask
        # mask = gen_input_mask(
        #     shape=(1, 1, x.shape[2], x.shape[3]),
        #     hole_size=(
        #         (args.hole_min_w, args.hole_max_w),
        #         (args.hole_min_h, args.hole_max_h),
        #     ),
        #     max_holes=args.max_holes,
        # )

        mask = gen_input_mask(
            shape=(1, 1, x.shape[2], x.shape[3]),
            hole_size=(80, 80),
            max_holes=args.max_holes,
            hole_type='center'
        )

        # inpaint
        with torch.no_grad():
            x_mask = x - x * mask + mpv * mask
            input = torch.cat((x_mask, mask), dim=1)

            imgs = torch.cat((x, x_mask), dim=0)

            for model_path in model_list:
                print(model_path)
                if args.step == 'snap1':
                    model = CompletionNetwork()
                elif args.step == 'snap2':
                    model = CompletionNetwork_SE()
                    # model = CompletionNetwork()
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                output = model(input)
                inpainted = poisson_blend(x, output, mask)
                imgs = torch.cat((imgs, inpainted), dim=0)
            save_image(imgs, os.path.join(args.output_img, new_path), nrow=6)

        # imgs = cv2.imread(args.output_img)
        # cv2.imshow('res', imgs)
        # cv2.waitKey(0)

        print('output img was saved as %s.' % os.path.join(args.output_img, new_path))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
