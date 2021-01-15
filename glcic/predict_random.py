import numpy as np
import cv2
import torch
import sys
sys.path.append('..')
from .models import CompletionNetwork
from PIL import Image
import os
import torchvision.transforms as transforms
import argparse
import json
from .utils import poisson_blend, gen_input_mask
from torchvision.utils import save_image
from torch.nn import DataParallel
from collections import OrderedDict

drawing = False  # true if mouse is pressed
ix, iy = -1, -1
color = (255, 255, 255)
size = 5

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('config')
parser.add_argument('input_img')
parser.add_argument('output_img')
parser.add_argument('--max_holes', type=int, default=5)
parser.add_argument('--img_size', type=int, default=160)
parser.add_argument('--hole_min_w', type=int, default=24)
parser.add_argument('--hole_max_w', type=int, default=48)
parser.add_argument('--hole_min_h', type=int, default=24)
parser.add_argument('--hole_max_h', type=int, default=48)
parser.add_argument('--test_num', type=int, default=1)


def erase_img(img, args):
    print(img.shape)
    # mouse callback function

    def erase_rect(event, x, y, flags, param):
        global ix, iy, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            if drawing == True:
                # cv2.circle(img,(x,y),10,(255,255,255),-1)
                cv2.rectangle(img, (x - size, y - size), (x + size, y + size), color, -1)
                cv2.rectangle(mask, (x - size, y - size), (x + size, y + size), color, -1)

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                # cv2.circle(img,(x,y),10,(255,255,255),-1)
                cv2.rectangle(img, (x - size, y - size), (x + size, y + size), color, -1)
                cv2.rectangle(mask, (x - size, y - size), (x + size, y + size), color, -1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            # cv2.circle(img,(x,y),10,(255,255,255),-1)
            cv2.rectangle(img, (x - size, y - size), (x + size, y + size), color, -1)
            cv2.rectangle(mask, (x - size, y - size), (x + size, y + size), color, -1)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', erase_rect)
    cv2.namedWindow('mask')
    cv2.setMouseCallback('mask', erase_rect)
    mask = np.zeros([img.shape[0], img.shape[1], 1])

    while (1):
        img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('image', img_show)
        k = cv2.waitKey(1) & 0xFF
        if k == 13:
            break

    mask = mask / 255.0

    # print(test_mask)
    # print(test_mask.shape)
    # print(test_mask)
    # fill mask region to 1
    # test_img = (test_img * (1 - test_mask)) + test_mask
    # t = np.tile(test_mask,[BATCH_SIZE, 1, 1, 1])
    # print(t.shape)

    cv2.destroyAllWindows()
    # return np.tile(test_img[np.newaxis, ...], [BATCH_SIZE, 1, 1, 1]), np.tile(test_mask[np.newaxis, :, :, np.newaxis],
    #                                                                                [BATCH_SIZE, 1, 1, 1])
    return mask[np.newaxis, :, :, :]


def test(args):
    # saver

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
    model = CompletionNetwork()
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model = model.cuda()

    # =============================================
    # Predict
    # =============================================
    # convert img to tensor
    img = Image.open(args.input_img)
    img = transforms.Resize(args.img_size)(img)
    # img = transforms.RandomCrop((args.img_size, args.img_size))(img)
    x = transforms.ToTensor()(img)
    x = torch.unsqueeze(x, dim=0)

    # img_input = cv2.imread(args.input_img)
    # img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    mask = erase_img(np.asarray(img), args)
    print(mask.shape)
    # mask = 1 - mask
    mask = np.transpose(mask, [0, 3, 1, 2])
    mask = torch.from_numpy(mask).type(torch.float32)

    # print(mask)
    # inpaint
    with torch.no_grad():
        x_mask = x - x * mask + mpv * mask
        input = torch.cat((x_mask, mask), dim=1)
        input = input.cuda()
        output = model(input)

        inpainted = poisson_blend(x, output, mask)
        imgs = torch.cat((x, x_mask, inpainted), dim=0)
        save_image(imgs, args.output_img, nrow=3)
    print('output img was saved as %s.' % args.output_img)


def main():
    args = parser.parse_args()
    test(args)


if __name__ == '__main__':
    main()