# -*- coding: utf-8 -*-
# @Time    : 2020/5/15 上午9:46
# @Author  : LinX
# @File    : make_flist.py

import os

dataset_path = '/media/linx/dataset/img_align_celeba'


def mk_flist(root_path):
    train_path = os.path.join(root_path, 'train')
    test_path = os.path.join(root_path, 'test')

    train_list = os.listdir(train_path)
    test_list = os.listdir(test_path)

    with open('train.flist', 'w') as f:
        for name in train_list:
            path = train_path + '/' + name + '\n'
            f.write(path)

    with open('valid.flist', 'w') as f:
        for name in test_list:
            path = test_path + '/' + name + '\n'
            f.write(path)


if __name__ == '__main__':
    mk_flist(dataset_path)
