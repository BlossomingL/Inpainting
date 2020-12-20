# -*- coding: utf-8 -*-
# @Time    : 2020/12/19 上午6:50
# @Author  : LinX
# @File    : test.py
import cv2
import numpy as np
import tensorflow as tf
from neuralgym.ops.image_ops import np_random_crop
import os


def test(args, image, mask, output_path):

    h, w, _ = mask.shape

    if args.rand_crop:
        image, random_h, random_w = np_random_crop(
            image, (h, w),
            None, None, align=False)  # use last rand

    grid = 8
    image = image[:h // grid * grid, :w // grid * grid, :]
    mask = mask[:h // grid * grid, :w // grid * grid, :]
    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    print('Shape of image: {}'.format(image.shape))

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    tf.reset_default_graph()

    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)

        if args.test_deepFillV1:
            res = image
            import models.CA as ca
            model = ca.InpaintCAModel()
            output, incomplete = model.build_server_graph(input_image)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)

            incomplete = (incomplete + 1.) * 127.5
            incomplete = tf.reverse(incomplete, [-1])
            incomplete = tf.saturate_cast(incomplete, tf.uint8)

            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.contrib.framework.load_variable(args.checkpoint_dir_CA, from_name)
                assign_ops.append(tf.assign(var, var_value))
            sess.run(assign_ops)
            print('CA Model loaded.')
            result = sess.run(output)
            incomplete = sess.run(incomplete)
            print(res.shape, incomplete.shape, result.shape)
            res = np.concatenate((res[0], incomplete[0][:, :, ::-1], result[0][:, :, ::-1]), axis=1)
            cv2.imwrite(output_path, res)

        if args.test_deepFillV1_gradient_branch:
            import models.CA_GB as ca_gb
            model = ca_gb.InpaintCAModel()
            output, gb_gt, batch_complete_gb = model.build_server_graph(input_image)  # 修复后图片

            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)

            batch_complete_gb = (batch_complete_gb + 1.) * 127.5
            batch_complete_gb = tf.reverse(batch_complete_gb, [-1])
            batch_complete_gb = tf.saturate_cast(batch_complete_gb, tf.uint8)

            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.contrib.framework.load_variable(args.checkpoint_dir_GB, from_name)
                assign_ops.append(tf.assign(var, var_value))
            sess.run(assign_ops)
            print('CA_GB Model loaded.')
            result = sess.run(output)
            batch_complete_gb = sess.run(batch_complete_gb)
            res = np.concatenate((result[0][:, :, ::-1], batch_complete_gb[0][:, :, ::-1]), axis=1)

            cv2.imwrite(output_path, res)


def test_batch(args):
    imgs = os.listdir(args.batch_image)
    mask = cv2.imread(args.mask)

    for img_name in imgs:
        path = os.path.join(args.batch_image, img_name)
        image = cv2.imread(path)
        output_path = os.path.join(args.output_path, img_name)
        test(args, image, mask, output_path)


def test_single(args):
    pass


def concanate(args):
    imgs = os.listdir(args.file1)
    for img in imgs:
        path_head = os.path.join(args.file1, img)
        path_tail = os.path.join(args.file2, img)
        output_path = os.path.join(args.final_file, img)

        img_head = cv2.imread(path_head)
        img_tail = cv2.imread(path_tail)

        res = np.concatenate((img_head, img_tail), axis=1)
        if args.reverse:
            from copy import deepcopy
            third = res[:, 512: 768, :]
            x = deepcopy(third)
            fourth = res[:, 768: 1024, :]
            res[:, 512: 768, :] = deepcopy(fourth)
            res[:, 768: 1024, :] = x

        cv2.imwrite(output_path, res)






