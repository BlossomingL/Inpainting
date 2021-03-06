""" common model for DCGAN """
import logging

import cv2
import neuralgym as ng
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from neuralgym.models import Model
from neuralgym.ops.summary_ops import scalar_summary, images_summary
from neuralgym.ops.summary_ops import gradients_summary
from neuralgym.ops.layers import flatten, resize
from neuralgym.ops.gan_ops import gan_wgan_loss, gradients_penalty
from neuralgym.ops.gan_ops import random_interpolates

from models.inpaint_ops import gen_conv, gen_deconv, dis_conv
from models.inpaint_ops import random_bbox, bbox2mask, local_patch
from models.inpaint_ops import spatial_discounting_mask
from models.inpaint_ops import resize_mask_like, contextual_attention


logger = logging.getLogger()


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


class InpaintCAModel(Model):
    def __init__(self):
        super().__init__('InpaintCAModel')
        self.get_grad = Get_gradient_tf()

    def build_inpaint_net(self, x, mask, config=None, reuse=False,
                          training=True, padding='SAME', name='inpaint_net'):
        """Inpaint network.

        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """
        xin = x
        offset_flow = None
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        x = tf.concat([x, ones_x, ones_x*mask], axis=3)

        # two stage network
        cnum = 32
        with tf.variable_scope(name, reuse=reuse), \
                arg_scope([gen_conv, gen_deconv],
                          training=training, padding=padding):
            # stage1
            x = gen_conv(x, cnum, 5, 1, name='conv1')
            x = gen_conv(x, 2*cnum, 3, 2, name='conv2_downsample')
            x = gen_conv(x, 2*cnum, 3, 1, name='conv3')
            x = gen_conv(x, 4*cnum, 3, 2, name='conv4_downsample')
            x = gen_conv(x, 4*cnum, 3, 1, name='conv5')
            x = gen_conv(x, 4*cnum, 3, 1, name='conv6')
            mask_s = resize_mask_like(mask, x)
            x = gen_conv(x, 4*cnum, 3, rate=2, name='conv7_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=4, name='conv8_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=8, name='conv9_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=16, name='conv10_atrous')
            x = gen_conv(x, 4*cnum, 3, 1, name='conv11')
            x = gen_conv(x, 4*cnum, 3, 1, name='conv12')
            x = gen_deconv(x, 2*cnum, name='conv13_upsample')
            x = gen_conv(x, 2*cnum, 3, 1, name='conv14')
            x = gen_deconv(x, cnum, name='conv15_upsample')
            x = gen_conv(x, cnum//2, 3, 1, name='conv16')
            x = gen_conv(x, 3, 3, 1, activation=None, name='conv17')
            x = tf.clip_by_value(x, -1., 1.)
            x_stage1 = x
            # return x_stage1, None, None

            # stage2, paste result as input
            # x = tf.stop_gradient(x)
            x = x*mask + xin*(1.-mask)
            x_gb_in = x
            x.set_shape(xin.get_shape().as_list())
            # conv branch
            xnow = tf.concat([x, ones_x, ones_x*mask], axis=3)
            x = gen_conv(xnow, cnum, 5, 1, name='xconv1')
            x = gen_conv(x, cnum, 3, 2, name='xconv2_downsample')
            x = gen_conv(x, 2*cnum, 3, 1, name='xconv3')
            x_feat1 = x
            x = gen_conv(x, 2*cnum, 3, 2, name='xconv4_downsample')
            x = gen_conv(x, 4*cnum, 3, 1, name='xconv5')
            x = gen_conv(x, 4*cnum, 3, 1, name='xconv6')
            x_feat2 = x
            x = gen_conv(x, 4*cnum, 3, rate=2, name='xconv7_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=4, name='xconv8_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=8, name='xconv9_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=16, name='xconv10_atrous')
            x_hallu = x
            x_feat3 = x

            # attention branch
            x = gen_conv(xnow, cnum, 5, 1, name='pmconv1')
            x = gen_conv(x, cnum, 3, 2, name='pmconv2_downsample')
            x = gen_conv(x, 2*cnum, 3, 1, name='pmconv3')
            x = gen_conv(x, 4*cnum, 3, 2, name='pmconv4_downsample')
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv5')
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv6',
                         activation=tf.nn.relu)
            x, offset_flow = contextual_attention(x, x, mask_s, 3, 1, rate=2)
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv9')
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv10')
            pm = x

            # gradient branch

            grad_x_stage1 = self.get_grad.get_gradient_tf(x_gb_in)
            x = gen_conv(grad_x_stage1, cnum, 5, 1, name='gbconv1')
            x = gen_conv(x, 2 * cnum, 3, 2, name='gbconv2_downsample')
            x = gen_conv(x, 2 * cnum, 3, 1, name='gbconv3')
            x = tf.concat([x, x_feat1], axis=3)  # 融合主网络的修复信息
            x = gen_conv(x, 2 * cnum, 3, 1, name='gbfushion1')

            x = gen_conv(x, 4 * cnum, 3, 2, name='gbconv4_downsample')
            x = gen_conv(x, 4 * cnum, 3, 1, name='gbconv5')
            x = gen_conv(x, 4 * cnum, 3, 1, name='gbconv6')
            x = tf.concat([x, x_feat2], axis=3)
            x = gen_conv(x, 4 * cnum, 3, 1, name='gbfushiuon2')

            x = gen_conv(x, 4 * cnum, 3, rate=2, name='gbconv7_atrous')
            x = gen_conv(x, 4 * cnum, 3, rate=4, name='gbconv8_atrous')
            x = gen_conv(x, 4 * cnum, 3, rate=8, name='gbconv9_atrous')
            x = gen_conv(x, 4 * cnum, 3, rate=16, name='gbconv10_atrous')
            x = tf.concat([x, x_feat3], axis=3)
            x = gen_conv(x, 4 * cnum, 3, 1, name='gbfushiuon3')
            gb = x

            x = gen_conv(x, 4 * cnum, 3, 1, name='gbconv11')
            x = gen_conv(x, 4 * cnum, 3, 1, name='gbconv12')
            x = gen_deconv(x, 2 * cnum, name='gbconv13_upsample')
            x = gen_conv(x, 2 * cnum, 3, 1, name='gbconv14')
            x = gen_deconv(x, cnum, name='gbconv15_upsample')
            x = gen_conv(x, cnum // 2, 3, 1, name='gbconv16')
            x = gen_conv(x, 3, 3, 1, activation=None, name='gbconv17')
            x_gb = tf.clip_by_value(x, -1., 1.)

            x = tf.concat([x_hallu, pm, gb], axis=3)
            x = gen_conv(x, 4*cnum, 3, 1, name='allconv11')
            x = gen_conv(x, 4*cnum, 3, 1, name='allconv12')
            x = gen_deconv(x, 2*cnum, name='allconv13_upsample')
            x = gen_conv(x, 2*cnum, 3, 1, name='allconv14')
            x = gen_deconv(x, cnum, name='allconv15_upsample')
            x = gen_conv(x, cnum//2, 3, 1, name='allconv16')
            x = gen_conv(x, 3, 3, 1, activation=None, name='allconv17')
            x_stage2 = tf.clip_by_value(x, -1., 1.)

            return x_stage1, x_stage2, x_gb, grad_x_stage1, offset_flow

    def build_wgan_local_discriminator(self, x, reuse=False, training=True, scope_name='discriminator_local'):
        with tf.variable_scope(scope_name, reuse=reuse):
            cnum = 64
            x = dis_conv(x, cnum, name='conv1', training=training)
            x = dis_conv(x, cnum*2, name='conv2', training=training)
            x = dis_conv(x, cnum*4, name='conv3', training=training)
            x = dis_conv(x, cnum*8, name='conv4', training=training)
            x = flatten(x, name='flatten')
            return x

    def build_wgan_global_discriminator(self, x, reuse=False, training=True, scope_name='discriminator_global'):
        with tf.variable_scope(scope_name, reuse=reuse):
            cnum = 64
            x = dis_conv(x, cnum, name='conv1', training=training)
            x = dis_conv(x, cnum*2, name='conv2', training=training)
            x = dis_conv(x, cnum*4, name='conv3', training=training)
            x = dis_conv(x, cnum*4, name='conv4', training=training)
            x = flatten(x, name='flatten')
            return x

    def build_wgan_discriminator(self, batch_local, batch_global,
                                 reuse=False, training=True, scope_name='discriminator'):
        with tf.variable_scope(scope_name, reuse=reuse):
            dlocal = self.build_wgan_local_discriminator(
                batch_local, reuse=reuse, training=training)
            dglobal = self.build_wgan_global_discriminator(
                batch_global, reuse=reuse, training=training)
            dout_local = tf.layers.dense(dlocal, 1, name='dout_local_fc')
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            return dout_local, dout_global

    def build_graph_with_losses(self, batch_data, config, training=True,
                                summary=False, reuse=False):
        batch_pos = batch_data / 127.5 - 1.
        # generate mask, 1 represents masked point
        bbox = random_bbox(config)
        mask = bbox2mask(bbox, config, name='mask_c')
        batch_incomplete = batch_pos*(1.-mask)
        if config.ADD_GRADIENT_BRANCH:
            x1, x2, fake_gb, gb, offset_flow = self.build_inpaint_net(
                batch_incomplete, mask, config, reuse=reuse, training=training,
                padding=config.PADDING)

        else:
            x1, x2, offset_flow = self.build_inpaint_net(
                batch_incomplete, mask, config, reuse=reuse, training=training,
                padding=config.PADDING)

        if config.PRETRAIN_COARSE_NETWORK:
            batch_predicted = x1
            logger.info('Set batch_predicted to x1.')
        else:
            batch_predicted = x2
            logger.info('Set batch_predicted to x2.')
        losses = {}
        # apply mask and complete image
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)  # 网络修复图
        # local patches
        local_patch_batch_pos = local_patch(batch_pos, bbox)  # gt局部图
        local_patch_batch_predicted = local_patch(batch_predicted, bbox)
        local_patch_x1 = local_patch(x1, bbox)  # 粗修复局部图
        local_patch_x2 = local_patch(x2, bbox)  # 细修复局部图
        local_patch_batch_complete = local_patch(batch_complete, bbox)  # 送入局部判别器局部图
        local_patch_mask = local_patch(mask, bbox)
        l1_alpha = config.COARSE_L1_ALPHA
        losses['l1_loss'] = l1_alpha * tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_x1) * spatial_discounting_mask(config))

        if not config.PRETRAIN_COARSE_NETWORK:
            losses['l1_loss'] += tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_x2)*spatial_discounting_mask(config))

        # =============================================================================================================
        if config.ADD_GRADIENT_BRANCH:
            gb_gt = self.get_grad.get_gradient_tf(batch_pos)
            batch_complete_gb = fake_gb * mask + gb_gt * (1. - mask)
            local_patch_batch_gb = local_patch(gb_gt, bbox)  # 梯度图gt局部图
            local_patch_fake_gb = local_patch(fake_gb, bbox)  # 生成的梯度图局部图

            x2_grad = self.get_grad.get_gradient_tf(x2)
            local_patch_x2_grad = local_patch(x2_grad, bbox)

            losses['l1_loss'] += tf.reduce_mean(
                tf.abs(local_patch_batch_gb - local_patch_fake_gb) * spatial_discounting_mask(config))
            losses['l1_loss'] += tf.reduce_mean(
                tf.abs(local_patch_batch_gb - local_patch_x2_grad) * spatial_discounting_mask(config))
        # =============================================================================================================

        losses['ae_loss'] = l1_alpha * tf.reduce_mean(tf.abs(batch_pos - x1) * (1.-mask))
        if not config.PRETRAIN_COARSE_NETWORK:
            losses['ae_loss'] += tf.reduce_mean(tf.abs(batch_pos - x2) * (1.-mask))
        losses['ae_loss'] /= tf.reduce_mean(1.-mask)
        if summary:
            scalar_summary('losses/l1_loss', losses['l1_loss'])
            scalar_summary('losses/ae_loss', losses['ae_loss'])
            viz_img = [batch_pos, batch_incomplete, batch_complete]
            if config.ADD_GRADIENT_BRANCH:
                viz_img.extend([gb_gt, batch_complete_gb])

            if offset_flow is not None:
                viz_img.append(
                    resize(offset_flow, scale=4,
                           func=tf.image.resize_nearest_neighbor))

            images_summary(
                tf.concat(viz_img, axis=2),
                'raw_incomplete_predicted_complete', config.VIZ_MAX_OUT)

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)
        # local deterministic patch
        local_patch_batch_pos_neg = tf.concat([local_patch_batch_pos, local_patch_batch_complete], 0)
        if config.GAN_WITH_MASK:
            batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(mask, [config.BATCH_SIZE*2, 1, 1, 1])], axis=3)
        # wgan with gradient penalty
        if config.GAN == 'wgan_gp':
            # seperate gan
            pos_neg_local, pos_neg_global = self.build_wgan_discriminator(local_patch_batch_pos_neg, batch_pos_neg, training=training, reuse=reuse)
            pos_local, neg_local = tf.split(pos_neg_local, 2)
            pos_global, neg_global = tf.split(pos_neg_global, 2)
            # wgan loss
            g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local, name='gan/local_gan')
            g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global, name='gan/global_gan')
            losses['g_loss'] = config.GLOBAL_WGAN_LOSS_ALPHA * g_loss_global + g_loss_local
            losses['d_loss'] = d_loss_global + d_loss_local

            # =============================================================================================================
            if config.ADD_GRADIENT_BRANCH:

                gb_batch_pos_neg = tf.concat([gb_gt, batch_complete_gb], axis=0)
                # local deterministic patch
                gb_local_patch_batch_pos_neg = tf.concat([local_patch_batch_gb, local_patch_fake_gb], 0)
                gb_pos_neg_local, gb_pos_neg_global = self.build_wgan_discriminator(gb_local_patch_batch_pos_neg, gb_batch_pos_neg,
                                                                              training=training, reuse=reuse, scope_name='gb_discriminator')
                gb_pos_local, gb_neg_local = tf.split(gb_pos_neg_local, 2)
                gb_pos_global, gb_neg_global = tf.split(gb_pos_neg_global, 2)
                # wgan loss
                gb_g_loss_local, gb_d_loss_local = gan_wgan_loss(gb_pos_local, gb_neg_local, name='gan/gb_local_gan')
                gb_g_loss_global, gb_d_loss_global = gan_wgan_loss(gb_pos_global, gb_neg_global, name='gan/gb_global_gan')
                losses['g_loss'] += (config.GLOBAL_WGAN_LOSS_ALPHA * gb_g_loss_global + gb_g_loss_local)
                losses['d_loss'] += (gb_d_loss_global + gb_d_loss_local)
            # =============================================================================================================

            # gp
            interpolates_local = random_interpolates(local_patch_batch_pos, local_patch_batch_complete)
            interpolates_global = random_interpolates(batch_pos, batch_complete)
            dout_local, dout_global = self.build_wgan_discriminator(
                interpolates_local, interpolates_global, reuse=True)
            # apply penalty
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask=local_patch_mask)
            penalty_global = gradients_penalty(interpolates_global, dout_global, mask=mask)
            losses['gp_loss'] = config.WGAN_GP_LAMBDA * (penalty_local + penalty_global)
            losses['d_loss'] = losses['d_loss'] + losses['gp_loss']
            if summary and not config.PRETRAIN_COARSE_NETWORK:
                gradients_summary(g_loss_local, batch_predicted, name='g_loss_local')
                gradients_summary(g_loss_global, batch_predicted, name='g_loss_global')
                scalar_summary('convergence/d_loss', losses['d_loss'])
                scalar_summary('convergence/local_d_loss', d_loss_local)
                scalar_summary('convergence/global_d_loss', d_loss_global)
                scalar_summary('convergence/gb_d_loss_local', gb_d_loss_local)
                scalar_summary('convergence/gb_d_loss_global', gb_d_loss_global)
                scalar_summary('gan_wgan_loss/gp_loss', losses['gp_loss'])
                scalar_summary('gan_wgan_loss/gp_penalty_local', penalty_local)
                scalar_summary('gan_wgan_loss/gp_penalty_global', penalty_global)

        if summary and not config.PRETRAIN_COARSE_NETWORK:
            # summary the magnitude of gradients from different losses w.r.t. predicted image
            gradients_summary(losses['g_loss'], batch_predicted, name='g_loss')
            gradients_summary(losses['g_loss'], x1, name='g_loss_to_x1')
            gradients_summary(losses['g_loss'], x2, name='g_loss_to_x2')
            gradients_summary(losses['l1_loss'], x1, name='l1_loss_to_x1')
            gradients_summary(losses['l1_loss'], x2, name='l1_loss_to_x2')
            gradients_summary(losses['ae_loss'], x1, name='ae_loss_to_x1')
            gradients_summary(losses['ae_loss'], x2, name='ae_loss_to_x2')
        if config.PRETRAIN_COARSE_NETWORK:
            losses['g_loss'] = 0
        else:
            losses['g_loss'] = config.GAN_LOSS_ALPHA * losses['g_loss']
        losses['g_loss'] += config.L1_LOSS_ALPHA * losses['l1_loss']
        logger.info('Set L1_LOSS_ALPHA to %f' % config.L1_LOSS_ALPHA)
        logger.info('Set GAN_LOSS_ALPHA to %f' % config.GAN_LOSS_ALPHA)
        if config.AE_LOSS:
            losses['g_loss'] += config.AE_LOSS_ALPHA * losses['ae_loss']
            logger.info('Set AE_LOSS_ALPHA to %f' % config.AE_LOSS_ALPHA)
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

    def build_infer_graph(self, batch_data, config, bbox=None, name='val'):
        """
        """
        config.MAX_DELTA_HEIGHT = 0
        config.MAX_DELTA_WIDTH = 0
        if bbox is None:
            bbox = random_bbox(config)
        mask = bbox2mask(bbox, config, name=name+'mask_c')
        batch_pos = batch_data / 127.5 - 1.

        edges = None
        batch_incomplete = batch_pos*(1.-mask)
        # inpaint
        if not config.ADD_GRADIENT_BRANCH:
            x1, x2, offset_flow = self.build_inpaint_net(
                batch_incomplete, mask, config, reuse=True,
                training=False, padding=config.PADDING)
        else:
            x1, x2, fake_gb, gb, offset_flow = self.build_inpaint_net(
                batch_incomplete, mask, config, reuse=True, training=False,
                padding=config.PADDING)
        if config.PRETRAIN_COARSE_NETWORK:
            batch_predicted = x1
            logger.info('Set batch_predicted to x1.')
        else:
            batch_predicted = x2
            logger.info('Set batch_predicted to x2.')
        # apply mask and reconstruct
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
        # global image visualization
        viz_img = [batch_pos, batch_incomplete, batch_complete]

        if config.ADD_GRADIENT_BRANCH:
            gb_gt = self.get_grad.get_gradient_tf(batch_pos)
            batch_complete_gb = fake_gb * mask + gb_gt * (1. - mask)
            viz_img.extend([gb_gt, batch_complete_gb])

        if offset_flow is not None:
            viz_img.append(
                resize(offset_flow, scale=4,
                       func=tf.image.resize_nearest_neighbor))
        images_summary(
            tf.concat(viz_img, axis=2),
            name+'_raw_incomplete_complete', config.VIZ_MAX_OUT)
        return batch_complete

    def build_static_infer_graph(self, batch_data, config, name):
        """
        """
        # generate mask, 1 represents masked point
        bbox = (tf.constant(config.HEIGHT//2), tf.constant(config.WIDTH//2),
                tf.constant(config.HEIGHT), tf.constant(config.WIDTH))
        return self.build_infer_graph(batch_data, config, bbox, name)

    def build_server_graph(self, batch_data, reuse=False, is_training=False):
        """
        """
        # generate mask, 1 represents masked point
        batch_raw, masks_raw = tf.split(batch_data, 2, axis=2)
        masks = tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)

        batch_pos = batch_raw / 127.5 - 1.
        gb_gt = self.get_grad.get_gradient_tf(batch_pos)

        batch_incomplete = batch_pos * (1. - masks)
        # inpaint
        x1, x2, x_gb, grad_x_stage1, flow = self.build_inpaint_net(
            batch_incomplete, masks, reuse=reuse, training=is_training,
            config=None)
        batch_complete_gb = x_gb * masks + gb_gt * (1. - masks)

        batch_predict = x2
        # apply mask and reconstruct
        batch_complete = batch_predict * masks + batch_incomplete*(1 - masks)

        return batch_complete, gb_gt, batch_complete_gb
