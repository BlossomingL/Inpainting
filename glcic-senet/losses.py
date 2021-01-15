from torch.nn.functional import mse_loss
import torch
from contextual_loss.ContextualLoss import Contextual_Loss


def completion_network_loss(input, output, mask):
    return mse_loss(output * mask, input * mask)


def contextual_loss(I, T, mask):
    layers = {
        "conv_1_1": 1.0,
        "conv_3_2": 1.0
    }
    contex_loss = Contextual_Loss(layers, max_1d_size=64).cuda()
    return contex_loss(I, T)