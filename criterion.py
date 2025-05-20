import torch
from torch import nn


class CLFcriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_logit:torch.Tensor, target:torch.Tensor):
        '''
        :param x_logit: model output tensor, shape must be same with target tensor
        :param target: target tensor (num_cls, 1) tensor
        :return: loss
        '''

        # assert x_logit.shape == target.shape, "Tensors Must have same shape"

        # x = nn.functional.softmax(x_logit, 0)
        if target.dtype != torch.long:
            target = target.long()

        loss = nn.functional.cross_entropy(x_logit, target)

        return loss