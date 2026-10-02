import argparse
import torch
import numpy as np
import operator
from functools import reduce

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def get_pos_lst(size_lst, length):
    pos_lst = []
    for size in size_lst:
        nx, ny = size
        pos_x = torch.linspace(0, length[0], nx).float().cuda().unsqueeze(-1)
        pos_y = torch.linspace(0, length[1], ny).float().cuda().unsqueeze(-1)
        pos_lst.append([pos_x, pos_y])
    return pos_lst


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.shape[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class MaskedLpLoss:
    """Relative Lp error over valid spatial cells only."""

    def __init__(self, p=2, reduction=True, size_average=True, eps=1e-8):
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.eps = eps

    def __call__(self, prediction, target, valid_mask):
        if prediction.shape != target.shape or valid_mask.shape != target.shape[:-1]:
            raise ValueError("expected prediction/target [B, H, W, C] and mask [B, H, W]")
        has_ocean = valid_mask.reshape(valid_mask.shape[0], -1).any(dim=1)

        valid = valid_mask.bool().unsqueeze(-1)
        difference = torch.where(valid, prediction - target, 0).reshape(prediction.shape[0], -1)
        reference = torch.where(valid, target, 0).reshape(target.shape[0], -1)
        error = torch.linalg.vector_norm(difference, ord=self.p, dim=1)
        scale = torch.linalg.vector_norm(reference, ord=self.p, dim=1).clamp_min(self.eps)
        losses = error / scale
        if not self.reduction:
            return losses
        # All-land samples contribute zero gradient without diluting the mean.
        return losses.sum() / has_ocean.sum().clamp_min(1) if self.size_average else losses.sum()


def normalize_uv_input(x, mean, std):
    """Normalize U/V at valid ocean pixels while leaving the mask at 0/1."""
    if x.shape[-1] != 3 or mean.numel() != 2 or std.numel() != 2:
        raise ValueError("expected input [..., 3] and two-channel U/V statistics")
    velocity = torch.where(x[..., 2:3].bool(),
                           (x[..., :2] - mean) / std,
                           torch.zeros_like(x[..., :2]))
    return torch.cat((velocity, x[..., 2:3]), dim=-1)
    
    
    
# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c
