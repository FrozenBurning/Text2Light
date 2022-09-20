import torch
import torch.nn as nn
import numpy as np


def logl1(hdr, hdr_gt):
    bs, _, H, W = hdr_gt.shape
    hdr_gt = hdr_gt.permute(0, 2, 3, 1).reshape(-1, 3)
    l1 = nn.L1Loss()
    mu = 5000
    log_gt = torch.log10(1 + mu * hdr_gt)
    log_pred = torch.log10(1 + mu * hdr)
    return l1(log_gt, log_pred) / np.log10(1+mu)

def l1loss(hdr, hdr_gt, msk=None):
    bs, _, H, W = hdr_gt.shape
    hdr_gt = hdr_gt.permute(0, 2, 3, 1).reshape(-1, 3)
    l1 = nn.L1Loss()
    if msk is not None:
        msk = msk.reshape(-1, 1).detach()
        return l1(hdr_gt * msk, hdr * msk)
    else:
        return l1(hdr_gt, hdr)

def si_mse(hdr, hdr_gt):
    diff = hdr - hdr_gt
    return torch.mean(diff ** 2) - torch.mean(diff) ** 2

def get_loss(args):
    if args.loss == 'l1':
        return l1loss
    elif args.loss == 'logl1':
        return logl1
    elif args.loss =='simse':
        return si_mse
    else:
        raise NotImplementedError("Loss type [{}] is not supported.".format(args.loss))

