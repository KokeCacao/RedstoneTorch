import numpy as np
import torch.nn.functional as F
from torch import nn


class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, input, target):
        m = target.shape[0]
        p = F.softmax(input)
        log_likely = -np.log(p[range(m), target])
        loss = np.sum(log_likely) / m
        return loss

# pytorch, binary, differentiable, soft, logit, loss, bounded between 0 and inf
# https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/97456
def segmentation_weighted_binary_cross_entropy(input, target, pos_prob=0.75, neg_prob=0.25, smooth=1e-12, sum=False):
    input = input.view(input.shape[0], -1)
    target = target.view(target.shape[0], -1)
    assert(input.shape == target.shape)

    loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')

    pos = (target > 0.5).float()
    neg = (target < 0.5).float()
    pos_weight = pos.sum().item() + smooth
    neg_weight = neg.sum().item() + smooth

    # allow 75% background gradient and 25% layer gradient
    loss = (pos_prob*pos*loss/pos_weight + neg_prob*neg*loss/neg_weight)

    if sum: return loss.sum()
    return loss

def nonempty_segmentation_weighted_binary_cross_entropy(input, target, empty, pos_prob=0.75, neg_prob=0.25, smooth=1e-12, sum=False):
    input = input.view(input.shape[0], -1)
    target = target.view(target.shape[0], -1)

    empty = np.argwhere((empty.cpu().numpy().squeeze())==0).squeeze()
    print("-------", empty.shape, input.shape, target.shape)
    input = input[empty]
    target = target[empty]

    assert(input.shape == target.shape)

    loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')

    pos = (target > 0.5).float()
    neg = (target < 0.5).float()
    pos_weight = pos.sum().item() + smooth
    neg_weight = neg.sum().item() + smooth

    # allow 75% background gradient and 25% layer gradient
    loss = (pos_prob*pos*loss/pos_weight + neg_prob*neg*loss/neg_weight)

    if sum: return loss.sum()
    return loss