import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from loss.f1 import sigmoid_np


class FocalLoss0(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()


class FocalLossMultiLabel(nn.Module):
    def __init__(self, gamma, weight):
        super().__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(weight=weight, reduce=False)

    def forward(self, target, input):
        loss = self.nll(input, target)

        inv_probs = 1 - input.exp()
        focal_weights = (inv_probs * target).sum(dim=1) ** self.gamma
        loss = loss * focal_weights

        return loss.mean()


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, target, input):
        # y = one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(min = self.eps, max = 1. - self.eps)

        loss = -1 * logit * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss
        return loss.sum(dim=1)


class FocalLoss_reduced(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss_reduced, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, target, input):
        # y = one_hot(target, input.size(-1))
        # logit = F.softmax(input, dim=-1)
        # logit = logit.clamp(min = self.eps, max = 1. - self.eps)

        loss = -1 * input * F.log_softmax(input, dim=-1) # cross entropy
        loss = loss * (1 - F.softmax(input, dim=-1)) ** self.gamma # focal loss
        # print(loss.shape) -> (Batch, 28)
        return loss.sum(dim=1)


# class FocalLoss(nn.Module):
#
#     def __init__(self, gamma=0, eps=1e-7):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.eps = eps
#
#     def forward(self, input, target):
#         # y = one_hot(target, input.size(-1))
#         logit = F.softmax(input, dim=-1)
#         logit = logit.clamp(min = self.eps, max = 1. - self.eps)
#
#         loss = -1 * input * torch.log(logit) # cross entropy
#         loss = loss * (1 - logit) ** self.gamma # focal loss
#         return loss.sum(dim=1)

def F1_soft(preds,targs,th=0.5,d=50.0):
    preds = sigmoid_np(d*(preds - th))
    targs = targs.astype(np.float)
    score = 2.0*(preds*targs).sum(axis=0)/((preds+targs).sum(axis=0) + 1e-6)
    return score


def competitionMetric(predicted, label, threshold = 0.5, epsilon = 1e-8):
    """

    :param predicted: numpy array
    :param label: numpy array
    :param threshold: threshold
    :param epsilon: small number
    :return: scaler: (2 * precision * recall) / (precision + recall + epsilon)
    """
    try:
        logging.debug("The ideal input of loss function is numpy array, converting it.")
        if type(predicted) is not np.ndarray: predicted = predicted.numpy()
    except Exception as e:
        logging.debug("The tensor is on gpu, trying to detach.")
        try:
            predicted = predicted.cpu().numpy()
        except Exception as e:
            predicted = predicted.detach().cpu().numpy()
    try:
        logging.debug("The ideal input of loss function is numpy array, converting it.")
        if type(label) is not np.ndarray: label = label.numpy()
    except Exception as e:
        logging.debug("The tensor is on gpu, trying to detach.")
        try:
            label = label.cpu().numpy()
        except Exception as e:
            label = label.detach().cpu().numpy()

    predicted = (predicted > threshold).astype(np.float32)

    #f1 per feature
    groundPositives = label.sum(axis=1) + epsilon
    correctPositives = (label * predicted).sum(axis=1)
    predictedPositives = predicted.sum(axis=1) + epsilon

    precision = correctPositives / predictedPositives
    recall = correctPositives / groundPositives

    m = (2 * precision * recall) / (precision + recall + epsilon)

    return np.array(m).mean()