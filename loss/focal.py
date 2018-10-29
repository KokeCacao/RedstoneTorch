# from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-448-public-lb
import torch

from torch import nn
import torch.nn.functional as F

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#
#     def forward(self, predict, target):
#         if not (target.size() == predict.size()):
#             raise ValueError("Target size ({}) must be the same as input size ({})"
#                              .format(target.size(), predict.size()))
#
#         max_val = (-predict).clamp(min=0)
#         loss = predict - predict * target + max_val + ((-max_val).exp() + (-predict - max_val).exp()).log()
#
#         invprobs = F.logsigmoid(-predict * (target * 2.0 - 1.0))
#         loss = (invprobs * self.gamma).exp() * loss
#
#         return loss.sum(dim=1)

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#
#     def forward(self, predict, target):
#         if not (target.size() == predict.size()):
#             raise ValueError("Target size ({}) must be the same as input size ({})"
#                              .format(target.size(), predict.size()))
#
#         max_val = (-predict).clamp(min=0)
#         loss = predict - predict * target + max_val + ((-max_val).exp() + (-predict - max_val).exp()).log()
#
#         invprobs = F.logsigmoid(-predict * (target * 2.0 - 1.0))
#         loss = (invprobs * self.gamma).exp() * loss
#
#         return loss.sum(dim=1)

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        # y = one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        print(logit)
        logit = logit.clamp(min = self.eps, max = 1. - self.eps)

        loss = -1 * input * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss
        return loss.sum(dim=1)

class FocalLoss_reduced(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss_reduced, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        # y = one_hot(target, input.size(-1))
        # logit = F.softmax(input, dim=-1)
        # logit = logit.clamp(min = self.eps, max = 1. - self.eps)

        loss = -1 * input * F.log_softmax(input, dim=-1) # cross entropy
        loss = loss * (1 - F.softmax(input, dim=-1)) ** self.gamma # focal loss
        # print(loss.shape) -> (Batch, 28)
        return loss.sum(dim=1)