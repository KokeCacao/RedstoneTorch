# from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-448-public-lb
import torch
import numpy as np

from torch import nn
import torch.nn.functional as F

class Focal_Loss_from_git(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, eps=1e-7):
        super(Focal_Loss_from_git, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, y_true, y_pred):
        """
        focal loss for multi-class classification
        fl(pt) = -alpha*(1-pt)^(gamma)*log(pt)
        :param y_true: ground truth one-hot vector shape of [batch_size, nb_class]
        :param y_pred: prediction after softmax shape of [batch_size, nb_class]
        :param alpha:
        :param gamma:
        :return:
        """
        # # parameters
        # alpha = 0.25
        # gamma = 2

        # To avoid divided by zero
        y_pred = y_pred + self.eps
        print(y_pred)

        # Cross entropy
        ce = -(y_true * y_pred.log())

        print(ce)
        # Not necessary to multiply y_true(cause it will multiply with CE which has set unconcerned index to zero ),
        # but refer to the definition of p_t, we do it
        weight = ((1 - y_pred) **self.gamma) * y_true

        # Now fl has a shape of [batch_size, nb_class]
        # alpha should be a step function as paper mentioned, but it doesn't matter like reason mentioned above
        # (CE has set unconcerned index to zero)
        #
        # alpha_step = tf.where(y_true, alpha*np.ones_like(y_true), 1-alpha*np.ones_like(y_true))
        fl = ce * weight * self.alpha

        # Both reduce_sum and reduce_max are ok
        reduce_fl = fl.sum(dim=1)
        return reduce_fl

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

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
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

    def forward(self, input, target):
        # y = one_hot(target, input.size(-1))
        # logit = F.softmax(input, dim=-1)
        # logit = logit.clamp(min = self.eps, max = 1. - self.eps)

        loss = -1 * input * F.log_softmax(input, dim=-1) # cross entropy
        loss = loss * (1 - F.softmax(input, dim=-1)) ** self.gamma # focal loss
        # print(loss.shape) -> (Batch, 28)
        return loss.sum(dim=1)

def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(-x))