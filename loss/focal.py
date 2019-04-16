# from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-448-public-lb
import torch
import numpy as np

from torch import nn


class focalloss_sigmoid(nn.Module):
    """
    Since a multiclass multilabel task is considered,
    there are several things about the model that should
    be pointed out. First, the SOFTMAX MUST NOT BE USED
    as an output layer because it encourages a single label
    prediction. The common output function for multilabel
    tasks is sigmoid. However, combining the sigmoid with
    the loss function (like in BCE with logits loss or in
    Focal loss used in this kernel) allows log(sigmoid)
    optimization of the numerical stability of the loss
    function. Therefore, sigmoid is also removed.

    """
    # https://xmfbit.github.io/2017/08/14/focal-loss-paper/
    def __init__(self, alpha=0.25, gamma=2, eps=1e-7):
        super(focalloss_sigmoid, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        print("This Focal Loss Sigmoid function is deprecated and will give you really bad result.")

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

        # softmax layer
        y_pred = torch.sigmoid(y_pred)

        # To avoid divided by zero
        y_pred = y_pred + self.eps

        # Cross entropy
        ce = -(y_true * y_pred.log())

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

class focalloss_sigmoid_refined(nn.Module):
    def __init__(self, alpha=None, gamma=2, eps=1e-7):
        super(focalloss_sigmoid_refined, self).__init__()
        self.gamma = gamma

    def forward(self, target, logit):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = torch.nn.LogSigmoid()(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss.mean()

class focalloss_softmax(nn.Module):
    """
    Since a multiclass multilabel task is considered,
    there are several things about the model that should
    be pointed out. First, the SOFTMAX MUST NOT BE USED
    as an output layer because it encourages a single label
    prediction. The common output function for multilabel
    tasks is sigmoid. However, combining the sigmoid with
    the loss function (like in BCE with logits loss or in
    Focal loss used in this kernel) allows log(sigmoid)
    optimization of the numerical stability of the loss
    function. Therefore, sigmoid is also removed.

    """
    # https://xmfbit.github.io/2017/08/14/focal-loss-paper/
    def __init__(self, alpha=0.25, gamma=2, eps=1e-7):
        super(focalloss_softmax, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        print("This Focal Loss Sigmoid function is deprecated and will give you really bad result.")

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

        # softmax layer
        y_pred = torch.nn.Softmax(dim=1)(y_pred) # TODO: dim really = 1?

        # To avoid divided by zero
        y_pred = y_pred + self.eps

        # Cross entropy
        ce = -(y_true * y_pred.log())

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
