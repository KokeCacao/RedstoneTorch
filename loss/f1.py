import logging
import torch

import numpy as np
from torch import nn
import torch.nn.functional as F


def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(-x))

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


def f1_micro(y_preds, y_true, thresh=0.5, eps=1e-20):
    try:
        logging.debug("The ideal input of loss function is numpy array, converting it.")
        if type(y_preds) is not np.ndarray: y_preds = y_preds.numpy()
    except Exception as e:
        logging.debug("The tensor is on gpu, trying to detach.")
        try:
            y_preds = y_preds.cpu().numpy()
        except Exception as e:
            y_preds = y_preds.detach().cpu().numpy()
    try:
        logging.debug("The ideal input of loss function is numpy array, converting it.")
        if type(y_true) is not np.ndarray: y_true = y_true.numpy()
    except Exception as e:
        logging.debug("The tensor is on gpu, trying to detach.")
        try:
            y_true = y_true.cpu().numpy()
        except Exception as e:
            y_true = y_true.detach().cpu().numpy()

    preds_bin = y_preds > thresh  # binary representation from probabilities (not relevant)
    truepos = preds_bin * y_true

    p = truepos.sum() / (preds_bin.sum() + eps)  # take sums and calculate precision on scalars
    r = truepos.sum() / (y_true.sum() + eps)  # take sums and calculate recall on scalars

    f1 = 2 * p * r / (p + r + eps)  # we calculate f1 on scalars
    return f1


def f1_macro(y_preds, y_true, thresh=0.5, eps=1e-20):
    try:
        logging.debug("The ideal input of loss function is numpy array, converting it.")
        if type(y_preds) is not np.ndarray: y_preds = y_preds.numpy()
    except Exception as e:
        logging.debug("The tensor is on gpu, trying to detach.")
        try:
            y_preds = y_preds.cpu().numpy()
        except Exception as e:
            y_preds = y_preds.detach().cpu().numpy()
    try:
        logging.debug("The ideal input of loss function is numpy array, converting it.")
        if type(y_true) is not np.ndarray: y_true = y_true.numpy()
    except Exception as e:
        logging.debug("The tensor is on gpu, trying to detach.")
        try:
            y_true = y_true.detach().cpu().numpy()
        except Exception as e:
            y_true = y_true.cpu().numpy()

    preds_bin = y_preds > thresh  # binary representation from probabilities (not relevant)
    truepos = preds_bin * y_true

    p = truepos.sum(axis=1) / (preds_bin.sum(axis=1) + eps)  # sum along axis=0 (classes)
    # and calculate precision array
    r = truepos.sum(axis=1) / (y_true.sum(axis=1) + eps)  # sum along axis=0 (classes)
    #  and calculate recall array


    """Macro F1 calculates metrics for each label, 
    and find their unweighted mean. (7 missing classes in LB) / (28 total classes) = 0.25, 
    and if the organizer is interpreting 0/0 as 0, this explains 0.25 of the LB drop. 
    The other 0.1 could be that public LB has more hard examples. 
    This made me suspicious of if the public/private LB split is truly random. 
    It is possible that private dataset has more balanced classes.
    
    https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/69462
    https://storage.googleapis.com/kaggle-forum-message-attachments/inbox/637434/44b045d6c4e1ddcd8ecbb9f8350f8dd1/cv-lb.png
    """

    f1 = 2 * p * r / (p + r + eps)  # we calculate f1 on arrays
    return f1

class Differenciable_F1(nn.Module):
    def differenciable_f1_loss(self, eps=1e-6, beta=1):
        self.eps = eps
        self.beta = beta

    def forward(self, labels, logits):
        batch_size = logits.size()[0]
        p = F.sigmoid(logits)
        l = labels
        num_pos = torch.sum(p, 1) + self.eps
        num_pos_hat = torch.sum(l, 1) + self.eps
        tp = torch.sum(l * p, 1)
        precise = tp / num_pos
        recall = tp / num_pos_hat
        fs = (1 + self.beta * self.beta) * precise * recall / (self.beta * self.beta * precise + recall + self.eps)
        loss = fs.sum() / batch_size
        return (1 - loss)
