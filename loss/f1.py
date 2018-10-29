import numpy as np

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
        print("The ideal input of loss function is numpy array, converting it.")
        if type(predicted) is not np.ndarray: predicted = predicted.numpy()
    except Exception as e:
        print("The tensor is on gpu, trying to detach.")
        predicted = predicted.detach().cpu().numpy()
    try:
        print("The ideal input of loss function is numpy array, converting it.")
        if type(label) is not np.ndarray: label = label.numpy()
    except Exception as e:
        print("The tensor is on gpu, trying to detach.")
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
        print("The ideal input of loss function is numpy array, converting it.")
        if type(y_preds) is not np.ndarray: y_preds = y_preds.numpy()
    except Exception as e:
        print("The tensor is on gpu, trying to detach.")
        y_preds = y_preds.detach().cpu().numpy()
    try:
        print("The ideal input of loss function is numpy array, converting it.")
        if type(y_true) is not np.ndarray: y_true = y_true.numpy()
    except Exception as e:
        print("The tensor is on gpu, trying to detach.")
        y_true = y_true.detach().cpu().numpy()

    preds_bin = y_preds > thresh  # binary representation from probabilities (not relevant)
    truepos = preds_bin * y_true

    p = truepos.sum() / (preds_bin.sum() + eps)  # take sums and calculate precision on scalars
    r = truepos.sum() / (y_true.sum() + eps)  # take sums and calculate recall on scalars

    f1 = 2 * p * r / (p + r + eps)  # we calculate f1 on scalars
    return f1


def f1_macro(y_preds, y_true, thresh=0.5, eps=1e-20):
    try:
        print("The ideal input of loss function is numpy array, converting it.")
        if type(y_preds) is not np.ndarray: y_preds = y_preds.numpy()
    except Exception as e:
        print("The tensor is on gpu, trying to detach.")
        y_preds = y_preds.detach().cpu().numpy()
    try:
        print("The ideal input of loss function is numpy array, converting it.")
        if type(y_true) is not np.ndarray: y_true = y_true.numpy()
    except Exception as e:
        print("The tensor is on gpu, trying to detach.")
        y_true = y_true.detach().cpu().numpy()

    preds_bin = y_preds > thresh  # binary representation from probabilities (not relevant)
    truepos = preds_bin * y_true

    p = truepos.sum(axis=1) / (preds_bin.sum(axis=1) + eps)  # sum along axis=0 (classes)
    # and calculate precision array
    r = truepos.sum(axis=1) / (y_true.sum(axis=1) + eps)  # sum along axis=0 (classes)
    #  and calculate recall array

    f1 = 2 * p * r / (p + r + eps)  # we calculate f1 on arrays
    return f1