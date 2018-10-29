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
    predicted = (predicted > threshold).float()

    #f1 per feature
    groundPositives = np.sum(label, axis=0) + epsilon
    correctPositives = np.sum(label * predicted, axis=0)
    predictedPositives = np.sum(predicted, axis=0) + epsilon

    precision = correctPositives / predictedPositives
    recall = correctPositives / groundPositives

    m = (2 * precision * recall) / (precision + recall + epsilon)

    return m.mean()


def f1_micro(y_true, y_preds, thresh=0.5, eps=1e-20):
    preds_bin = y_preds > thresh  # binary representation from probabilities (not relevant)
    truepos = preds_bin * y_true

    p = truepos.sum() / (preds_bin.sum() + eps)  # take sums and calculate precision on scalars
    r = truepos.sum() / (y_true.sum() + eps)  # take sums and calculate recall on scalars

    f1 = 2 * p * r / (p + r + eps)  # we calculate f1 on scalars
    return f1


def f1_macro(y_true, y_preds, thresh=0.5, eps=1e-20):
    preds_bin = y_preds > thresh  # binary representation from probabilities (not relevant)
    truepos = preds_bin * y_true

    p = truepos.sum(axis=0) / (preds_bin.sum(axis=0) + eps)  # sum along axis=0 (classes)
    # and calculate precision array
    r = truepos.sum(axis=0) / (y_true.sum(axis=0) + eps)  # sum along axis=0 (classes)
    #  and calculate recall array

    f1 = 2 * p * r / (p + r + eps)  # we calculate f1 on arrays