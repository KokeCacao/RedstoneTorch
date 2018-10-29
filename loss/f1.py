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
    predicted = (predicted > threshold).astype(np.float32)

    #f1 per feature
    groundPositives = np.sum(label, axis=0) + epsilon
    correctPositives = np.sum(label * predicted, axis=0)
    predictedPositives = np.sum(predicted, axis=0) + epsilon

    precision = correctPositives / predictedPositives
    recall = correctPositives / groundPositives

    m = (2 * precision * recall) / (precision + recall + epsilon)

    return m.mean()