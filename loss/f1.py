import numpy as np

def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(-x))

def F1_soft(preds,targs,th=0.5,d=50.0):
    preds = sigmoid_np(d*(preds - th))
    targs = targs.astype(np.float)
    score = 2.0*(preds*targs).sum(axis=0)/((preds+targs).sum(axis=0) + 1e-6)
    return score

def competitionMetric(true,pred):
    pred = K.cast(K.greater(pred, 0.5), K.floatx())

    #f1 per feature
    groundPositives = K.sum(true, axis=0) + K.epsilon()
    correctPositives = K.sum(true * pred, axis=0)
    predictedPositives = K.sum(pred, axis=0) + K.epsilon()

    precision = correctPositives / predictedPositives
    recall = correctPositives / groundPositives

    m = (2 * precision * recall) / (precision + recall + K.epsilon())

    #macro average
    return K.mean(m)