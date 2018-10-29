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