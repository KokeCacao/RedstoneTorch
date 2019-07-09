import numpy as np
from torch import nn
import torch.nn.functional as F


def iou_score(outputs, labels, threshold=0.5):
    outputs = outputs > threshold  # threshold

    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1).byte()  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1).byte()

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zero if both are 0

    iou = np.array((intersection + 1e-10) / (union + 1e-10))  # We smooth our devision to avoid 0/0

    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    return iou  # Or thresholded.mean() if you are interested in average across the batch

class mIoULoss(nn.Module):
    def __init__(self, mean=False, eps=1e-5):
        super(mIoULoss, self).__init__()
        self.mean = mean
        self.eps = eps

    def forward(self, target, pred):
        # pred => N x Classes x H x W
        # target => N x Classes x H x W
        n_classes = pred.shape[1]
        N = pred.size()[0]

        # Numerator Product
        inter = pred * target
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, n_classes, -1).sum(2)

        # Denominator
        union = pred + target - (pred * target)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, n_classes, -1).sum(2)
        if self.mean: return 1 - ((inter + self.eps) / (union + self.eps)).mean()
        return 1 - ((inter + self.eps) / (union + self.eps))