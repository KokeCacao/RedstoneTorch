# from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-448-public-lb
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, predict, target):
        if not (target.size() == predict.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), predict.size()))

        max_val = (-predict).clamp(min=0)
        loss = predict - predict * target + max_val + \
               ((-max_val).exp() + (-predict - max_val).exp()).log()

        invprobs = F.logsigmoid(-predict * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1)
