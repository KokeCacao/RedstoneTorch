# --------------------------- DiceLoss ---------------------------
import torch
from torch import Tensor
from torch.autograd import Function

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + 1e-10

        t = (2 * self.inter.float() + 1e-10) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union + self.inter) \
                         / self.union * self.union
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

# adapted from https://www.kaggle.com/iafoss/hypercolumns-pneumothorax-fastai-0-818-lb
class denoised_siim_dice(torch.nn.Module):
    def __init__(self, threshold, iou = False, eps = 1e-8, denoised = False):
        super(denoised_siim_dice).__init__()
        self.threshold = threshold
        self.iou = iou
        self.eps = eps
        self.denoised = denoised

    def forward(self, input, targs):
        n = targs.shape[0]

        # eliminate all predictions with a few (noise_th) pixesls
        # area < 0.6103516% of the image can be seen as noise
        noise_th = 100.0 * (n / 128.0) ** 2  # threshold for the number of predicted pixels

        input = torch.softmax(input, dim=1)[:,1,...].view(n,-1)
        input = (input > self.threshold).long()
        if self.denoised: input[input.sum(-1) < noise_th,...] = 0.0
        #input = input.argmax(dim=1).view(n,-1)
        targs = targs.view(n,-1)
        intersect = (input * targs).sum(-1).float()
        union = (input+targs).sum(-1).float()
        if not self.iou: return ((2.0*intersect + self.eps) / (union+self.eps)).mean()
        else: return ((intersect + self.eps) / (union - intersect + self.eps)).mean()

def siim_dice_overall(preds, targs):
    n = preds.shape[0]
    preds = preds.view(n, -1)
    targs = targs.view(n, -1)
    intersect = (preds * targs).sum(-1).float()
    union = (preds+targs).sum(-1).float()
    u0 = union==0
    intersect[u0] = 1
    union[u0] = 2
    return (2. * intersect / union)