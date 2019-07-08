# --------------------------- DiceLoss ---------------------------
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
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
# work for pytorch, soft, activated, differentiable, loss
class denoised_siim_dice(torch.nn.Module):
    def __init__(self, threshold, iou=False, eps=1e-8, denoised=False, mean=False):
        super(denoised_siim_dice, self).__init__()
        self.threshold = threshold
        self.iou = iou
        self.eps = eps
        self.denoised = denoised
        self.mean = mean

    def forward(self, label, pred):
        n = label.shape[0]

        # eliminate all predictions with a few (noise_th) pixesls
        # area < 0.6103516% of the image can be seen as noise
        noise_th = 100.0 * (n / 128.0) ** 2  # threshold for the number of predicted pixels

        """dim here should be 0 instead of 1?"""
        pred = pred.view(n, -1)
        pred = (pred > self.threshold).float()
        if self.denoised: pred[pred.sum(-1) < noise_th, ...] = 0.0
        # pred = pred.argmax(dim=1).view(n,-1)
        label = label.view(n, -1)
        intersect = (pred * label).sum(-1).float()
        union = (pred + label).sum(-1).float()
        if not self.iou:
            if self.mean: return ((2.0 * intersect + self.eps) / (union + self.eps)).mean()
            else: return ((2.0 * intersect + self.eps) / (union + self.eps))
        else:
            if self.mean: return ((intersect + self.eps) / (union - intersect + self.eps)).mean()
            else: return ((intersect + self.eps) / (union - intersect + self.eps))

# adapted from: https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
def dice_loss(true, logits, eps=1e-7):
    """Computes the Sorensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    """add eps on top"""
    dice_loss = (2. * intersection  + eps / (cardinality + eps)).mean()
    return (1 - dice_loss)

# adapted from: https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
def jaccard_loss(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(probas, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = ((intersection + eps) / (union + eps)).mean()
    return (1 - jacc_loss)


# work for pytorch, activated, hard, score
def siim_dice_overall(label, pred):
    n = pred.shape[0]
    pred = pred.view(n, -1)
    label = label.view(n, -1)
    intersect = (pred * label).sum(-1).float()
    union = (pred + label).sum(-1).float()
    u0 = union == 0
    intersect[u0] = 1
    union[u0] = 2
    return (2. * intersect / union)


# adapted from https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/98747
# work in numpy, hard, non-differentiable, score
def cmp_instance_dice(labels, preds, mean=False):
    '''
    instance dice score
    preds: list of N_i mask (0,1) per image - variable preds per image
    labels: list of M_i mask (0,1) target per image - variable labels per image
    '''

    scores = []
    for i in range(len(preds)):
        # Case when there is no GT mask
        if np.sum(preds[i]) == 0:
            scores.append(int(np.sum(preds[i]) == 0))
        # Case when there is no pred mask but there is GT mask
        elif np.sum(preds[i]) == 0:
            scores.append(0)
        # Case when there is both pred and gt masks
        else:
            m, _, _ = labels[i].shape
            n, _, _ = preds[i].shape

            label = labels[i].reshape(m, -1)
            pred = preds[i].reshape(n, -1)

            # intersect: matrix of target x preds (M, N)
            intersect = ((label[:, None, :] * pred[None, :, :]) > 0).sum(2)
            label_area, pred_area = label.sum(1), pred.sum(1)
            union = label_area[:, None] + pred_area[None, :]

            dice = (2 * intersect / union)

            dice_scores = dice[linear_sum_assignment(1 - dice)]
            mean_dice_score = sum(dice_scores) / max(n, m)  # unmatched gt or preds are counted as 0
            scores.append(mean_dice_score)
    if mean: return np.array(scores).mean()
    else: return np.array(scores)
