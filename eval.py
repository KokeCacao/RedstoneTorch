import torch
import torch.nn.functional as F

from dice_loss import dice_coeff


def eval_net(net, dataset, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    total_loss = 0
    num = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[2]

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)[0]
        # threshole transform from probability to solid mask
        mask_pred = (F.sigmoid(mask_pred) > 0.5).float()

        total_loss += dice_coeff(mask_pred, true_mask).item()
        num = i+1
    return total_loss / num
