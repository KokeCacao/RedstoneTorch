import torch

from dice_loss import dice_coeff


def eval_net(net, dataset, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    total_loss = 0
    num = 0
    for i, b in enumerate(dataset):
        imgs = b[0]
        depths = b[1]
        true_mask = b[2]

        imgs = torch.from_numpy(imgs).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)

        if gpu:
            imgs = imgs.cuda()
            true_mask = true_mask.cuda()

        # why do you do [0]
        mask_pred = net(imgs, depths)[0]
        # threshole transform from probability to solid mask
        mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()

        total_loss += dice_coeff(mask_pred, true_mask).item()
        num=num+1
    return total_loss / num
