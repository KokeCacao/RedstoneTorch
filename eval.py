import torch

from dice_loss import dice_coeff


def eval_net(net, validation_loader, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    total_loss = 0
    num = 0
    for batch_index, (id, z, image, true_mask) in enumerate(validation_loader, 0):

        # image = image.unsqueeze(0)
        # true_mask = true_mask.unsqueeze(0)

        if gpu:
            image = image.cuda()
            true_mask = true_mask.cuda()

        # why do you do [0]
        
        # masks_pred = net(image, z)
        masks_pred = net(image)

        masks_probs = torch.sigmoid(masks_pred)
        masks_probs_flat = masks_probs.view(-1)
        # threshole transform from probability to solid mask
        masks_probs_flat = (masks_probs_flat > 0.5).float()

        true_mask_flat = true_mask.view(-1)

        total_loss += dice_coeff(masks_probs_flat, true_mask_flat).item()
        num=num+1
    return total_loss / (num+0.1e-10)
