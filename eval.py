import os

import matplotlib as mpl
import numpy as np
import torch
import config
from PIL import ImageChops
from torchvision import transforms

if os.environ.get('DISPLAY','') == '':
    print('WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.')
    mpl.use('Agg')
from matplotlib import pyplot as plt

global_plot_step = 0
def eval_net(net, validation_loader, dataset, gpu, visualization, writer, epoch_num=0):
    thresold_dict = dict()
    """Evaluation without the densecrf with the dice coefficient"""
    # total_loss = 0
    total_ious = []
    for batch_index, (id, z, image, true_mask, image_0, true_mask_0) in enumerate(validation_loader, 0):

        if gpu != "":
            # z = z.cuda()
            image = image.cuda()
            true_mask = true_mask.cuda()

        masks_pred = net(image).repeat(1, 3, 1, 1)
        ious = iou_score(masks_pred, true_mask, threshold=0.5)
        if config.TRAIN_THRESHOLD_TEST == True:
            for threshold in config.TRAIN_TRY_THRESHOLD:
                if thresold_dict.get(threshold) == None:
                    thresold_dict.update({threshold: [iou_score(masks_pred, true_mask, threshold).mean().float()]})
                else:
                    thresold_dict.update({threshold, thresold_dict.get(threshold).append(iou_score(masks_pred, true_mask, threshold).mean().float())})
        total_ious = total_ious + list(ious)
        # iou = ious.mean().float()

        if visualization and batch_index==0:
            writer.add_pr_curve("loss/epoch_validation_image", true_mask, masks_pred, global_step=epoch_num)
            global global_plot_step
            global_plot_step=global_plot_step+1
            for index, input_id in enumerate(id):
                F = plt.figure()

                plt.subplot(321)
                plt.imshow(tensor_to_PIL(image_0[index]))
                plt.title("Image_Real")
                plt.grid(False)

                plt.subplot(322)
                plt.imshow(tensor_to_PIL(image[index]))
                plt.title("Image_Trans")
                plt.grid(False)

                plt.subplot(323)
                plt.imshow(tensor_to_PIL(true_mask_0[index]))
                plt.title("Mask_Real")
                plt.grid(False)

                plt.subplot(324)
                plt.imshow(tensor_to_PIL(true_mask[index]))
                plt.title("Mask_Trans")
                plt.grid(False)

                plt.subplot(325)
                plt.imshow(ImageChops.difference(tensor_to_PIL(true_mask[index]), tensor_to_PIL(masks_pred[index])))
                plt.title("Error: {}".format(ious[index]))
                plt.grid(False)

                plt.subplot(326)
                plt.imshow(tensor_to_PIL(masks_pred[index]))
                plt.title("Predicted")
                plt.grid(False)
                writer.add_figure("image/epoch_validation/"+str(index)+"img", F, global_step=global_plot_step)
        # print("iou:", iou.mean())

        # masks_probs = torch.sigmoid(masks_pred)
        # masks_probs_flat = masks_probs.view(-1)
        # # threshole transform from probability to solid mask
        # masks_probs_flat = (masks_probs_flat > 0.5).float()
        #
        # true_mask_flat = true_mask.view(-1)
        #
        # total_loss += dice_coeff(masks_probs_flat, true_mask_flat).item()
        # return total_loss / (num+1e-10)
        del id, z, image, true_mask
        if gpu != "": torch.cuda.empty_cache()  # release gpu memory

    for key, item in thresold_dict.items():
        item = np.mean(item)
        writer.add_scalars('val/threshold', {'Thresold': item}, key)

    writer.add_scalars('val/max_threshold', {'MaxThresold': np.max(thresold_dict.values())}, global_plot_step)

    writer.add_histogram("iou", total_ious, global_plot_step)
    return total_ious.mean().float()

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    if image.size()[0] == 1: image = image.repeat(3, 1, 1) # from gray sacale to RGB
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image


def iou_score(outputs, labels, threshold=0.5):
    outputs = outputs > threshold # threshold

    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1).byte()  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1).byte()

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zero if both are 0

    iou = (intersection + 1e-10) / (union + 1e-10)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  # Or thresholded.mean() if you are interested in average across the batch



# def calculate_scores(y_true, y_pred):
#     iou = intersection_over_union(y_true, y_pred)
#     iout = intersection_over_union_thresholds(y_true, y_pred)
#     return iou, iout
#
# def intersection_over_union(y_true, y_pred):
#     ious = []
#     for y_t, y_p in list(zip(y_true, y_pred)):
#         iou = compute_ious(y_t, y_p)
#         iou_mean = 1.0 * np.sum(iou) / len(iou)
#         ious.append(iou_mean)
#     return np.mean(ious)
#
#
# def intersection_over_union_thresholds(y_true, y_pred):
#     iouts = []
#     for y_t, y_p in list(zip(y_true, y_pred)):
#         iouts.append(compute_eval_metric(y_t, y_p))
#     return np.mean(iouts)
#
# def compute_ious(gt, predictions):
#     gt_ = get_segmentations(gt)
#     predictions_ = get_segmentations(predictions)
#
#     if len(gt_) == 0 and len(predictions_) == 0:
#         return np.ones((1, 1))
#     elif len(gt_) != 0 and len(predictions_) == 0:
#         return np.zeros((1, 1))
#     else:
#         iscrowd = [0 for _ in predictions_]
#         ious = cocomask.iou(gt_, predictions_, iscrowd)
#         if not np.array(ious).size:
#             ious = np.zeros((1, 1))
#         return ious
#
# def get_segmentations(labeled):
#     nr_true = labeled.max()
#     segmentations = []
#     for i in range(1, nr_true + 1):
#         msk = labeled == i
#         segmentation = rle_from_binary(msk.astype('uint8'))
#         segmentation['counts'] = segmentation['counts'].decode("UTF-8")
#         segmentations.append(segmentation)
#     return segmentations
#
# def rle_from_binary(prediction):
#     prediction = np.asfortranarray(prediction)
#     return cocomask.encode(prediction)
#
# def compute_eval_metric(gt, predictions):
#     thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
#     ious = compute_ious(gt, predictions)
#     precisions = [compute_precision_at(ious, th) for th in thresholds]
#     return sum(precisions) / len(precisions)
#
# def compute_precision_at(ious, threshold):
#     mx1 = np.max(ious, axis=0)
#     mx2 = np.max(ious, axis=1)
#     tp = np.sum(mx2 >= threshold)
#     fp = np.sum(mx2 < threshold)
#     fn = np.sum(mx1 < threshold)
#     return float(tp) / (tp + fp + fn)