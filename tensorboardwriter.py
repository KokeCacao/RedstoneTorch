import os
import numpy as np

import config
from utils.memory import write_memory
import matplotlib as mpl

if os.environ.get('DISPLAY', '') == '':
    print('WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.')
    mpl.use('Agg')
from matplotlib import pyplot as plt

write_memory = write_memory


def write_text(writer, text, step):
    writer.add_text('text', text, global_step=step, walltime=None)


def write_loss(writer, loss_dict, global_step):
    writer.add_scalars('train/loss_scalar', loss_dict, global_step)


def write_threshold(writer, classes, score, threshold, fold):
    writer.add_scalars('threshold/threshold_distribution/'.format(fold), {"Class/{}".format(classes): score}, threshold)


def write_best_threshold(writer, classes, score, threshold, epoch, fold):
    writer.add_scalars('threshold/best_threshold/{}'.format(fold), {"Threshold/{}".format(classes): threshold}, epoch)


def write_data_distribution(writer, F, fold, unique=False):
    if unique:
        writer.add_figure("data/fold_distribution", F, 0)
        return
    writer.add_figure("data/fold_distribution/{}".format(fold), F, 0)


def write_loss_distribution(writer, loss_list, epoch):
    writer.add_histogram("eval/loss_distribution", loss_list, epoch)


def write_pred_distribution(writer, pred_list, epoch):
    writer.add_histogram("eval/pred_distribution", pred_list, epoch)


def write_pr_curve(writer, label, predicted, epoch, fold):
    writer.add_pr_curve("eval/pr_curve/{}".format(fold), label, predicted, epoch)


def write_image(writer, msg, F, epoch):
    writer.add_figure("eval/image/{}".format(msg), F, epoch)


def write_predict_image(writer, msg, F, epoch):
    writer.add_figure("predict/image/{}".format(msg), F, epoch)


def write_eval_loss(writer, loss_dict, epoch):
    writer.add_scalars('eval/loss_scalar', loss_dict, epoch)


def write_epoch_loss(writer, loss_dict, epoch):
    writer.add_scalars('eval/loss_epoch', loss_dict, epoch)


def write_best_img(writer, img, label, id, loss, fold):
    F = plt.figure()
    img = np_three_channel_with_green(img, (3, config.AUGMENTATION_IMG_ORIGINAL_SIZE, config.AUGMENTATION_IMG_ORIGINAL_SIZE), green_intensity=1, other_intensity=1)
    plt.imshow(img)
    plt.title("Id:{} label:{} loss:{}".format(id, label, loss))
    plt.grid(False)
    writer.add_figure("worst/{}".format(fold), F, 0)


def write_worst_img(writer, img, label, id, loss, fold):
    F = plt.figure()
    img = np_three_channel_with_green(img, (3, config.AUGMENTATION_IMG_ORIGINAL_SIZE, config.AUGMENTATION_IMG_ORIGINAL_SIZE), green_intensity=1, other_intensity=1)
    plt.imshow(img)
    plt.title("Id:{} label:{} loss:{}".format(id, label, loss))
    plt.grid(False)
    writer.add_figure("worst/{}".format(fold), F, 0)

def np_three_channel_with_green(image, shape, green_intensity=1, other_intensity=1):
    """

    :param image: ndarray with channel of (r, g, b, y), shape of (W, H, 4)
    :return: np_three_channel_with_green for mpl display
    """
    image = image.transpose((2, 0, 1))
    img = np.zeros(shape)
    img[0] = other_intensity*(0.5*image[0] + 0.25*image[3])/0.75
    img[1] = (green_intensity*0.5*image[1] + 0*image[3])/0.75
    img[2] = other_intensity*image[2]
    return np.stack(img/255., axis=-1)