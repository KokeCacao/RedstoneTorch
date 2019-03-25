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


def write_find_lr(writer, loss, lr):
    writer.add_scalars('threshold/find_lr/', {"LRx100000": loss}, int(lr*100000))


def write_best_threshold(writer, classes, score, threshold, epoch, fold):
    writer.add_scalars('threshold/best_threshold/{}'.format(fold), {"Threshold/{}".format(classes): threshold}, epoch)


def write_data_distribution(writer, F, fold, unique=False):
    if unique:
        writer.add_figure("data/fold_distribution", F, 0)
        return
    writer.add_figure("data/fold_distribution/{}".format(fold), F, 0)

def write_shakeup(writer, dictionary, sorted_keys, epoch):
    for i, key in enumerate(sorted_keys):
        public_lb, private_lb = dictionary[key]
        writer.add_scalars('threshold/Shakeup/', {"Public LB": public_lb}, i)
        writer.add_scalars('threshold/Shakeup/', {"Private LB": private_lb}, i)
    writer.add_histogram("eval/loss_distribution", sorted_keys, epoch)


def write_loss_distribution(writer, loss_list, epoch):
    writer.add_histogram("eval/loss_distribution", loss_list, epoch)


def write_pred_distribution(writer, pred_list, epoch):
    writer.add_histogram("eval/pred_distribution", pred_list, epoch)


def write_pr_curve(writer, label, predicted, epoch, fold):
    writer.add_pr_curve("eval/pr_curve/{}".format(fold), label, predicted, epoch)


def write_image(writer, msg, F, epoch):
    writer.add_figure("eval/image/{}".format(msg), F, 0)


def write_predict_image(writer, msg, F, epoch):
    writer.add_figure("predict/image/{}".format(msg), F, epoch)


def write_eval_loss(writer, loss_dict, epoch):
    writer.add_scalars('eval/loss_scalar', loss_dict, epoch)


def write_epoch_loss(writer, loss_dict, epoch):
    writer.add_scalars('eval/loss_epoch', loss_dict, epoch)


def write_best_img(writer, img, label, id, loss, fold):
    F = plt.figure()
    plt.imshow(img)
    plt.title("Id:{} label:{} loss:{}".format(id.split("/")[-1], label, loss))
    plt.grid(False)
    writer.add_figure("worst/{}".format(fold), F, 0)

def write_focus(writer, id, cam, img, label, pred, n, fold):
    F = plt.figure()
    plt.imshow(cam)
    plt.title("Id:{} label:{} pred:{}".format(id, label, pred))
    plt.grid(False)
    writer.add_figure("focus/#{}-{}-cam".format(n, fold), F, 0)

    F = plt.figure()
    plt.imshow(img)
    plt.title("Id:{} label:{} pred:{}".format(id, label, pred))
    plt.grid(False)
    writer.add_figure("focus/#{}-{}-img".format(n, fold), F, 0)

def write_plot(writer, F, message):
    writer.add_figure("figure/{}".format(message), F, 0)

def write_worst_img(writer, img, label, id, loss, fold):
    F = plt.figure()
    plt.imshow(img)
    plt.title("Id:{} label:{} loss:{}".format(id.split("/")[-1], label, loss))
    plt.grid(False)
    writer.add_figure("worst/{}".format(fold), F, 0)

def valid_imshow_data(data):
    data = np.asarray(data)
    if data.ndim == 2:
     return True
    elif data.ndim == 3:
     if 3 <= data.shape[2] <= 4:
      return True
     else:
      print('The "data" has 3 dimensions but the last dimension ' 
        'must have a length of 3 (RGB) or 4 (RGBA), not "{}".' 
        ''.format(data.shape[2]))
      return False
    else:
     print('To visualize an image the data must be 2 dimensional or ' 
       '3 dimensional, not "{}".' 
       ''.format(data.ndim))
     return False
