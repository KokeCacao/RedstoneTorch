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
    writer.add_scalars('threshold/threshold_distribution', {"Class/{}".format(classes): score}, threshold)

def write_threshold_class(writer, best_threshold_dict, best_val_dict, class_list, dic):
    current_freq = -1
    current_thres = 0
    current_val = 0
    current_count = 0
    freq = -1

    for c in class_list:
        freq = dic[c]
        thres = best_threshold_dict[c]
        val = best_val_dict[c]
        if current_freq == freq:
            current_count = current_count + 1
            current_thres = current_thres + thres
            current_val = current_val + val
        else:
            if current_count != 0: writer.add_scalars('threshold/threshold_frequency', {"Threshold": current_thres/current_count, "Validation": current_val/current_count}, freq)

            current_freq = freq
            current_thres = thres
            current_val = val
            current_count = 1
    writer.add_scalars('threshold/threshold_frequency', {"Threshold": current_thres/current_count, "Validation": current_val/current_count}, freq)


def write_find_lr(writer, loss, lr):
    writer.add_scalars('threshold/find_lr/', {"LRx100000": loss}, int(lr*100000))


def write_best_threshold(writer, classes, score, threshold, area_under, epoch, fold):
    if classes == -1:
        writer.add_scalars('threshold/best_threshold/{}'.format(fold), {"Score/{}".format(classes): score,
                                                                        "Threshold/{}".format(classes): threshold,
                                                                        "AreaUnder/{}".format(classes): area_under}, epoch)
    else:
        writer.add_scalars('threshold/best_threshold/{}'.format(fold), {"Threshold/{}".format(classes): threshold}, epoch)


def write_data_distribution(writer, F, fold, unique=False):
    if unique:
        writer.add_figure("data/fold_distribution", F, 0)
        return
    writer.add_figure("data/fold_distribution/{}".format(fold), F, 0)

def write_shakeup(writer, dictionary, sorted_keys, std, epoch):
    for i, key in enumerate(sorted_keys):
        public_lb, private_lb = dictionary[key]
        writer.add_scalars('threshold/Shakeup/', {"Public LB": public_lb}, i)
        writer.add_scalars('threshold/Shakeup/', {"Private LB": private_lb}, i)
    writer.add_scalars('threshold/shake_up_change/', {"Standard Deviation": std}, epoch)
    try:
        writer.add_histogram("eval/loss_distribution", np.array(sorted_keys), epoch)
    except Exception as e:
        print("Having some trouble writing histogram: `writer.add_histogram(\"eval/loss_distribution\", sorted_keys, epoch)`")


def write_loss_distribution(writer, loss_list, epoch):
    writer.add_histogram("eval/skfacc_distribution", loss_list, epoch)

def write_classwise_loss_distribution(writer, loss_list, epoch):
    writer.add_histogram("eval/classwise_skfacc_distribution", loss_list, epoch)


def write_pred_distribution(writer, pred_list, epoch):
    writer.add_histogram("eval/pred_distribution", pred_list, epoch)


def write_pr_curve(writer, label, predicted, epoch, fold):
    writer.add_pr_curve("eval/pr_curve/{}".format(fold), label, predicted, epoch)


def write_image(writer, msg, F, epoch, category="image"):
    writer.add_figure("eval/{}/{}".format(category, msg), F, 0)


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
