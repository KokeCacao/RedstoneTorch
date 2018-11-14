import os
import matplotlib as mpl
from utils.memory import write_memory

if os.environ.get('DISPLAY', '') == '':
    print('WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.')
    mpl.use('Agg')
from matplotlib import pyplot as plt

def write_loss(writer, loss_dict, global_step):
    writer.add_scalars('train/loss_scalar', loss_dict, global_step)

def write_threshold(writer, fold_value_dict, threshold):
    writer.add_scalars('train/threshold_distribution', fold_value_dict, threshold)

def write_best_threshold(writer, threshold, f1, epoch):
    writer.add_scalars('train/best_threshold', {"Threshold":threshold, "F1": f1}, epoch)

def write_loss_distribution(writer, loss_list, epoch):
    writer.add_histogram("eval/loss_distribution", loss_list, epoch)

def write_pred_distribution(writer, pred_list, epoch):
    writer.add_histogram("eval/pred_distribution", pred_list, epoch)

def write_pr_curve(writer, label, predicted, epoch, fold):
    writer.add_pr_curve("eval/pr_curve/{}".format(fold), label, predicted, epoch)

def write_image(writer, msg, F, epoch):
    writer.add_figure("eval/image/" + msg, F, epoch)

def write_predict_image(writer, msg, F, epoch):
    writer.add_figure("predict/image/" + msg, F, epoch)

def write_eval_loss(writer, loss_dict, epoch):
    writer.add_scalars('eval/loss_scalar', loss_dict, epoch)

def write_epoch_loss(writer, loss_dict, epoch):
    writer.add_scalars('eval/loss_epoch', loss_dict, epoch)

def write_best_img(writer, img, label, id, loss, fold):
    F = plt.imshow(img)
    plt.title("id: {}; label: {}; loss: {}".format(id, label, loss))
    plt.grid(False)
    writer.add_figure("eval/best", F, fold)

def write_worst_img(writer, img, label, id, loss, fold):
    F = plt.imshow(img)
    plt.title("id: {}; label: {}; loss: {}".format(id, label, loss))
    plt.grid(False)
    writer.add_figure("eval/worst", F, fold)