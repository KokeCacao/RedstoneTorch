import random
import sys
import os
import numpy as np
from optparse import OptionParser

import torch

import config
import imgaug as ia

from project.TGSProject import TGSProject
from model.resunet.resunet_model import UNetResNet
from datetime import datetime
from tensorboardX import SummaryWriter

# dir_prefix = 'drive/My Drive/ML/Pytorch-UNet/'
from utils.memory import memory_thread


def get_args():
    parser = OptionParser()
    parser.add_option('-c', '--load', dest='load', default=False, help='load file model')
    parser.add_option('-t', '--tag', dest='tag', default="", help='tag for tensorboard-log')
    # parser.add_option('-e', '--continue', dest='continu', default=False, help='continue in the same folder (but potentially break down the statistics')

    (options, args) = parser.parse_args()
    return options


def log_data(file_name, data):
    with open(file_name + ".txt", "a+") as file:
        file.write(data + "\n")


def save_checkpoint(state_dict, optimizer_dict, interupt=False):
    tag = config.tag + "-" if config.tag != "" else ""
    interupt = "INTERUPT-" if interupt else ""
    if config.TRAIN_SAVE_CHECKPOINT:
        if not os.path.exists(config.DIRECTORY_CHECKPOINT):
            os.makedirs(config.DIRECTORY_CHECKPOINT)
    torch.save({
        'epoch': config.epoch,
        'global_step': config.global_step,
        'state_dict': state_dict,
        'optimizer': optimizer_dict,
    }, config.DIRECTORY_CHECKPOINT + interupt + tag + config.DIRECTORY_CP_NAME.format(config.epoch))
    print('Checkpoint: {} step, dir: {}'.format(config.global_step, config.DIRECTORY_CHECKPOINT + interupt + config.tag + config.DIRECTORY_CP_NAME.format(config.epoch)))


def load_checkpoint(net, optimizer, load_path):
    if load_path and os.path.isfile(load_path):
        print("=> Loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        if 'state_dict' not in checkpoint:
            net.load_state_dict(checkpoint)
            print("=> Loaded only the model")
            return
        config.epoch = checkpoint['epoch']
        config.global_step = checkpoint['global_step']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        move_optimizer_to_cuda(optimizer)
        print("=> Loaded checkpoint 'epoch = {}' (global_step = {})".format(config.epoch, config.global_step))
    else:
        print("=> Nothing loaded")


def move_optimizer_to_cuda(optimizer):
    """
    Move the optimizer state to GPU, if necessary.
    After calling, any parameter specific state in the optimizer
    will be located on the same device as the parameter.
    """
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            # if param.is_cuda:
            param_state = optimizer.state[param]
            for k in param_state.keys():
                if torch.is_tensor(param_state[k]):
                    param_state[k] = param_state[k].cuda()


def load_args():
    args = get_args()
    if args.tag != "":
        # if args.continu and args.loca != False: config.TRAIN_TAG = args.load.split("/", 2)[1]
        # else: config.TRAIN_TAG = str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-") + "-" + args.tag
        config.tag = args.tag
        config.TRAIN_TAG = str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-") + "-" + config.tag
        config.DIRECTORY_CHECKPOINT = config.DIRECTORY_PREFIX + "tensorboard/" + config.TRAIN_TAG + "/checkpoints/"
    if args.load != None:
        config.TRAIN_LOAD = args.load
        if config.TRAIN_CONTINUE:
            config.TRAIN_TAG = args.load.split("/", 3)[1]
            config.DIRECTORY_CHECKPOINT = config.DIRECTORY_PREFIX + "tensorboard/" + config.TRAIN_TAG + "/checkpoints/"


# from ml-arsenal-public/blob/master/reproducibility.py
"""The main python module that is run should import random and call random.seed(n) - this is shared between all other imports of random as long as somewhere else doesn't reset the seed."""


def reproduceability():
    print('=> Setting random seed to {}.'.format(config.TRAIN_SEED))
    print('')
    ia.seed(config.TRAIN_SEED)
    random.seed(config.TRAIN_SEED)
    np.random.seed(config.TRAIN_SEED)
    torch.manual_seed(config.TRAIN_SEED)
    torch.cuda.manual_seed_all(config.TRAIN_SEED)

    print('=> Enable torch.backends.cudnn...')
    print('')
    torch.backends.cudnn.benchmark = True  # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.deterministic = True  # deterministic result
    torch.backends.cudnn.enabled = True  # enable
    print('=> Setting CUDA environment...')
    print('=> torch.__version__              =', torch.__version__)
    print('=> torch.version.cuda             =', torch.version.cuda)
    print('=> torch.backends.cudnn.version() =', torch.backends.cudnn.version())


def cuda(net):
    if config.TRAIN_GPU_ARG != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = config.TRAIN_GPU_ARG  # default
        print('=> Using GPU: [' + config.TRAIN_GPU_ARG + ']')
        net.cuda()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('=> torch.cuda.device_count()      =', torch.cuda.device_count())
    print('')
    return net


if __name__ == '__main__':
    load_args()

    writer = SummaryWriter("tensorboard/" + config.TRAIN_TAG)
    print("=> Tensorboard: " + "python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/tensorboard/" + config.TRAIN_TAG + " --port=6006")

    reproduceability()

    memory = memory_thread(1, writer, config.TRAIN_GPU_ARG)
    memory.setDaemon(True)
    memory.start()

    print("=> Current Directory: " + str(os.getcwd()))
    print("=> Loading neuronetwork...")

    TGSProject.run
