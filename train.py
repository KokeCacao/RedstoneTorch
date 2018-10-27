import random
import sys
import os
import numpy as np
from optparse import OptionParser

import torch

import config
import imgaug as ia

from datetime import datetime
from tensorboardX import SummaryWriter

# dir_prefix = 'drive/My Drive/ML/Pytorch-UNet/'
from utils.memory import memory_thread
from project import hpa_project


def log_data(file_name, data):
    with open(file_name + ".txt", "a+") as file:
        file.write(data + "\n")


def save_checkpoint_fold(state_dicts, optimizer_dicts, interupt=False):
    if interupt: print("WARNING: loading interupt models may be buggy")
    tag = config.versiontag + "-" if config.versiontag else ""
    interupt = "INTERUPT-" if interupt else ""
    if config.TRAIN_SAVE_CHECKPOINT:
        if not os.path.exists(config.DIRECTORY_CHECKPOINT):
            os.makedirs(config.DIRECTORY_CHECKPOINT)
    config.lastsave = interupt + tag + config.DIRECTORY_CP_NAME.format(config.epoch)
    torch.save({
        'epoch': config.epoch,
        'global_steps': config.global_steps,
        'state_dicts': state_dicts,
        'optimizers': optimizer_dicts,
    }, config.DIRECTORY_CHECKPOINT + config.lastsave)
    print('Checkpoint: {} epoch; {}-{} step; dir: {}'.format(config.epoch, config.global_steps[0], config.global_steps[-1], config.DIRECTORY_CHECKPOINT + interupt + config.versiontag + config.DIRECTORY_CP_NAME.format(config.epoch)))


def load_checkpoint_all_fold(nets, optimizers, load_path):
    if load_path and os.path.isfile(load_path):
        print("=> Loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        if 'state_dicts' not in checkpoint:
            print("=> Checkpoint is broken, nothing loaded")
            return
        config.epoch = checkpoint['epoch']
        config.global_steps = checkpoint['global_steps']
        for fold, (net, optimizer) in zip(nets, optimizers):
            net.load_state_dict(checkpoint['state_dict'][fold])
            optimizer.load_state_dict(checkpoint['optimizer'][fold])
            move_optimizer_to_cuda(optimizer)
            print("=> Loading checkpoint {} epoch; {} step".format(config.epoch, config.global_steps[fold]))
        print("=> Loaded checkpoint {} epoch; {}-{} step".format(config.epoch, config.global_steps[0], config.global_steps[-1]))
    else:
        print("=> Nothing loaded because of invalid directory")


def load_checkpoint_one_fold(net, optimizer, fold, load_path):
    if load_path and os.path.isfile(load_path):
        print("=> Loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        if 'state_dicts' not in checkpoint:
            print("=> Checkpoint is broken, nothing loaded")
            return
        config.epoch = checkpoint['epoch']
        config.global_steps = checkpoint['global_steps']
        net.load_state_dict(checkpoint['state_dict'][fold])
        optimizer.load_state_dict(checkpoint['optimizer'][fold])
        move_optimizer_to_cuda(optimizer)
        print("=> Loading checkpoint {} epoch; {} step".format(config.epoch, config.global_steps[fold]))
        print("=> Loaded checkpoint {} epoch; {} step".format(config.epoch, config.global_steps[fold]))
    else:
        print("=> Nothing loaded because of invalid directory")


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


def get_args():
    parser = OptionParser()
    parser.add_option('--projecttag', dest='projecttag', default=False, help='tag you want to load')
    parser.add_option('--versiontag', dest='versiontag', default="", help='tag for tensorboard-log')
    parser.add_option('--loadfile', dest='loadfile', default=False, help='file you want to load')
    parser.add_option('--resume', dest='resume', default=True, help='resume or create a new folder')

    (options, args) = parser.parse_args()
    return options


def load_args():
    args = get_args()
    if args.versiontag: config.versiontag = args.versiontag
    if args.projecttag:
        config.TRAIN_RESUME = args.resume
        if config.TRAIN_RESUME:
            config.PROJECT_TAG = args.projecttag
            config.DIRECTORY_LOAD = config.DIRECTORY_PREFIX + "model/" + args.projecttag + "/" + args.loadfile
            config.DIRECTORY_CHECKPOINT = config.DIRECTORY_PREFIX + "model/" + config.PROJECT_TAG + "/"
        else:
            config.PROJECT_TAG = str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-") + "-" + config.versiontag
            config.DIRECTORY_LOAD = config.DIRECTORY_PREFIX + "model/" + args.projecttag + "/" + args.loadfile
            config.DIRECTORY_CHECKPOINT = config.DIRECTORY_PREFIX + "model/" + config.PROJECT_TAG + "/"


def reproduceability():
    """
        from ml-arsenal-public/blob/master/reproducibility.py
        The main python module that is run should import random and call random.seed(n)
        this is shared between all other imports of random as long as somewhere else doesn't reset the seed.
    """
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
    if config.TRAIN_GPU_ARG:
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

    writer = SummaryWriter(config.DIRECTORY_CHECKPOINT)
    print("=> Tensorboard: " + "python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=MineRedstone/" + config.DIRECTORY_CHECKPOINT + " --port=6006")

    reproduceability()

    memory = memory_thread(1, writer)
    memory.setDaemon(True)
    memory.start()

    print("=> Current Directory: " + str(os.getcwd()))
    print("=> Loading neuronetwork...")

    project = hpa_project.HPAProject()
