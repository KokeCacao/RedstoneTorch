import os

import torch

import config
import numpy as np


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
        config.epoch = np.zeros(len(nets))
        config.global_steps = np.zeros(len(nets))


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
        config.epoch = np.zeros(1)
        config.global_steps = np.zeros(1)


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


def cuda(net):
    if config.TRAIN_GPU_ARG:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.TRAIN_GPU_ARG  # default
        print('=> Using GPU: [' + config.TRAIN_GPU_ARG + ']')
        net.cuda()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('=> torch.cuda.device_count()      =', torch.cuda.device_count())
    return net