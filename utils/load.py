import os
from os import listdir
from os.path import isfile, join

import torch
import torch.onnx
import torchvision
from torch.autograd import Variable

import config
import numpy as np

from utils.backup import keep_things_at_day

def set_milestone(file):
    if not file:
        print("No MILESTONE set.")
        return
    if os.path.isfile(file) and os.path.splitext(file)[1] == ".pth" and "-MILESTONE" not in os.path.splitext(file)[0]:
        new = os.path.splitext(file)[0]+"-MILESTONE"+os.path.splitext(file)[1]
        os.rename(file, new)
        print("Setting MILESTONE: {}".format(new))


def remove_checkpoint_fold(a=2, b=3):
    folder = config.DIRECTORY_CHECKPOINT
    cps = [f for f in listdir(folder) if isfile(join(folder, f)) and os.path.splitext(f)[1] == ".pth"]
    delete_nums = set(list(range(0, config.epoch))) - keep_things_at_day(config.epoch, a, b)
    # print("Check for CP removal: cps={}, delete_nums={}".format(cps, delete_nums))
    for cp in cps:
        if "-MILESTONE." in cp: continue
        for delete_num in delete_nums:
            if "-CP{}_".format(delete_num) in cp:
                print('Removing CP: {}'.format(folder + cp))
                os.remove(folder + cp)

def save_checkpoint_fold(state_dicts, optimizer_dicts, interupt=False):
    if interupt: print("WARNING: loading interupt models may be buggy")
    tag = config.versiontag + "-" if config.versiontag else ""
    interupt = "INTERUPT-" if interupt else ""
    if not os.path.exists(config.DIRECTORY_CHECKPOINT):
        os.makedirs(config.DIRECTORY_CHECKPOINT)
    config.lastsave = interupt + tag + config.DIRECTORY_CP_NAME.format(config.epoch, config.MODEL_TRAIN_FOLD, config.PROJECT_TAG, config.versiontag, config.MODEL_INIT_LEARNING_RATE, config.MODEL_BATCH_SIZE, config.AUGMENTATION_RESIZE)
    torch.save({
        'epoch': config.epoch,
        'global_steps': config.global_steps,
        'state_dicts': state_dicts,
        'optimizers': optimizer_dicts,
    }, config.DIRECTORY_CHECKPOINT + config.lastsave)
    print('Checkpoint: {} epoch; {}-{} step; dir: {}'.format(config.epoch, config.global_steps[0], config.global_steps[-1], config.DIRECTORY_CHECKPOINT + config.lastsave))


def load_checkpoint_all_fold(nets, optimizers, load_path):
    if not load_path or load_path == "False":
        config.epoch = 0
        config.global_steps = np.zeros(len(nets))
        print("=> Nothing loaded because no specify loadfile")
        return
    if not load_path or not os.path.isfile(load_path):
        load_path = os.path.splitext(load_path)[0]+"-MILESTONE"+os.path.splitext(load_path)[1]
    if load_path and os.path.isfile(load_path):
        print("=> Loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        if 'state_dicts' not in checkpoint:
            raise ValueError("=> Checkpoint is broken, nothing loaded")
        config.epoch = checkpoint['epoch']
        config.global_steps = checkpoint['global_steps']
        for fold, (net, optimizer) in enumerate(zip(nets, optimizers)):
            if fold not in config.MODEL_TRAIN_FOLD:
                continue
            net.load_state_dict(checkpoint['state_dicts'][fold])
            optimizer.load_state_dict(checkpoint['optimizers'][fold])
            # move_optimizer_to_cuda(optimizer)
            print("=> Loading checkpoint {} epoch; {} step".format(config.epoch, config.global_steps[fold]))
        print("=> Loaded checkpoint {} epoch; {}-{} step".format(config.epoch, config.global_steps[0], config.global_steps[-1]))
    else:
        raise ValueError("=> Nothing loaded because of invalid directory: {}".format(load_path))

def load_checkpoint_all_fold_without_optimizers(nets, load_path):
    if not load_path or load_path == "False":
        config.epoch = 0
        config.global_steps = np.zeros(len(nets))
        print("=> Nothing loaded because no specify loadfile")
        return
    if not load_path or not os.path.isfile(load_path):
        load_path = os.path.splitext(load_path)[0]+"-MILESTONE"+os.path.splitext(load_path)[1]
    if load_path and os.path.isfile(load_path):
        print("=> Loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        if 'state_dicts' not in checkpoint:
            raise ValueError("=> Checkpoint is broken, nothing loaded")
        config.epoch = checkpoint['epoch']
        config.global_steps = checkpoint['global_steps']
        for fold, net in enumerate(nets):
            if fold not in config.MODEL_TRAIN_FOLD:
                continue
            net.load_state_dict(checkpoint['state_dicts'][fold])
            print("=> Loading checkpoint {} epoch; {} step".format(config.epoch, config.global_steps[fold]))
        print("=> Loaded checkpoint {} epoch; {}-{} step".format(config.epoch, config.global_steps[0], config.global_steps[-1]))
    else:
        raise ValueError("=> Nothing loaded because of invalid directory: {}".format(load_path))


def load_checkpoint_one_fold(net, optimizer, fold, load_path):
    if not load_path or load_path == "False":
        config.epoch = 0
        config.global_steps = np.zeros(1)
        print("=> Nothing loaded because no specify loadfile")
        return
    if load_path and os.path.isfile(load_path):
        print("=> Loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        if 'state_dicts' not in checkpoint:
            raise ValueError("=> Checkpoint is broken, nothing loaded")
        config.epoch = checkpoint['epoch']
        config.global_steps = checkpoint['global_steps']
        net.load_state_dict(checkpoint['state_dict'][fold])
        optimizer.load_state_dict(checkpoint['optimizer'][fold])
        # move_optimizer_to_cuda(optimizer)
        print("=> Loading checkpoint {} epoch; {} step".format(config.epoch, config.global_steps[fold]))
        print("=> Loaded checkpoint {} epoch; {} step".format(config.epoch, config.global_steps[fold]))
    else:
        raise ValueError("=> Nothing loaded because of invalid directory: {}".format(load_path))


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
    return optimizer

def move_optimizer_to_cpu(optimizer):
    """
    Move the optimizer state to GPU, if necessary.
    After calling, any parameter specific state in the optimizer
    will be located on the same device as the parameter.
    """
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            # if not param.is_cuda:
            param_state = optimizer.state[param]
            for k in param_state.keys():
                if torch.is_tensor(param_state[k]):
                    param_state[k] = param_state[k].cpu()
    return optimizer

def save_onnx(cudaed_net, net_input_shape, dir, export_params=False, verbose=True):
    print("=> Start Saving ONNX file...")
    if os.path.exists(dir):
        os.remove(dir)
        print("WARNING: delete .onnx file '{}'".format(dir))

    # Standard ImageNet input - 3 channels, 224x224,
    # values don't matter as we care about network structure.
    # But they can also be real inputs.
    dummy_input = Variable(torch.randn(1, 3, 224, 224))
    # Obtain your model, it can be also constructed in your script explicitly
    model = torchvision.models.alexnet(pretrained=True)
    # Invoke export
    torch.onnx.export(model, dummy_input, dir)
    print("=> try good")

    if os.path.exists(dir):
        os.remove(dir)
        print("WARNING: delete .onnx file '{}'".format(dir))

    dummy_input = Variable(torch.randn(net_input_shape))
    torch.onnx.export(cudaed_net.cpu(), dummy_input, dir, export_params=export_params, verbose=verbose)
    print("=> Saving ONNX file correctly!")

def cuda(net):
    if config.TRAIN_GPU_ARG:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.TRAIN_GPU_ARG  # default
        # print('=> Using GPU: [' + config.TRAIN_GPU_ARG + ']')
        net.cuda()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # print('=> torch.cuda.device_count()      =', torch.cuda.device_count())
    return net
