import os
import sys
import pickle
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

def save_checkpoint_fold(state_dicts, optimizer_dicts, lr_schedulers_dicts, interupt=False):
    if interupt: print("WARNING: loading interupt models may be buggy")
    tag = config.versiontag + "-" if config.versiontag else ""
    interupt = "INTERUPT-" if interupt else ""
    if not os.path.exists(config.DIRECTORY_CHECKPOINT):
        os.makedirs(config.DIRECTORY_CHECKPOINT)
    config.lastsave = interupt + tag + config.DIRECTORY_CP_NAME.format(config.epoch, config.train_fold, config.PROJECT_TAG, config.versiontag, config.MODEL_INIT_LEARNING_RATE, config.MODEL_BATCH_SIZE, config.AUGMENTATION_RESIZE)
    torch.save({
        'epoch': config.epoch,
        'global_steps': config.global_steps,
        'state_dicts': state_dicts,
        'optimizers': optimizer_dicts,
        'lr_schedulers': lr_schedulers_dicts,
    }, config.DIRECTORY_CHECKPOINT + config.lastsave)
    print("""
        Checkpoint: {} epoch; {}-{} step; dir: {}""".format(config.epoch, config.global_steps[0], config.global_steps[-1], config.DIRECTORY_CHECKPOINT + config.lastsave))

def load_unsave(model, state_dict):
    model_state = model.state_dict()
    pretrained_state = {k: v for k, v in state_dict.items() if k in model_state and v.size() == model_state[k].size()}
    print("Loaded State Dict: {}".format(pretrained_state.keys()))
    model_state.update(pretrained_state)
    model.load_state_dict(model_state, strict=False)

    if config.freeze_loaded:
        """freezing the loaded parameters"""
        for name, param in model.named_parameters():
            if name in pretrained_state.keys():
                param.requires_grad = False
                print("Set {} require_grad = False".format(name))


def load_checkpoint_all_fold(nets, optimizers, lr_schedulers, load_path):
    if not load_path or load_path == "False":
        config.epoch = 0
        config.global_steps = np.zeros(len(nets))
        print("=> Nothing loaded because no specify loadfile")
        return
    if not load_path or not os.path.isfile(load_path):
        load_path = os.path.splitext(load_path)[0]+"-MILESTONE"+os.path.splitext(load_path)[1]
    if load_path and os.path.isfile(load_path):
        print("=> Loading checkpoint '{}'".format(load_path))
        checkpoint = load_file(load_path)
        if 'state_dicts' not in checkpoint:
            raise ValueError("=> Checkpoint is broken, nothing loaded")

        if config.load_epoch:
            config.epoch = checkpoint['epoch']
            config.global_steps = checkpoint['global_steps']
        else:
            config.epoch = 0
            config.global_steps = 0

        optimizers = [None] * len(nets) if optimizers is None else optimizers
        lr_schedulers = [None] * len(nets) if lr_schedulers is None else lr_schedulers
        for fold, (net, optimizer, lr_scheduler) in enumerate(zip(nets, optimizers, lr_schedulers)):
            if fold not in config.train_fold:
                continue

            if config.load_state_dicts:
                if 'state_dicts' in checkpoint.keys():
                    if fold >= len(checkpoint['state_dicts']):
                        net.load_state_dict(checkpoint['state_dicts'][0])
                        print("[WARNING] No state_dict for the fold found, loading checkpoint['state_dicts'][0]")
                    else:
                        if checkpoint['state_dicts'][fold] is None: print("[ERROR] The fold number of your input is not correct or no fold found")
                        try:
                            net.load_state_dict(checkpoint['state_dicts'][fold])
                        except RuntimeError as e:
                            print("[WARNING] State dict has something missing: {}".format(e))
                            load_unsave(net, checkpoint['state_dicts'][fold])
                else:
                    print("[WARNING] No keys [state_dicts] detected from loading")
            else:
                print("[MESSAGE] No state_dicts loaded because of your settings")

            if config.load_optimizers:
                if 'optimizers' in checkpoint.keys():
                    if fold >= len(checkpoint['optimizers']):
                        optimizer.load_state_dict(checkpoint['optimizers'][0]) # BAD CODE
                        print("[WARNING] No optimizer for the fold found, loading checkpoint['optimizers'][0]")
                    else:
                        if checkpoint['optimizers'][fold] is None: print("[ERROR] The fold number of your input is not correct or no fold found")
                        optimizer.load_state_dict(checkpoint['optimizers'][fold]) # BAD CODE
                else:
                    print("[WARNING] No keys [optimizers] detected from loading")
            else:
                print("[MESSAGE] No optimizers loaded because of your settings")

            if config.load_lr_schedulers:
                if 'lr_schedulers' in checkpoint.keys():
                    if fold >= len(checkpoint['lr_schedulers']):
                        lr_scheduler.load_state_dict(checkpoint['lr_schedulers'][0]) # BAD CODE
                        print("[WARNING] No lr_schedulers for the fold found, loading checkpoint['lr_schedulers'][0]")
                    else:
                        if checkpoint['lr_schedulers'][fold] is None: print("[ERROR] The fold number of your input is not correct or no fold found")
                        lr_scheduler.load_state_dict(checkpoint['lr_schedulers'][fold]) # BAD CODE
                else:
                    print("[WARNING] No keys [lr_schedulers] detected from loading")
            else:
                print("[MESSAGE] No lr_schedulers loaded because of your settings")

            # move_optimizer_to_cuda(optimizer)
            if fold < len(config.global_steps): print("=> Loading checkpoint {} epoch; {} step".format(config.epoch, config.global_steps[fold]))
            else: print("=> Loading checkpoint {} epoch; {} step".format(config.epoch, config.global_steps[0]))
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
        checkpoint = load_file(load_path)
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

def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_file(file):
    if sys.version_info[0] < 3:
        return torch.load(file)
    else:
        from functools import partial
        import pickle
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        return torch.load(file, map_location=lambda storage, loc: storage, pickle_module=pickle)