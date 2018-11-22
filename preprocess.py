import os
import random
import sys
from datetime import datetime
from optparse import OptionParser

import imgaug as ia
import numpy as np
import torch

import config
import project.hpa_project.hpa_preprocess
from gpu import gpu_profile


def get_args():
    parser = OptionParser()
    parser.add_option('--projecttag', dest='projecttag', default=False, help='tag you want to load')
    parser.add_option('--versiontag', dest='versiontag', default="", help='tag for tensorboard-log')
    parser.add_option('--loadfile', dest='loadfile', default=False, help='file you want to load')
    parser.add_option('--resume', dest='resume', default=False, help='resume or create a new folder')

    (options, args) = parser.parse_args()
    return options

def load_args():
    args = get_args()
    if args.versiontag: config.versiontag = args.versiontag
    if args.projecttag: config.PROJECT_TAG = args.projecttag
    config.TRAIN_RESUME = True if args.resume == "True" else False

    if args.loadfile:
        if config.TRAIN_RESUME:
            config.DIRECTORY_LOAD = config.DIRECTORY_PREFIX + "model/" + args.projecttag + "/" + args.loadfile
            config.DIRECTORY_CHECKPOINT = config.DIRECTORY_PREFIX + "model/" + config.PROJECT_TAG + "/"
        else:
            config.PROJECT_TAG = str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-") + "-" + config.PROJECT_TAG
            config.DIRECTORY_LOAD = config.DIRECTORY_PREFIX + "model/" + args.projecttag + "/" + args.loadfile
            config.DIRECTORY_CHECKPOINT = config.DIRECTORY_PREFIX + "model/" + config.PROJECT_TAG + "/"
    else:
        config.PROJECT_TAG = str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-") + "-" + config.PROJECT_TAG
        config.DIRECTORY_LOAD = None
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
    print('     torch.__version__              = {}'.format(torch.__version__))
    print('     torch.version.cuda             = {}'.format(torch.version.cuda))
    print('     torch.backends.cudnn.version() = {}'.format(torch.backends.cudnn.version()))

if __name__ == '__main__':
    """
    PLAYGROUND
    """

    config.DEBUG_TEST_CODE = False
    config.DEBUG_LAPTOP = False
    if not config.DEBUG_TEST_CODE:
        if config.DEBUG_TRAISE_GPU: sys.settrace(gpu_profile)
        load_args()
        reproduceability()

        print("=> Current Directory: " + str(os.getcwd()))
        print("=> Loading neuronetwork...")

        preprocess = project.hpa_project.hpa_preprocess.HPAPreprocess()

