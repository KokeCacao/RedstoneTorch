import sys
import random
from datetime import datetime
from optparse import OptionParser

import imgaug as ia
import numpy as np
import torch
from tensorboardX import SummaryWriter

import config
from gpu import gpu_profile
from project import hpa_project
from utils.memory import memory_thread


def get_args():
    parser = OptionParser()
    parser.add_option('--projecttag', dest='projecttag', default=False, help='tag you want to load')
    parser.add_option('--versiontag', dest='versiontag', default="", help='tag for tensorboard-log')
    parser.add_option('--loadfile', dest='loadfile', default=False, help='file you want to load')

    (options, args) = parser.parse_args()
    return options


def load_args():
    args = get_args()
    if args.versiontag: config.versiontag = args.versiontag
    if args.loadfile:
        config.PROJECT_TAG = args.projecttag
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
    print('     torch.__version__              = {}'.format(torch.__version__))
    print('     torch.version.cuda             = {}'.format(torch.version.cuda))
    print('     torch.backends.cudnn.version() = {}'.format(torch.backends.cudnn.version()))


if __name__ == '__main__':
    """
    PLAYGROUND
    """
    if not config.DEBUG_TEST_CODE:
        if config.DEBUG_TRAISE_GPU: sys.settrace(gpu_profile)
        load_args()

        writer = SummaryWriter(config.DIRECTORY_CHECKPOINT)
        print("=> Tensorboard: " + "python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/" + config.DIRECTORY_CHECKPOINT + " --port=6006")

        reproduceability()

        memory = memory_thread(1, writer)
        memory.setDaemon(True)
        memory.start()

        prediction = hpa_project.HPAPrediction