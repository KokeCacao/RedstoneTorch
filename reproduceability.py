import random
import imgaug as ia
import numpy as np
import torch
import cv2

import config

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

    # recommanded by https://github.com/albu/albumentations
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)