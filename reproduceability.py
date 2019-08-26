import random
import sys
from subprocess import call

import imgaug as ia
import numpy as np
import torch
import cv2

import config

def reproduceability(log=None):
    """
        from ml-arsenal-public/blob/master/reproducibility.py
        The main python module that is run should import random and call random.seed(n)
        this is shared between all other imports of random as long as somewhere else doesn't reset the seed.
    """
    if log is None:
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
        torch.backends.cudnn.deterministic = False  # deterministic result
        torch.backends.cudnn.enabled = True  # enable
        print('=> Setting CUDA environment...')
        print('     torch.__version__              = {}'.format(torch.__version__))
        print('     torch.version.cuda             = {}'.format(torch.version.cuda))
        print('     torch.backends.cudnn.version() = {}'.format(torch.backends.cudnn.version()))

        # recommanded by https://github.com/albu/albumentations
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        print('__Python VERSION: {}'.format(sys.version))
        print('__pyTorch VERSION: {}'.format(torch.__version__))
        print('__CUDA VERSION: {}'.format(torch.version.cuda))
        # call(["nvcc", "-V"])
        print('__CUDNN VERSION: {}'.format(torch.backends.cudnn.version()))
        print('__Number CUDA Devices: {}'.format(torch.cuda.device_count()))
        print('__Devices')
        call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
        print('Active CUDA Device: GPU{}'.format(torch.cuda.current_device()))

        print('Available devices: {}'.format(torch.cuda.device_count()))
        print('Current cuda device: {}'.format(torch.cuda.current_device()))

        try:
            import pycuda
            from pycuda import compiler
            import pycuda.driver as drv

            drv.init()
            print("%d device(s) found." % drv.Device.count())

            for ordinal in range(drv.Device.count()):
                dev = drv.Device(ordinal)
                print(ordinal, dev.name())
        except ImportError:
            print("No pycuda found, skip printing device")
    else:
        log.write('=> Setting random seed to {}.'.format(config.TRAIN_SEED))
        log.write('')
        ia.seed(config.TRAIN_SEED)
        random.seed(config.TRAIN_SEED)
        np.random.seed(config.TRAIN_SEED)
        torch.manual_seed(config.TRAIN_SEED)
        torch.cuda.manual_seed_all(config.TRAIN_SEED)

        log.write('=> Enable torch.backends.cudnn...')
        log.write('')
        torch.backends.cudnn.benchmark = True  # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
        torch.backends.cudnn.deterministic = True  # deterministic result
        torch.backends.cudnn.enabled = True  # enable
        log.write('=> Setting CUDA environment...')

        # recommanded by https://github.com/albu/albumentations
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        log.write('__Python VERSION: {}'.format(sys.version))
        log.write('__pyTorch VERSION: {}'.format(torch.__version__))
        log.write('__CUDA VERSION: {}'.format(torch.version.cuda))
        # call(["nvcc", "-V"])
        log.write('__CUDNN VERSION: {}'.format(torch.backends.cudnn.version()))
        log.write('__Number CUDA Devices: {}'.format(torch.cuda.device_count()))
        log.write('__Devices')
        call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
        log.write('Active CUDA Device: GPU{}'.format(torch.cuda.current_device()))

        log.write('Available devices: {}'.format(torch.cuda.device_count()))
        log.write('Current cuda device: {}'.format(torch.cuda.current_device()))

        try:
            import pycuda
            from pycuda import compiler
            import pycuda.driver as drv

            drv.init()
            log.write("%d device(s) found." % drv.Device.count())

            for ordinal in range(drv.Device.count()):
                dev = drv.Device(ordinal)
                log.write(ordinal, dev.name())
        except ImportError:
            log.write("No pycuda found, skip printing device")
