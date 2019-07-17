import os
import sys
from optparse import OptionParser

from tensorboardX import SummaryWriter

import config
# import project.HisCancer_project.HisCancer_prediction
import project.siim_project.siim_prediction
import project.imet_project.imet_prediction
from gpu import gpu_profile
from reproduceability import reproduceability
from utils.memory import memory_thread


def get_args():
    parser = OptionParser()
    parser.add_option('--loaddir', dest='loaddir', default=False, help='tag you want to load')
    parser.add_option('--versiontag', dest='versiontag', default="predict", help='tag for tensorboard-log and prediction')
    parser.add_option('--loadfile', dest='loadfile', default=False, help='file you want to load')
    parser.add_option('--fold', type="float", dest='fold', default=-1., help='set training fold')
    parser.add_option('--net', type="string", dest='net', default=None, help='Network You Want to Use')
    parser.add_option('--tta', type="float", dest='tta', default=-1., help='tta')
    parser.add_option('--threshold', type="float", dest='threshold', default=-1., help='threshold')
    parser.add_option('--batch_size', type="float", dest='batch_size', default=0, help='batch size')
    parser.add_option('--image_size', type="float", dest='image_size', default=0, help='image resize')

    (options, args) = parser.parse_args()
    return options


def load_args():
    args = get_args()
    if args.versiontag: config.PREDICTION_TAG = args.versiontag
    if args.batch_size != 0: config.MODEL_BATCH_SIZE = int(args.batch_size)
    if args.loadfile:
        config.DIRECTORY_LOAD = config.DIRECTORY_PREFIX + "model/" + args.loaddir + "/" + args.loadfile
        config.DIRECTORY_CHECKPOINT = config.DIRECTORY_PREFIX + "model/" + args.loaddir + "/predict/"
    else: raise ValueError("You must set --loadfile directory in prediction mode")

    if args.threshold != -1:
        config.PREDICTION_CHOSEN_THRESHOLD = [float(args.threshold)]
    else:
        raise NotImplementedError("Please specify threshold")

    if args.fold != -1 and args.fold < config.MODEL_FOLD:
        config.train_fold = [int(args.fold)]
        print("=> Set training fold to: {}".format(config.train_fold))
    else:
        raise NotImplementedError("Please specify fold number")

    if args.net == None: raise NotImplementedError("Please specify net")
    else: config.net = args.net

    if args.image_size != 0:
        config.AUGMENTATION_RESIZE = int(args.image_size)
        config.AUGMENTATION_RESIZE_CHANGE = int(args.image_size)
    else:
        raise NotImplementedError("Please specify image size")

    if args.tta >= 0:
        config.prediction_tta = int(args.tta)
        print("=> Set tta to: {}".format(config.prediction_tta))
    else:
        raise NotImplementedError("Please specify tta number")

if __name__ == '__main__':
    """
    PLAYGROUND
    """
    os.system("sudo shutdown -c")
    config.DEBUG_TEST_CODE = False
    config.DEBUG_LAPTOP = False
    if not config.DEBUG_TEST_CODE:
        if config.DEBUG_TRAISE_GPU: sys.settrace(gpu_profile)
        load_args()

        if config.PREDICTION_WRITER:
            writer = SummaryWriter(config.DIRECTORY_CHECKPOINT)
            print("=> Tensorboard: " + "tensorboard --logdir=" + config.DIRECTORY_CHECKPOINT + " --port=6006")
            memory = memory_thread(1, writer)
            memory.setDaemon(True)
            memory.start()
        else: writer = None

        reproduceability()

        try:
            # prediction = project.HisCancer_project.HisCancer_prediction.HisCancerPrediction(writer)
            # prediction = project.HisCancer_project.HisCancer_prediction.HisCancerPrediction(writer)
            prediction = project.siim_project.siim_prediction.SIIMPrediction(writer)
        except Exception as e:
            if not isinstance(e, KeyboardInterrupt):
                os.system("sudo shutdown -P +20")
                print("""
                    WARNING: THE SYSTEM WILL SHUTDOWN
                    Use command: sudo shutdown -c
                """)
            raise
        os.system("sudo shutdown -P +20")
        print("""
                            WARNING: THE SYSTEM WILL SHUTDOWN
                            Use command: sudo shutdown -c
                        """)
