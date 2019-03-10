import sys
from optparse import OptionParser

from tensorboardX import SummaryWriter

import config
import project.HisCancer_project.HisCancer_postprocess
from gpu import gpu_profile
from reproduceability import reproduceability
from utils.memory import memory_thread


def get_args():
    parser = OptionParser()
    parser.add_option('--loaddir', dest='loaddir', default=False, help='tag you want to load')
    parser.add_option('--versiontag', dest='versiontag', default="predict", help='tag for tensorboard-log and prediction')
    parser.add_option('--loadfile', dest='loadfile', default=False, help='file you want to load')

    (options, args) = parser.parse_args()
    return options


def load_args():
    args = get_args()
    if args.versiontag: config.PREDICTION_TAG = args.versiontag
    if args.loadfile:
        config.DIRECTORY_LOAD = config.DIRECTORY_PREFIX + "model/" + args.loaddir + "/" + args.loadfile
        config.DIRECTORY_CHECKPOINT = config.DIRECTORY_PREFIX + "model/" + args.loaddir + "/predict/"
    else: raise ValueError("You must set --loadfile directory in prediction mode")

if __name__ == '__main__':
    """
    PLAYGROUND
    """
    config.DEBUG_TEST_CODE = False
    config.DEBUG_LAPTOP = False
    if not config.DEBUG_TEST_CODE:
        if config.DEBUG_TRAISE_GPU: sys.settrace(gpu_profile)
        load_args()

        if config.PREDICTION_WRITER:
            writer = SummaryWriter(config.DIRECTORY_CHECKPOINT)
            print("=> Tensorboard: " + "python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/" + config.DIRECTORY_CHECKPOINT + " --port=6006")
            memory = memory_thread(1, writer)
            memory.setDaemon(True)
            memory.start()
        else: writer = None

        reproduceability()

        postprocess = project.HisCancer_project.HisCancer_postprocess.HisCancerPostprocess(writer)