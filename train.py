import os
import sys
from datetime import datetime
from optparse import OptionParser

from tensorboardX import SummaryWriter

import config
# dir_prefix = 'drive/My Drive/ML/Pytorch-UNet/'
from gpu import gpu_profile
from project.hpa_project import hpa_train


# def log_data(file_name, data):
#     with open(file_name + ".txt", "a+") as file:
#         file.write(data + "\n")
from project.qubo_project import qubo_train
from reproduceability import reproduceability


def get_args():
    parser = OptionParser()
    parser.add_option('--projecttag', dest='projecttag', default=False, help='tag you want to load')
    parser.add_option('--versiontag', dest='versiontag', default="", help='tag for tensorboard-log')
    parser.add_option('--loadfile', dest='loadfile', default=False, help='file you want to load')
    parser.add_option('--loaddir', dest='loaddir', default=False, help='file you want to load')
    parser.add_option('--resume', dest='resume', default=False, help='resume or create a new folder')

    (options, args) = parser.parse_args()
    return options


def load_args():
    args = get_args()
    if args.versiontag: config.versiontag = args.versiontag
    if args.projecttag: config.PROJECT_TAG = args.projecttag
    config.TRAIN_RESUME = True if args.resume == "True" else False

    if args.loadfile:
        config.lastsave = args.loadfile
        if config.TRAIN_RESUME:
            config.DIRECTORY_LOAD = config.DIRECTORY_PREFIX + "model/" + args.loaddir + "/" + args.loadfile
            config.DIRECTORY_CHECKPOINT = config.DIRECTORY_PREFIX + "model/" + args.loaddir + "/"
        else:
            config.PROJECT_TAG = str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-") + "-" + config.PROJECT_TAG
            config.DIRECTORY_LOAD = config.DIRECTORY_PREFIX + "model/" + args.loaddir + "/" + args.loadfile
            config.DIRECTORY_CHECKPOINT = config.DIRECTORY_PREFIX + "model/" + config.PROJECT_TAG + "/"
    else:
        config.PROJECT_TAG = str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-") + "-" + config.PROJECT_TAG
        config.DIRECTORY_LOAD = None
        config.DIRECTORY_CHECKPOINT = config.DIRECTORY_PREFIX + "model/" + config.PROJECT_TAG + "/"

if __name__ == '__main__':
    """
    PLAYGROUND
    """
    # tensor = torch.Tensor([
    #     [0.1, 0.9, 0.9, 0.1],
    #     [0.1, 0.9, 0.1, 0.1],
    # ])
    # label = torch.Tensor([
    #     [0, 1, 1, 1],
    #     [0, 1, 1, 1],
    # ])
    # loss = Focal_Loss_from_git(alpha=0.25, gamma=2, eps=1e-7)(label, tensor)
    # print(loss)

    config.DEBUG_TEST_CODE = False
    config.DEBUG_LAPTOP = False
    if not config.DEBUG_TEST_CODE:
        if config.DEBUG_TRAISE_GPU: sys.settrace(gpu_profile)
        load_args()

        writer = SummaryWriter(config.DIRECTORY_CHECKPOINT)
        print("=> Tensorboard: " + "python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/" + config.DIRECTORY_CHECKPOINT + " --port=6006")

        reproduceability()

        # memory = memory_thread(1, writer)
        # memory.setDaemon(True)
        # memory.start()

        print("=> Current Directory: " + str(os.getcwd()))
        print("=> Loading neuronetwork...")
        try:
            # project = hpa_train.HPATrain(writer)
            project = qubo_train.QUBOTrain(writer)
        except Exception as e:
            # with open('Exception.txt', 'a+') as f:
            #     f.write(str(e))
            if not isinstance(e, KeyboardInterrupt):
                os.system("sudo shutdown -P +20")
                print("""
                    WARNING: THE SYSTEM WILL SHUTDOWN
                    Use command: sudo shutdown -c
                """)
            raise
