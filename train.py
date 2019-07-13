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
# from project.HisCancer_project import HisCancer_train
# from project.imet_project import imet_train
from project.siim_project import siim_train
from reproduceability import reproduceability


def get_args():
    parser = OptionParser()
    parser.add_option('--projecttag', type="string", dest='projecttag', default=False, help='tag you want to load')
    parser.add_option('--versiontag', type="string", dest='versiontag', default="", help='tag for tensorboard-log')
    parser.add_option('--loadfile', type="string", dest='loadfile', default=False, help='file you want to load')
    parser.add_option('--loaddir', type="string", dest='loaddir', default=False, help='file you want to load')
    parser.add_option('--resume', type="string", dest='resume', default=False, help='resume or create a new folder')
    parser.add_option('--resetlr', type="float", dest='resetlr', default=0., help='reset the learning rate')
    parser.add_option('--fold', type="float", dest='fold', default=-1., help='set training fold')
    parser.add_option('--testlr', type="string", dest='testlr', default=False, help='test lr')
    parser.add_option('--state_dict', type="string", dest='state_dict', default=True, help='whether to load state_dicts')
    parser.add_option('--optimizer', type="string", dest='optimizer', default=True, help='whether to load optimizers')
    parser.add_option('--lr_scheduler', type="string", dest='lr_scheduler', default=True, help='whether to load lr_schedulers')
    parser.add_option('--train', type="string", dest='train', default=True, help='whether to train')
    parser.add_option('--image_size', type="float", dest='image_size', default=0, help='image resize')
    parser.add_option('--total_epoch', type="float", dest='total_epoch', default=0, help='additional epoch for training')
    parser.add_option('--batch_size', type="float", dest='batch_size', default=0, help='batch size')
    parser.add_option('--accumulation', type="float", dest='accumulation', default=0, help='gradient accumulation')

    (options, args) = parser.parse_args()
    return options


def load_args():
    args = get_args()
    if args.versiontag: config.versiontag = args.versiontag
    if args.projecttag: config.PROJECT_TAG = args.projecttag
    config.train_resume = True if args.resume == "True" else False
    config.debug_lr_finder = True if args.testlr == "True" else False
    config.load_state_dicts = False if args.state_dict == "False" else True
    config.load_optimizers = False if args.optimizer == "False" else True
    config.load_lr_schedulers = False if args.lr_scheduler == "False" else True
    config.train = False if args.train == "False" else True
    config.resetlr = args.resetlr

    if args.image_size != 0: config.AUGMENTATION_RESIZE = int(args.image_size)
    if args.total_epoch != 0: config.MODEL_EPOCHS = int(args.total_epoch)
    if args.batch_size != 0: config.MODEL_BATCH_SIZE = int(args.batch_size)
    if args.accumulation != 0: config.TRAIN_GRADIENT_ACCUMULATION = int(args.accumulation)

    if args.loadfile:
        config.lastsave = args.loadfile
        if config.train_resume:
            config.DIRECTORY_LOAD = config.DIRECTORY_PREFIX + "model/" + args.loaddir + "/" + args.loadfile
            config.DIRECTORY_CHECKPOINT = config.DIRECTORY_PREFIX + "model/" + args.loaddir + "/"
        else:
            config.PROJECT_TAG = str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-") + "-" + config.PROJECT_TAG
            config.DIRECTORY_LOAD = config.DIRECTORY_PREFIX + "model/" + args.loaddir + "/" + args.loadfile
            config.DIRECTORY_CHECKPOINT = config.DIRECTORY_PREFIX + "model/" + config.PROJECT_TAG + "/"
    else:
        config.PROJECT_TAG = str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-") + "-" + config.PROJECT_TAG
        config.DIRECTORY_LOAD = False
        config.DIRECTORY_CHECKPOINT = config.DIRECTORY_PREFIX + "model/" + config.PROJECT_TAG + "/"

    if args.fold != -1 and args.fold < config.MODEL_FOLD:
        config.train_fold = [int(args.fold)]
        print("=> Set training fold to: {}".format(config.train_fold))
    else:
        raise NotImplementedError("Please specify fold number")

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

    os.system("sudo shutdown -c")
    config.DEBUG_TEST_CODE = False
    config.DEBUG_LAPTOP = False
    if not config.DEBUG_TEST_CODE:
        if config.DEBUG_TRAISE_GPU: sys.settrace(gpu_profile)
        load_args()

        writer = SummaryWriter(config.DIRECTORY_CHECKPOINT)
        print("=> Tensorboard: " + "tensorboard --logdir=" + config.DIRECTORY_CHECKPOINT + " --port=6006")

        reproduceability()

        # memory = memory_thread(1, writer)
        # memory.setDaemon(True)
        # memory.start()

        print("=> Current Directory: " + str(os.getcwd()))
        print("=> Loading neuronetwork...")
        try:
            # project = hpa_train.HPATrain(writer)
            # project = HisCancer_train.HisCancerTrain(writer)
            # project = imet_train.IMetTrain(writer)
            project = siim_train.SIIMTrain(writer)
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
        os.system("sudo shutdown -P +20")
        print("""
                            WARNING: THE SYSTEM WILL SHUTDOWN
                            Use command: sudo shutdown -c
                        """)
