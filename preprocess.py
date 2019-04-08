import os
import sys
from datetime import datetime
from optparse import OptionParser

import config
# import project.HisCancer_project.HisCancer_preprocess
import project.imet_project.imet_preprocess
from gpu import gpu_profile
from reproduceability import reproduceability


def get_args():
    parser = OptionParser()
    parser.add_option('--projecttag', dest='projecttag', default=False, help='tag you want to load')
    parser.add_option('--versiontag', dest='versiontag', default="", help='tag for tensorboard-log')
    parser.add_option('--loadfile', action="store_true", dest='loadfile', default=False, help='file you want to load')
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
        config.DIRECTORY_LOAD = False
        config.DIRECTORY_CHECKPOINT = config.DIRECTORY_PREFIX + "model/" + config.PROJECT_TAG + "/"

if __name__ == '__main__':
    """
    PLAYGROUND
    """

    os.system("sudo shutdown -c")
    config.DEBUG_TEST_CODE = False
    config.DEBUG_LAPTOP = False
    if not config.DEBUG_TEST_CODE:
        if config.DEBUG_TRAISE_GPU: sys.settrace(gpu_profile)
        # load_args()
        reproduceability()

        print("=> Current Directory: " + str(os.getcwd()))
        # print("=> Loading neuronetwork...")

        # preprocess = project.qubo_project.qubo_preprocess.QUBOPreprocess()

        try:
            # preprocess = project.HisCancer_project.HisCancer_preprocess.HisCancerPreprocess(from_dir=config.DIRECTORY_TEST, to_dir=config.DIRECTORY_TEST)
            preprocess = project.imet_project.imet_preprocess.IMetPreprocess(from_dir=config.DIRECTORY_TEST, to_dir=config.DIRECTORY_TEST)
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
