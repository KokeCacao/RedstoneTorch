from optparse import OptionParser
from submission import config
from reproduceability import reproduceability
from submission.imet_prediction import IMetPrediction


def get_args():
    parser = OptionParser()
    parser.add_option('--loaddir', dest='loaddir', default=False, help='tag you want to load')
    parser.add_option('--versiontag', dest='versiontag', default="predict", help='tag for tensorboard-log and prediction')
    parser.add_option('--loadfile', dest='loadfile', default=False, help='file you want to load')
    parser.add_option('--fold', type="float", dest='fold', default=-1., help='set training fold')
    parser.add_option('--tta', type="float", dest='tta', default=-1., help='tta')

    (options, args) = parser.parse_args()
    return options


def load_args():
    args = get_args()
    if args.versiontag: config.PREDICTION_TAG = args.versiontag
    if args.loadfile:
        config.DIRECTORY_LOAD = config.DIRECTORY_PREFIX + args.loaddir + "/" + args.loadfile
        config.DIRECTORY_CHECKPOINT = config.DIRECTORY_PREFIX + "model/" + args.loaddir + "/predict/"
    else: raise ValueError("You must set --loadfile directory in prediction mode")

    if args.fold != -1 and args.fold < config.MODEL_FOLD:
        config.train_fold = [int(args.fold)]
        print("=> Set training fold to: {}".format(config.train_fold))
    else:
        raise NotImplementedError("Please specify fold number")

    if args.tta >= 0:
        config.PREDICTION_TTA = int(args.tta)
        print("=> Set tta to: {}".format(config.PREDICTION_TTA))
    else:
        raise NotImplementedError("Please specify tta number")

if __name__ == '__main__':
    """
    PLAYGROUND
    """
    config.DEBUG_TEST_CODE = False
    config.DEBUG_LAPTOP = False
    load_args()
    reproduceability()
    prediction = IMetPrediction()
