import socket
import time
import numpy as np

from datetime import datetime

DIRECTORY_SUFFIX_IMG = ".png"
DIRECTORY_PREPROCESSED_SUFFIX_IMG = ".npy"
# DIRECTORY_IMG = DIRECTORY_PREFIX + "data/train/"
# DIRECTORY_SELECTED_IMG = DIRECTORY_PREFIX + "data/iMet_dataset/selected/"

PREDICTION_WRITER = False
PREDICTION_TAG = "test"
PREDICTION_LOAD_TAG = ""
PREDICTION_CHOSEN_THRESHOLD = [0.3]
PREDICTION_TTA = 0










PROJECT_TAG = "test"
PROJECT_TAG = str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-") + "-" + PROJECT_TAG

DEBUG_TRAISE_GPU = False
DEBUG_LAPTOP = True
DEBUG_TEST_CODE = False
DEBUG_AUTO_SHUTDOWN = True
DEBUG_WRITE_SPLIT_CSV = False


MODEL_EPOCHS = 30
MODEL_BATCH_SIZE = 64
MODEL_MOMENTUM = 0.9
MODEL_WEIGHT_DECAY = 0.0005
MODEL_INIT_LEARNING_RATE = 0.0001 # this is useless for lr_schedulers
MODEL_LR_SCHEDULER_REDUCE_FACTOR = 0.5
MODEL_LR_SCHEDULER_PATIENT = 1
MODEL_LR_SCHEDULER_BASELR = 0.001
MODEL_LR_SCHEDULER_MAXLR = 0.01
MODEL_LR_SCHEDULER_STEP = 1363*2
MODEL_LR_SCHEDULER_REDUCE_RESTART = 3
MODEL_LR_SCHEDULER_RESTART_COEF = 0.25
MODEL_FOLD = 5
# MODEL_NO_GRAD = [[],]
MODEL_NO_GRAD = [[-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]]
MODEL_LEARNING_RATE_AFTER_UNFREEZE = 0.001
MODEL_FREEZE_EPOCH = 2

TRAIN_GPU_DICT = {
    "kokecacao-ThinkPad-P50-Ubuntu": "0",
    "ml-k80-3": "0",
    "ml-k80-4": "0",
    "ml-p100-1": "0",
    "presudo-0": "0",
    "presudo-1": "0",
    "presudo-2": "0",
    "presudo-3": "0",
    "KokeCacao-Ubuntu": "0",
}
TRAIN_LOAD_OPTIMIZER = True
TRAIN_GPU_ARG = "0" if socket.gethostname() not in TRAIN_GPU_DICT else TRAIN_GPU_DICT[socket.gethostname()]
if socket.gethostname() not in TRAIN_GPU_DICT: print("Machine {} is not in record, use gpu #0")
TRAIN_GPU_LIST = [int(i) for i in TRAIN_GPU_ARG.split(",")]
TRAIN_DATA_PERCENT = 1
TRAIN_SEED = 19
TRAIN_NUM_WORKER = 8
TRAIN_NUM_GPU = len(TRAIN_GPU_LIST)
TRAIN_NUM_CLASS = 1103

TRAIN_RATIO = 1
EVAL_RATIO = 1 # to 8 when needed
FIND_LR_ON_VALIDATION = False
FIND_LR_RATIO = 10 if FIND_LR_ON_VALIDATION else 100

DIRECTORY_PREFIX = "" # remember to add '/' at the end
DIRECTORY_CSV = DIRECTORY_PREFIX + '../input/imet-2019-fgvc6/train.csv'
DIRECTORY_CSV_ID = 'id'
DIRECTORY_CSV_TARGET = 'attribute_ids'
DIRECTORY_SAMPLE_CSV = DIRECTORY_PREFIX + '../input/imet-2019-fgvc6/sample_submission.csv'
DIRECTORY_TRAIN = DIRECTORY_PREFIX + "data/imet_dataset/train/"
DIRECTORY_TEST = DIRECTORY_PREFIX + '../input/imet-2019-fgvc6/test/'
DIRECTORY_SPLIT = DIRECTORY_PREFIX + 'data/imet_dataset/split.npy'
# DIRECTORY_PRESUDO_CSV = DIRECTORY_PREFIX + 'data/imet_dataset/presudo_labels.csv'
DIRECTORY_CHECKPOINT = DIRECTORY_PREFIX + "model/" + PROJECT_TAG + "/"
DIRECTORY_CP_NAME = 'CP{}_F{}_PT{}_VT{}_LR{}_BS{}_IMG{}.pth'

AUGMENTATION_RESIZE = 224
AUGMENTATION_MEAN = [0.485, 0.456, 0.406]
AUGMENTATION_STD = [0.229, 0.224, 0.225]

DISPLAY_HISTOGRAM = False
DISPLAY_VISUALIZATION = False
DISPLAY_SAVE_ONNX = False

EVAL_IF_THRESHOLD_TEST = True
EVAL_IF_PRED_DISTRIBUTION = True
EVAL_TRY_THRESHOLD = np.linspace(0.0, 1.0, 1000)
EVAL_THRESHOLD = 0.8405

EVAL_SHAKEUP_RATIO = 100

DIRECTORY_LOAD = None
TRAIN = True
TRAIN_RESUME = False
DEBUG_LR_FINDER = False # CAREFUL: open the lr finder would mess up the optimizers, you gonna retrain the network. So the program will end after the graph created.
load_state_dicts = True
load_optimizers = True
load_lr_schedulers = True
train_fold = [-1]
eval_index = 0
start_time = time.time()
lastsave = None
global_steps = []
epoch = 0
fold = 0
versiontag = ""
resetlr = 0