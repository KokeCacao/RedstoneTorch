import multiprocessing
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


MODEL_EPOCHS = 64
# MODEL_BATCH_SIZE = 256
MODEL_BATCH_SIZE = 128
MODEL_MOMENTUM = 0.9
MODEL_WEIGHT_DECAY = 0.0001 # weight decay only used in AdamW and SGDW, other implementation may be wrong
MODEL_INIT_LEARNING_RATE = 0.0001 # this is useless for lr_schedulers
MODEL_LR_SCHEDULER_REDUCE_FACTOR = 0.5
MODEL_LR_SCHEDULER_PATIENT = 0
MODEL_LR_SCHEDULER_BASELR = 0.0001
# MODEL_LR_SCHEDULER_THRESHOLD = -2e-3
MODEL_LR_SCHEDULER_THRESHOLD = 0
MODEL_LR_SCHEDULER_MAXLR = 0.01
MODEL_LR_SCHEDULER_STEP = 1363*2 #767 when bs =128
MODEL_LR_SCHEDULER_REDUCE_RESTART = 6
MODEL_LR_SCHEDULER_RESTART_COEF = 1./8
MODEL_FOLD = 10
# MODEL_NO_GRAD = [[],]
MODEL_NO_GRAD = [[-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]]
MODEL_LEARNING_RATE_AFTER_UNFREEZE = 0.001
MODEL_FREEZE_EPOCH = 7

TRAIN_GPU_DICT = {
    "kokecacao-ThinkPad-P50-Ubuntu": "0",
    "ml-k80-3": "0,1,2,3", # 16280 MB
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
TRAIN_NUM_WORKER = multiprocessing.cpu_count()
TRAIN_NUM_GPU = len(TRAIN_GPU_LIST)
TRAIN_NUM_CLASS = 1103
TRAIN_GRADIENT_ACCUMULATION = 6
TRAIN_ACCUMULATION_AFTER_EPOCH = 99

TRAIN_RATIO = 1
EVAL_RATIO = 2 # to 8 when need
FIND_LR_ON_VALIDATION = False
FIND_LR_RATIO = 10 if FIND_LR_ON_VALIDATION else 100

DIRECTORY_PREFIX = "" # remember to add '/' at the end
DIRECTORY_CSV = DIRECTORY_PREFIX + 'data/imet_dataset/train.csv'
DIRECTORY_CSV_ID = 'id'
DIRECTORY_CSV_TARGET = 'attribute_ids'
DIRECTORY_SAMPLE_CSV = DIRECTORY_PREFIX + 'data/imet_dataset/sample_submission.csv'
DIRECTORY_TRAIN = DIRECTORY_PREFIX + "data/imet_dataset/train/"
DIRECTORY_TEST = DIRECTORY_PREFIX + 'data/imet_dataset/test/'
DIRECTORY_SPLIT = DIRECTORY_PREFIX + 'data/imet_dataset/split{}.npy'.format(MODEL_FOLD)
# DIRECTORY_PRESUDO_CSV = DIRECTORY_PREFIX + 'data/imet_dataset/presudo_labels.csv'
DIRECTORY_CHECKPOINT = DIRECTORY_PREFIX + "model/" + PROJECT_TAG + "/"
DIRECTORY_CP_NAME = 'CP{}_F{}_PT{}_VT{}_LR{}_BS{}_IMG{}.pth'

AUGMENTATION_RESIZE = 250
AUGMENTATION_RESIZE_CHANGE_EPOCH = 12
AUGMENTATION_RESIZE_CHANGE = 250 #1024
AUGMENTATION_MEAN = [0.485, 0.456, 0.406]
AUGMENTATION_STD = [0.229, 0.224, 0.225]

DISPLAY_HISTOGRAM = False
DISPLAY_VISUALIZATION = False
DISPLAY_SAVE_ONNX = False

EVAL_IF_THRESHOLD_TEST = True
EVAL_IF_PRED_DISTRIBUTION = True
EVAL_TRY_THRESHOLD = np.linspace(0.0, 1.0, 1000)
EVAL_THRESHOLD = 0.3

EVAL_SHAKEUP_RATIO = 500

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
