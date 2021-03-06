import socket
import time
import numpy as np

from datetime import datetime

DEBUG_TRAISE_GPU = False
DEBUG_LAPTOP = True
DEBUG_TEST_CODE = False
DEBUG_AUTO_SHUTDOWN = True

DEBUG_LR_FINDER = False

MODEL_EPOCHS = 100
MODEL_BATCH_SIZE = 32
MODEL_INIT_LEARNING_RATE = 0.1
MODEL_MOMENTUM = 0.9
MODEL_WEIGHT_DECAY = 0.0001
MODEL_FOLD = 10
train_fold = [1]

TRAIN_GPU_DICT = {
    "kokecacao-ThinkPad-P50-Ubuntu": "0",
    "ml-k80-3": "0",
    "ml-k80-4": "0",
    "ml-p100-1": "0",
    "KokeCacao-Ubuntu": "0",
}
TRAIN_GPU_ARG = TRAIN_GPU_DICT[socket.gethostname()]
TRAIN_GPU_LIST = [int(i) for i in TRAIN_GPU_ARG.split(",")]
TRAIN_DATA_PERCENT = 1.0
TRAIN_SEED = 19
TRAIN_NUM_WORKER = 6
TRAIN_NUM_GPU = len(TRAIN_GPU_LIST)
TRAIN_RESUME = True
TRAIN_NUM_CLASS = 5
# TRAIN_COSINE = lambda global_step: (0.1 / 2) * (np.cos(np.pi * (np.mod(global_step, 20 * 46808 / 64) / (20 * 46808 / 64))) + 1)  # y=(0.01/2)*(cos(pi*(mod(x-1,10000)/(10000)))+1)
# TRAIN_TRY_LR = False
# TRAIN_TRY_LR_FORMULA = lambda x: x / (8 * np.mod(-x - 1, 600) + 0.1) - 0.000207 * x  # y=x/(8*\operatorname{mod}(-x-1,600)+0.1)-0.000207*x
TRAIN_RATIO = 32
EVAL_RATIO = 32
FIND_LR_RATIO = 64

PROJECT_TAG = "test"
PROJECT_TAG = str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-") + "-" + PROJECT_TAG

DISPLAY_HISTOGRAM = False
DISPLAY_VISUALIZATION = True
DISPLAY_SAVE_ONNX = False

EVAL_IF_THRESHOLD_TEST = True
EVAL_TRY_THRESHOLD = np.linspace(0.0, 1.0, 1000)
EVAL_THRESHOLD = 0.2

DIRECTORY_PREFIX = "" # remember to add '/' at the end
# DIRECTORY_PREFIX = "/home/koke_cacao/Documents/Koke_Cacao/Python/WorkSpace/RedstoneTorch/" # remember to add '/' at the end
# ~/RedstoneTorch/data/qubo_dataset/preprocessed$ mv /home/k1412042720/qubo_dataset.zip ~/RedstoneTorch/data/qubo_dataset/preprocessed/
DIRECTORY_SUFFIX_IMG = ".png"
DIRECTORY_PREPROCESSED_SUFFIX_IMG = ".npy"
# DIRECTORY_IMG = DIRECTORY_PREFIX + "data/train/"
DIRECTORY_PREPROCESSED_IMG = DIRECTORY_PREFIX + "data/qubo_dataset/preprocessed/"
DIRECTORY_SELECTED_IMG = DIRECTORY_PREFIX + "data/qubo_dataset/selected/"
# DIRECTORY_LOAD = None
DIRECTORY_CSV = DIRECTORY_PREFIX + 'data/qubo_dataset/preprocessed/train.csv'
DIRECTORY_CHECKPOINT = DIRECTORY_PREFIX + "model/" + PROJECT_TAG + "/"
DIRECTORY_CP_NAME = 'CP{}_F{}_PT{}_VT{}_LR{}_BS{}_IMG{}.pth'

PREDICTION_WRITER = False
PREDICTION_TAG = "test"
PREDICTION_LOAD_TAG = ""
PREDICTION_CHOSEN_THRESHOLD = [0.01]

AUGMENTATION_IMG_ORIGINAL_SIZE = (640, 480)
AUGMENTATION_RESIZE = 224

eval_index = 0
start_time = time.time()
lastsave = None
global_steps = []
epoch = 0
fold = 0
versiontag = ""
resetlr = 0
