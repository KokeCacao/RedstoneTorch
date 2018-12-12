import socket
import time
import numpy as np

from datetime import datetime

PROJECT_NAME = "HPA"

if PROJECT_NAME == "HPA":
    DEBUG_TRAISE_GPU = False
    DEBUG_LAPTOP = True
    DEBUG_TEST_CODE = True
    DEBUG_AUTO_SHUTDOWN = True

    MODEL_EPOCHS = 100
    MODEL_BATCH_SIZE = 4
    MODEL_INIT_LEARNING_RATE = 0.1
    MODEL_MOMENTUM = 0.9
    MODEL_WEIGHT_DEFAY = 0.0001
    MODEL_FOLD = 10
    MODEL_TRAIN_FOLD = [1]

    TRAIN_GPU_DICT = {
        "kokecacao-ThinkPad-P50-Ubuntu": "0",
        "ml-k80-3": "0",
        "ml-k80-4": "0",
        "ml-p100-1": "0",
    }
    TRAIN_GPU_ARG = TRAIN_GPU_DICT[socket.gethostname()]
    TRAIN_GPU_LIST = [int(i) for i in TRAIN_GPU_ARG.split(",")]
    TRAIN_DATA_PERCENT = 1
    TRAIN_SEED = 19
    TRAIN_NUM_WORKER = 16 # idea from: https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
    TRAIN_NUM_GPU = len(TRAIN_GPU_LIST)
    TRAIN_RESUME = True
    TRAIN_NUMCLASS = 28
    # TRAIN_COSINE = lambda global_step: (0.1 / 2) * (np.cos(np.pi * (np.mod(global_step, 20 * 46808 / 64) / (20 * 46808 / 64))) + 1)  # y=(0.01/2)*(cos(pi*(mod(x-1,10000)/(10000)))+1)
    # TRAIN_TRY_LR = False
    # TRAIN_TRY_LR_FORMULA = lambda x: x / (8 * np.mod(-x - 1, 600) + 0.1) - 0.000207 * x  # y=x/(8*\operatorname{mod}(-x-1,600)+0.1)-0.000207*x

    PROJECT_TAG = "test"
    PROJECT_TAG = str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-") + "-" + PROJECT_TAG

    DISPLAY_HISTOGRAM = False
    DISPLAY_VISUALIZATION = True
    DISPLAY_SAVE_ONNX = False

    EVAL_IF_THRESHOLD_TEST = True
    EVAL_TRY_THRESHOLD = np.linspace(0.0, 1.0, 1000)

    DIRECTORY_PREFIX = ""
    DIRECTORY_SUFFIX_IMG = ".png"
    DIRECTORY_PREPROCESSED_SUFFIX_IMG = ".npy"
    DIRECTORY_IMG = DIRECTORY_PREFIX + "data/train/"
    DIRECTORY_PREPROCESSED_IMG = DIRECTORY_PREFIX + "preprocessed/train/"
    DIRECTORY_LOAD = None
    DIRECTORY_CSV = DIRECTORY_PREFIX + 'data/train.csv'
    DIRECTORY_SAMPLE_CSV = DIRECTORY_PREFIX + 'data/sample_submission.csv'
    DIRECTORY_CHECKPOINT = DIRECTORY_PREFIX + "model/" + PROJECT_TAG + "/"
    DIRECTORY_TEST = DIRECTORY_PREFIX + 'data/test/'
    DIRECTORY_CP_NAME = 'CP{}_F{}_PT{}_VT{}_LR{}_BS{}_IMG{}.pth'

    PREDICTION_WRITER = False
    PREDICTION_TAG = "test"
    PREDICTION_LOAD_TAG = ""
    PREDICTION_SAVE_IMG = True
    PREDICTION_CHOSEN_THRESHOLD = [0.01]

    AUGMENTATION_IMG_ORIGINAL_SIZE = 512
    AUGMENTATION_RESIZE = 224
    # AUGMENTATION_RESIZE = 384
    # AUGMENTATION_RESIZE = 512

    eval_index = 0
    start_time = time.time()
    lastsave = None
    global_steps = []
    epoch = 0
    fold = 0
    versiontag = ""

elif PROJECT_NAME == "QUBO":
    DEBUG_TRAISE_GPU = False
    DEBUG_LAPTOP = True
    DEBUG_TEST_CODE = True
    DEBUG_AUTO_SHUTDOWN = True

    MODEL_EPOCHS = 100
    MODEL_BATCH_SIZE = 1
    MODEL_INIT_LEARNING_RATE = 0.1
    MODEL_MOMENTUM = 0.9
    MODEL_WEIGHT_DEFAY = 0.0001
    MODEL_FOLD = 10
    MODEL_TRAIN_FOLD = [0,1,2,3]

    TRAIN_GPU_DICT = {
        "ml-k80-3": "0",
        "ml-k80-4": "0",
        "ml-p100-1": "0",
    }
    TRAIN_GPU_ARG = TRAIN_GPU_DICT[socket.gethostname()]
    TRAIN_GPU_LIST = [int(i) for i in TRAIN_GPU_ARG.split(",")]
    TRAIN_VAL_PERCENT = 0.05
    TRAIN_DATA_PERCENT = 0.1
    TRAIN_SEED = 19
    TRAIN_SAVE_CHECKPOINT = True
    TRAIN_NUM_WORKER = 16 # idea from: https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
    TRAIN_NUM_GPU = len(TRAIN_GPU_LIST)
    TRAIN_RESUME = True
    TRAIN_NUMCLASS = 28
    TRAIN_COSINE = lambda global_step: (0.1 / 2) * (np.cos(np.pi * (np.mod(global_step, 20 * 46808 / 64) / (20 * 46808 / 64))) + 1)  # y=(0.01/2)*(cos(pi*(mod(x-1,10000)/(10000)))+1)
    TRAIN_TRY_LR = False
    TRAIN_TRY_LR_FORMULA = lambda x: x / (8 * np.mod(-x - 1, 600) + 0.1) - 0.000207 * x  # y=x/(8*\operatorname{mod}(-x-1,600)+0.1)-0.000207*x

    PROJECT_TAG = "test"
    PROJECT_TAG = str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-") + "-" + PROJECT_TAG

    DISPLAY_HISTOGRAM = False
    DISPLAY_VISUALIZATION = True
    DISPLAY_SAVE_ONNX = False

    EVAL_IF_THRESHOLD_TEST = True
    EVAL_TRY_THRESHOLD = np.linspace(0.0, 1.0, 1000)

    DIRECTORY_PREFIX = ""
    DIRECTORY_SUFFIX_IMG = ".png"
    DIRECTORY_SUFFIX_MASK = ".png"
    DIRECTORY_PREPROCESSED_SUFFIX_IMG = ".npy"
    DIRECTORY_IMG = DIRECTORY_PREFIX + "data/qubo/train/"
    DIRECTORY_MASK = DIRECTORY_PREFIX + 'data/qubo/train/masks/'  # augmentation
    DIRECTORY_PREPROCESSED_IMG = DIRECTORY_PREFIX + "preprocessed/qubo/train/"
    DIRECTORY_LOAD = None
    DIRECTORY_CSV = DIRECTORY_PREFIX + 'data/qubo/train.csv'
    DIRECTORY_SAMPLE_CSV = DIRECTORY_PREFIX + 'data/sample_submission.csv'
    DIRECTORY_CHECKPOINT = DIRECTORY_PREFIX + "model/" + PROJECT_TAG + "/"
    DIRECTORY_TEST = DIRECTORY_PREFIX + 'data/qubo/test/'
    DIRECTORY_CP_NAME = 'CP{}.pth'

    PREDICTION_WRITER = False
    PREDICTION_TAG = "test"
    PREDICTION_LOAD_TAG = ""
    PREDICTION_SAVE_IMG = True
    PREDICTION_DARK_THRESHOLD = 0.05
    PREDICTION_CHOSEN_THRESHOLD = [0.2187]

    AUGMENTATION_IMG_ORIGINAL_SIZE = 512
    AUGMENTATION_RESIZE = 224
    # AUGMENTATION_RESIZE = 384
    # AUGMENTATION_RESIZE = 512

    start_time = time.time()
    lastsave = None
    global_steps = []
    epoch = 0
    fold = 0
    versiontag = ""