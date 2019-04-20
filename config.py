import socket
import time
import numpy as np

from datetime import datetime

PROJECT_NAME = "IMet"

if PROJECT_NAME == "HPA":
    DEBUG_TRAISE_GPU = False
    DEBUG_LAPTOP = True
    DEBUG_TEST_CODE = True
    DEBUG_AUTO_SHUTDOWN = True

    MODEL_EPOCHS = 100
    MODEL_BATCH_SIZE = 1
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
    }
    TRAIN_GPU_ARG = TRAIN_GPU_DICT[socket.gethostname()]
    TRAIN_GPU_LIST = [int(i) for i in TRAIN_GPU_ARG.split(",")]
    TRAIN_DATA_PERCENT = 1
    TRAIN_SEED = 19
    TRAIN_NUM_WORKER = 16 # idea from: https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
    TRAIN_NUM_GPU = len(TRAIN_GPU_LIST)
    TRAIN_RESUME = True
    TRAIN_NUM_CLASS = 28
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
    DIRECTORY_LOAD = False
    DIRECTORY_CSV = DIRECTORY_PREFIX + 'data/train.csv'
    DIRECTORY_SAMPLE_CSV = DIRECTORY_PREFIX + 'data/sample_submission.csv'
    DIRECTORY_CHECKPOINT = DIRECTORY_PREFIX + "model/" + PROJECT_TAG + "/"
    DIRECTORY_TEST = DIRECTORY_PREFIX + 'data/test/'
    DIRECTORY_CP_NAME = 'CP{}_F{}_PT{}_VT{}_LR{}_BS{}_IMG{}.pth'

    PREDICTION_WRITER = False
    PREDICTION_TAG = "test"
    PREDICTION_LOAD_TAG = ""
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
elif PROJECT_NAME == "HisCancer":
    DEBUG_TRAISE_GPU = False
    DEBUG_LAPTOP = True
    DEBUG_TEST_CODE = False
    DEBUG_AUTO_SHUTDOWN = True

    DEBUG_LR_FINDER = False # CAREFUL: open the lr finder would mess up the optimizers, you gonna retrain the network. So the program will end after the graph created.

    MODEL_EPOCHS = 8
    MODEL_BATCH_SIZE = 64
    MODEL_INIT_LEARNING_RATE = 0.005
    MODEL_MIN_LEARNING_RATE = 0.00007
    MODEL_COS_LEARNING_RATE_PERIOD = 1000
    MODEL_MOMENTUM = 0.9
    MODEL_WEIGHT_DECAY = 0.0000
    MODEL_FOLD = 5
    MODEL_NO_GRAD = [[-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]]
    MODEL_LEARNING_RATE_AFTER_UNFREEZE = 0.001
    MODEL_FREEZE_EPOCH = 4

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
    TRAIN_GPU_ARG = TRAIN_GPU_DICT[socket.gethostname()]
    TRAIN_GPU_LIST = [int(i) for i in TRAIN_GPU_ARG.split(",")]
    TRAIN_DATA_PERCENT = 1
    TRAIN_SEED = 19
    TRAIN_NUM_WORKER = 8
    TRAIN_NUM_GPU = len(TRAIN_GPU_LIST)
    TRAIN_RESUME = True
    TRAIN_NUM_CLASS = 2
    TRAIN_LOAD_OPTIMIZER = True
    # TRAIN_COSINE = lambda global_step: (0.1 / 2) * (np.cos(np.pi * (np.mod(global_step, 20 * 46808 / 64) / (20 * 46808 / 64))) + 1)  # y=(0.01/2)*(cos(pi*(mod(x-1,10000)/(10000)))+1)
    # TRAIN_TRY_LR = False
    # TRAIN_TRY_LR_FORMULA = lambda x: x / (8 * np.mod(-x - 1, 600) + 0.1) - 0.000207 * x  # y=x/(8*\operatorname{mod}(-x-1,600)+0.1)-0.000207*x
    TRAIN_RATIO = 1
    EVAL_RATIO = 1 # to 8 when needed
    FIND_LR_ON_VALIDATION = False
    FIND_LR_RATIO = 20 if FIND_LR_ON_VALIDATION else 100

    PROJECT_TAG = "test"
    PROJECT_TAG = str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-") + "-" + PROJECT_TAG

    DISPLAY_HISTOGRAM = False
    DISPLAY_VISUALIZATION = False
    DISPLAY_SAVE_ONNX = False

    EVAL_IF_THRESHOLD_TEST = False
    EVAL_TRY_THRESHOLD = np.linspace(0.0, 1.0, 1000)
    EVAL_THRESHOLD = 0.5

    DIRECTORY_PREFIX = "" # remember to add '/' at the end
    # DIRECTORY_PREFIX = "/home/koke_cacao/Documents/Koke_Cacao/Python/WorkSpace/RedstoneTorch/" # remember to add '/' at the end
    # ~/RedstoneTorch/data/qubo_dataset/preprocessed$ mv /home/k1412042720/qubo_dataset.zip ~/RedstoneTorch/data/qubo_dataset/preprocessed/
    DIRECTORY_SUFFIX_IMG = ".png"
    DIRECTORY_PREPROCESSED_SUFFIX_IMG = ".npy"
    # DIRECTORY_IMG = DIRECTORY_PREFIX + "data/train/"
    DIRECTORY_PREPROCESSED_IMG = DIRECTORY_PREFIX + "data/HisCancer_dataset/preprocessed/"
    DIRECTORY_SELECTED_IMG = DIRECTORY_PREFIX + "data/HisCancer_dataset/selected/"
    DIRECTORY_TEST = DIRECTORY_PREFIX + 'data/HisCancer_dataset/test/'
    # DIRECTORY_LOAD = None
    DIRECTORY_CSV = DIRECTORY_PREFIX + 'data/HisCancer_dataset/train.csv'
    DIRECTORY_SAMPLE_CSV = DIRECTORY_PREFIX + 'data/HisCancer_dataset/sample_submission.csv'
    DIRECTORY_PRESUDO_CSV = DIRECTORY_PREFIX + 'data/HisCancer_dataset/presudo_labels.csv'
    DIRECTORY_CHECKPOINT = DIRECTORY_PREFIX + "model/" + PROJECT_TAG + "/"
    DIRECTORY_CP_NAME = 'CP{}_F{}_PT{}_VT{}_LR{}_BS{}_IMG{}.pth'

    PREDICTION_WRITER = False
    PREDICTION_TAG = "test"
    PREDICTION_LOAD_TAG = ""
    PREDICTION_CHOSEN_THRESHOLD = [0.5]
    PREDICTION_TTA = 0

    AUGMENTATION_IMG_ORIGINAL_SIZE = (96, 96)
    # AUGMENTATION_RESIZE = 224
    AUGMENTATION_RESIZE = 128
    AUGMENTATION_MEAN = [0.485, 0.456, 0.406]
    AUGMENTATION_STD = [0.229, 0.224, 0.225]

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
elif PROJECT_NAME == "IMet":
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


    MODEL_EPOCHS = 32
    MODEL_BATCH_SIZE = 64
    MODEL_MOMENTUM = 0.9
    MODEL_WEIGHT_DECAY = 0
    MODEL_INIT_LEARNING_RATE = 0.0001 # this is useless for lr_schedulers
    MODEL_LR_SCHEDULER_REDUCE_FACTOR = 0.5
    MODEL_LR_SCHEDULER_PATIENT = 0
    MODEL_LR_SCHEDULER_BASELR = 0.0001
    # MODEL_LR_SCHEDULER_THRESHOLD = -2e-3
    MODEL_LR_SCHEDULER_THRESHOLD = 0
    # MODEL_LR_SCHEDULER_MAXLR = 0.01
    MODEL_LR_SCHEDULER_MAXLR = 0.0003125
    MODEL_LR_SCHEDULER_STEP = 1363*2
    MODEL_LR_SCHEDULER_REDUCE_RESTART = 6
    MODEL_LR_SCHEDULER_RESTART_COEF = 1./8
    MODEL_FOLD = 5
    # MODEL_NO_GRAD = [[],]
    MODEL_NO_GRAD = [[-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]]
    MODEL_LEARNING_RATE_AFTER_UNFREEZE = 0.001
    MODEL_FREEZE_EPOCH = 4

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
    EVAL_RATIO = 2 # to 8 when needed
    FIND_LR_ON_VALIDATION = False
    FIND_LR_RATIO = 10 if FIND_LR_ON_VALIDATION else 100

    DIRECTORY_PREFIX = "" # remember to add '/' at the end
    DIRECTORY_CSV = DIRECTORY_PREFIX + 'data/imet_dataset/train.csv'
    DIRECTORY_CSV_ID = 'id'
    DIRECTORY_CSV_TARGET = 'attribute_ids'
    DIRECTORY_SAMPLE_CSV = DIRECTORY_PREFIX + 'data/imet_dataset/sample_submission.csv'
    DIRECTORY_TRAIN = DIRECTORY_PREFIX + "data/imet_dataset/train/"
    DIRECTORY_TEST = DIRECTORY_PREFIX + 'data/imet_dataset/test/'
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
    EVAL_THRESHOLD = 0.3

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