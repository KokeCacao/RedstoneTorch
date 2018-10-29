import numpy as np

from datetime import datetime

import torch
from torch.autograd import Variable
from torchvision.transforms import transforms

from utils.encode import tensor_to_PIL

MODEL_EPOCHS = 3
MODEL_BATCH_SIZE = 8
MODEL_LEARNING_RATE = 0.0005
MODEL_MOMENTUM = 0.9
MODEL_WEIGHT_DEFAY = 0.0001
MODEL_FOLD = 3

TRAIN_GPU_ARG = "0,1"
TRAIN_GPU_LIST = [int(i) for i in TRAIN_GPU_ARG.split(",")]
TRAIN_VAL_PERCENT = 0.05
TRAIN_DATA_PERCENT = 0.01
TRAIN_SEED = 19
TRAIN_SAVE_CHECKPOINT = True
TRAIN_NUM_WORKER = 4*len(TRAIN_GPU_LIST) # idea from: https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
TRAIN_RESUME = True
TRAIN_NUMCLASS = 28

PROJECT_TAG = "test"
PROJECT_TAG = str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-") + "-" + PROJECT_TAG

DISPLAY_HISTOGRAM = False
DISPLAY_VISUALIZATION = True

EVAL_THRESHOLD_TEST = True
EVAL_TRY_THRESHOLD = np.linspace(0.3, 0.7, 21)
EVAL_CHOSEN_THRESHOLD = 0.5

DIRECTORY_PREFIX = ""
DIRECTORY_SUFFIX_IMG = ".png"
DIRECTORY_SUFFIX_MASK = ".png"
DIRECTORY_IMG = DIRECTORY_PREFIX + "data/train/"
DIRECTORY_MASK = DIRECTORY_PREFIX + 'data/train/masks/'  # augmentation

DIRECTORY_LOAD = None
DIRECTORY_CSV = DIRECTORY_PREFIX + 'data/train.csv'
DIRECTORY_CHECKPOINT = DIRECTORY_PREFIX + "model/" + PROJECT_TAG + "/"
DIRECTORY_TEST = DIRECTORY_PREFIX + 'data/test/'
DIRECTORY_CP_NAME = 'CP{}.pth'

PREDICTION_TAG = "test"
PREDICTION_LOAD_TAG = ""
PREDICTION_SAVE_IMG = False
PREDICTION_DARK_THRESHOLD = 0.05

lastsave = None
global_steps = []
epoch = 0
fold = 0
versiontag = ""

PREDICT_TRANSFORM_BACK = transforms.Compose([
    lambda x: (x > Variable(torch.Tensor([EVAL_CHOSEN_THRESHOLD])).cuda()).float() * 1,
    lambda x: tensor_to_PIL(x),
    transforms.Resize((101, 101)),
    transforms.Grayscale(),
])
