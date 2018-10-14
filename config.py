import PIL
import numpy as np

from datetime import datetime
from imgaug import augmenters as iaa
from torchvision.transforms import transforms

MODEL_EPOCHS = 300
MODEL_BATCH_SIZE= 16
MODEL_LEARNING_RATE = 0.001
MODEL_MOMENTUM = 0.9
MODEL_WEIGHT_DEFAY = 0.0001

TRAIN_GPU = "0,1"
TRAIN_LOAD = None
TRAIN_VAL_PERCENT = 0.05
TRAIN_DATA_PERCENT = 1.0
TRAIN_VISUALIZATION = True
TRAIN_TAG = "train"
TRAIN_SEED = 19
TRAIN_VALIDATION = True
TRAIN_TAG = str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-") + "-" + TRAIN_TAG
TRAIN_SAVE_CHECKPOINT = True
TRAIN_NUM_WORKER = 0
TRAIN_HISTOGRAM = False
TRAIN_TRY_THRESHOLD = np.linspace(0.3, 0.7, 31)
TRAIN_THRESHOLD_TEST = True

DIRECTORY_PREFIX = ""
DIRECTORY_SUFFIX_IMG = ".png"
DIRECTORY_SUFFIX_MASK = ".png"
DIRECTORY_IMG = DIRECTORY_PREFIX + 'data/train/images/'  # augmentation
DIRECTORY_MASK = DIRECTORY_PREFIX + 'data/train/masks/'  # augmentation
# DIRECTORY_UNTRANSFORMED_IMG = DIRECTORY_PREFIX + 'data/train/images/'  # augmentation
# DIRECTORY_UNTRANSFORMED_MASK = DIRECTORY_PREFIX + 'data/train/masks/'  # augmentation
DIRECTORY_DEPTH = DIRECTORY_PREFIX + 'data/depths.csv'
DIRECTORY_CHECKPOINT = DIRECTORY_PREFIX + "tensorboard/" + TRAIN_TAG + "/checkpoints/"
DIRECTORY_TEST = DIRECTORY_PREFIX + 'data/test/images/'

PREDICTION_TAG = "test"
PREDICTION_LOAD_TAG = ""

global_step = 0

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
               iaa.Scale({"height": 224, "width": 224}),
               iaa.Fliplr(0.5),
               iaa.Flipud(0.5),
               iaa.OneOf([iaa.Noop(), iaa.Add((-40, 40)), iaa.EdgeDetect(alpha=(0.0, 0.1)), iaa.Multiply((0.95, 1.05))], iaa.ContrastNormalization((0.95, 1.05))),
               iaa.OneOf([iaa.Noop(), iaa.PiecewiseAffine(scale=(0.00, 0.02)), iaa.Affine(rotate=(-10,10)), iaa.Affine(shear=(-10, 10))]),
               iaa.CropAndPad(percent=(-0.12, 0))
               ], random_order=False)

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

    def to_deterministic(self, n=None):
        self.aug = self.aug.to_deterministic(n)
        return self

PREDICT_TRANSFORM = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(3),
                transforms.ToTensor()
            ])

PREDICT_TRANSFORM_Back = transforms.Compose([
                transforms.Resize((101, 101)),
                transforms.Grayscale(),
                lambda x: x>args.shreshold * 1.0
            ])


# TRAIN_SEQUENCE = iaa.Sequential([
#                iaa.Scale({"height": 224, "width": 224}),
#                iaa.Fliplr(0.5),
#                iaa.Flipud(0.5),
#                # iaa.OneOf([iaa.Noop(), iaa.Add((-40, 40)), iaa.EdgeDetect(alpha=(0.0, 0.1)), iaa.Multiply((0.95, 1.05))], iaa.ContrastNormalization((0.95, 1.05))),
#                # iaa.OneOf([iaa.Noop(), iaa.PiecewiseAffine(scale=(0.00, 0.02)), iaa.Affine(rotate=(-10,10)), iaa.Affine(shear=(-10, 10))]),
#                iaa.CropAndPad(percent=(-0.12, 0))
#                ], random_order=True)

# transform = {
#     # 'depth': transforms.Compose([
#     #     transforms.ToTensor(),
#     #     transforms.Normalize([0.5], [0.5])
#     # ]),
#     'image': transforms.Compose([
#         transforms.Resize((224,224)),
#         # transforms.RandomResizedCrop(224),
#         transforms.Grayscale(),
#         # transforms.RandomHorizontalFlip(),
#         # transforms.RandomVerticalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean = [0.5], std = [0.225])
#     ]),
#     'mask': transforms.Compose([
#         transforms.Resize((224,224)),
#         # transforms.CenterCrop(224),
#         transforms.Grayscale(),
#         # transforms.RandomHorizontalFlip(),
#         # transforms.RandomVerticalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean = [0.5], std = [0.225]),
#         lambda x: x>0,
#         lambda x: x.float()
#     ])
# }