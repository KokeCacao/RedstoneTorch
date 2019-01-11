import collections
import os
import re

import imgaug
import matplotlib as mpl
import torch
import cv2
import config
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from torch._six import string_classes, int_classes
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
from sklearn.preprocessing import MultiLabelBinarizer
# from albumentations import (
#     HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
#     Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
#     IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
#     IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
# )

## https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/3
## https://discuss.pytorch.org/t/questions-about-imagefolder/774
## https://github.com/pytorch/tutorials/issues/78
## https://www.kaggle.com/mratsim/planet-understanding-the-amazon-from-space/starting-kit-for-pytorch-deep-learning
## http://pytorch.org/docs/_modules/torch/utils/data/dataset.html
## https://www.kaggle.com/mratsim/planet-understanding-the-amazon-from-space/starting-kit-for-pytorch-deep-learning/notebook
## https://devhub.io/repos/pytorch-vision
## https://github.com/ClementPinard/FlowNetPytorch/blob/master/balancedsampler.py
from torch.utils.data.dataloader import numpy_type_map, default_collate
from torchvision.transforms import transforms, Normalize

import tensorboardwriter

if os.environ.get('DISPLAY', '') == '':
    print('WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.')
    mpl.use('Agg')
from matplotlib import pyplot as plt


class QUBODataset(data.Dataset):

    def __init__(self, train_csv_dir, test_csv_dir, load_strategy="train", writer=None, column='Target'):
        self.writer = writer
        self.load_strategy = load_strategy
        print("     Reading Data with [test={}]".format(self.load_strategy))
        self.train_dataframe = pd.read_csv(train_csv_dir, engine='python').set_index('Id')
        self.test_dataframe = pd.read_csv(test_csv_dir, engine='python').set_index('Id')
        self.multilabel_binarizer = MultiLabelBinarizer().fit([list(range(5))])
        self.labelframe = self.multilabel_binarizer.transform([(int(i) for i in s.split()) for s in self.train_dataframe[column]])

        if self.load_strategy == "train":
            id = self.train_dataframe.index.tolist()
        elif self.load_strategy == "test" or self.load_strategy == "predict":
            id = self.test_dataframe.index.tolist()
        else:
            raise ValueError("the argument [load_strategy] recieved and undefined value: [{}], which is not one of 'train', 'test', 'predict'".format(load_strategy))
        id = list(id)
        self.id_len = int(len(id) * config.TRAIN_DATA_PERCENT)
        self.id = id[:self.id_len]

        self.indices = np.array(list(range(self.id_len)))
        self.indices_to_id = dict(zip(self.indices, self.id))
        self.id_to_indices = {v: k for k, v in self.indices_to_id.items()}

        print("""
            Load Dir:       {}, {}
            Data Size:      {}/{}
            Data Percent:   {}
            Label Size:     {}
            File Size:      {}/{}
        """.format(train_csv_dir, test_csv_dir, self.id_len, "?", config.TRAIN_DATA_PERCENT, len(self.labelframe), len(file), len([x.replace(self.img_suffix, "") for x in os.listdir(self.load_img_dir)])))

    def __len__(self):
        return self.id_len

    def get_stratified_samplers(self, fold=-1):
        """
        :param fold: fold number
        :return: dictionary[fold]["train" or "val"]
        """
        X = self.indices
        y = np.array(list(self.get_load_label_by_indice(x) for x in X))

        # print("Indice:{}, Id:{}, Label:{}".format(X[0], self.id[0], y[0]))

        mskf = MultilabelStratifiedKFold(n_splits=fold, random_state=None)
        folded_samplers = dict()
        for fold, (train_index, test_index) in enumerate(mskf.split(X, y)):
            print("#{} TRAIN: {}, TEST: {}".format(fold, train_index, test_index))
            x_t = train_index
            y_t = np.array([y[j] for j in train_index])
            x_e = test_index
            y_e = np.array([y[j] for j in test_index])
            folded_samplers[fold] = dict()
            folded_samplers[fold]["train"] = SubsetRandomSampler(x_t)

            # a = int(len(x_t)/config.MODEL_BATCH_SIZE)
            # b = 1-config.MODEL_BATCH_SIZE/x_t.shape[0]
            # c = MultilabelStratifiedShuffleSplit(int(a), test_size=b, random_state=None).split(x_t, y_t)
            # folded_samplers[fold]['train'] = iter(c[0])
            folded_samplers[fold]["val"] = SubsetRandomSampler(x_e)  # y[test_index]
            if self.writer:
                y_t_dict = np.bincount((y_t.astype(np.int8)*np.array(list(range(5)))).flatten())
                y_e_dict = np.bincount((y_e.astype(np.int8)*np.array(list(range(5)))).flatten())
                # F, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2), sharey='none')
                # ax1.bar(list(range(len(y_t_dict))), y_t_dict)
                # ax2.bar(list(range(len(y_e_dict))), y_e_dict)
                F = plt.figure()
                ax = F.add_subplot(111)
                tr = ax.bar(np.arange(len(y_t_dict)) -0.2, y_t_dict, width=0.4, color='tab:red', log=True)
                ev = ax.bar(np.arange(len(y_e_dict)) +0.2, y_e_dict, width=0.4, color='tab:blue', log=True)
                ax.legend((tr[0], ev[0]), ('trian', 'eval'))
                ax.set_ylabel('exp', color='tab:blue')
                for i, v in enumerate(y_t_dict): ax.text(i - 0.2, v + 3, str(v), color='red', fontweight='bold')
                for i, v in enumerate(y_e_dict): ax.text(i + 0.2, v + 3, str(v), color='blue', fontweight='bold')
                tensorboardwriter.write_data_distribution(self.writer, F, fold)


            # gc.collect()
        return folded_samplers

    def get_fold_samplers(self, fold=-1):

        data = self.indices[:-(self.id_len % fold)]
        left_over = self.indices[-(self.id_len % fold):]
        cv_size = (len(self.indices) - len(left_over)) / fold

        print("     CV_size: {}".format(cv_size))
        print("     Fold: {}".format(fold))

        folded_train_indice = dict()
        folded_val_indice = dict()
        folded_samplers = dict()
        for i in range(fold):
            folded_val_indice[i] = list(set(data[i * cv_size:(i + 1) * cv_size]))
            folded_train_indice[i] = list(set(data[:]) - set(folded_val_indice[i]))
            print("     Fold#{}_train_size: {}".format(i, len(folded_train_indice[i])))
            print("     Fold#{}_val_size: {} + {}".format(i, len(folded_val_indice[i]), len(left_over)))
            folded_samplers[i] = {}
            folded_samplers[i]["train"] = SubsetRandomSampler(folded_train_indice[i])
            folded_samplers[i]["val"] = SubsetRandomSampler(folded_val_indice[i] + left_over)

        return folded_samplers

    def __getitem__(self, indice):
        """

        :param indice:
        :return: id, one hot encoded label, nparray image of (r, g, b, y) from 0~255 (['red', 'green', 'blue', 'yellow']) (4, W, H)
        """
        return (self.indices_to_id[indice], self.get_load_image_by_indice(indice), self.get_load_label_by_indice(indice))

    def get_load_image_by_indice(self, indice, suffix=".npy"):
        """

        :param indice: id
        :return: nparray image of (r, g, b, y) from 0~255 (['red', 'green', 'blue', 'yellow']) (4, W, H)
        """
        id = self.indices_to_id[indice]
        return self.get_load_image_by_id(id, suffix)

    def get_load_image_by_id(self, id, suffix=".npy"):
        """

        :param indice: id
        :return: nparray image of (r, g, b, y) from 0~255 (['red', 'green', 'blue', 'yellow']) (4, W, H)
        """
        return np.load(id + suffix)

    def get_load_label_by_indice(self, indice):
        """

        :param indice: id
        :return: one hot encoded label
        """
        if len(self.labelframe) - 1 < indice: return None
        return np.float32(self.labelframe[indice])

    def get_load_label_by_id(self, id):
        """

        :param indice: id
        :return: one hot encoded label
        """
        return np.float32(self.labelframe[self.id_to_indices[id]])

class TrainImgAugTransform:
    """
    Dont crop: not every cell has perfect data, some lose some class if you crop it
    You can mess up with location info: they are not that important
    Dont twist too much green data: their structure is important
    You can think of this challenge as
    Add Negative Sample: no green layer should have no sample
    Wrap is bad
    Green layer's Sharpon should be carefully designed. It should adjust with other paremeters.
    Dropout need to tested by network
    iaa.ContrastNormalization((x, x)) Will change background and amount of green. be careful.

    """
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.CropAndPad(percent=(0, 0.1), pad_mode=["constant", "reflect"], pad_cval=0),
            iaa.OneOf([
                iaa.Noop(),
                iaa.PiecewiseAffine(scale=(0.00, 0.02), nb_rows=4, nb_cols=4, mode=["constant", "reflect", "wrap"], cval=0),
                iaa.Affine(rotate=(-10, 10), mode=["constant", "reflect"], cval=0),
                iaa.Affine(shear=(-10, 10), mode=["constant", "reflect"], cval=0),
            ]),
            iaa.PiecewiseAffine(scale=(0.00, 0.05), nb_rows=4, nb_cols=4, mode=["constant", "reflect", "wrap"], cval=0),
            iaa.ContrastNormalization((1.0, 1.01)),
            iaa.Scale({"height": config.AUGMENTATION_RESIZE, "width": config.AUGMENTATION_RESIZE}, interpolation=['nearest', 'linear', 'area', 'cubic']),
            iaa.WithChannels([0,2,3], iaa.Sequential([
                iaa.OneOf([
                    iaa.Noop(),
                    iaa.EdgeDetect(alpha=(0.0, 0.1)),
                    iaa.Multiply((0.8, 1.3), per_channel=1.0),
                    iaa.ContrastNormalization((0.95, 1.05))
                ]),
                iaa.OneOf(
                    iaa.CoarseDropout((0.0, 0.02), size_percent=(0.005, 0.005), per_channel=1.0),
                ),
                iaa.Sharpen(alpha=(0.0, 0.25), lightness=(0.0, 0.45)),
                ])),
            iaa.WithChannels([1], iaa.Sequential([
                iaa.OneOf([
                    iaa.Noop(),
                    iaa.Multiply((1.0, 1.15)),
                    iaa.ContrastNormalization((1.0, 1.01)),
                ]),
                iaa.Sharpen(alpha=(0.24, 0.26), lightness=(0.44, 0.46)),
            ])),

        ], random_order=False)
        chance = config.epoch / 8
        if chance == 0:
            pass
        elif chance == 1:
            self.aug.add(iaa.Fliplr(1))
        elif chance == 2:
            self.aug.add(iaa.Fliplr(1))
            self.aug.add(iaa.Flipud(1))
        elif chance == 3:
            self.aug.add(iaa.Flipud(1))
        elif chance == 4:
            self.aug.add(iaa.Affine(rotate=90))
        elif chance == 5:
            self.aug.add(iaa.Affine(rotate=90))
            self.aug.add(iaa.Fliplr(1))
        elif chance == 6:
            self.aug.add(iaa.Affine(rotate=90))
            self.aug.add(iaa.Fliplr(1))
            self.aug.add(iaa.Flipud(1))
        elif chance == 7:
            self.aug.add(iaa.Affine(rotate=90))
            self.aug.add(iaa.Flipud(1))
        else:
            raise ValueError("Chance cannot equal to other number other than [0, 7]")

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

    def to_deterministic(self, n=None):
        self.aug = self.aug.to_deterministic(n)
        return self

class AggressiveTrainImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.CropAndPad(percent=(0, 0.1), pad_mode=["constant", "reflect"], pad_cval=0),
            iaa.OneOf([
                iaa.Noop(),
                iaa.PiecewiseAffine(scale=(0.00, 0.02), nb_rows=4, nb_cols=4, mode=["constant", "reflect", "wrap"], cval=0),
                iaa.Affine(rotate=(-10, 10), mode=["constant", "reflect"], cval=0),
                iaa.Affine(shear=(-10, 10), mode=["constant", "reflect"], cval=0),
            ]),
            iaa.PiecewiseAffine(scale=(0.00, 0.05), nb_rows=4, nb_cols=4, mode=["constant", "reflect", "wrap"], cval=0),
            iaa.ContrastNormalization((1.0, 1.01)),
            iaa.Scale({"height": config.AUGMENTATION_RESIZE, "width": config.AUGMENTATION_RESIZE}, interpolation=['nearest', 'linear', 'area', 'cubic']),
            iaa.WithChannels([0,2,3], iaa.Sequential([
                iaa.OneOf([
                    iaa.Noop(),
                    iaa.EdgeDetect(alpha=(0.0, 0.1)),
                    iaa.Multiply((0.2, 1.3), per_channel=1.0),
                    iaa.ContrastNormalization((0.95, 1.05))
                ]),
                iaa.OneOf(
                    iaa.CoarseDropout((0.0, 0.02), size_percent=(0.005, 0.005), per_channel=1.0),
                ),
                iaa.Sharpen(alpha=(0.0, 0.25), lightness=(0.0, 0.45)),
                ])),
            iaa.WithChannels([1], iaa.Sequential([
                iaa.OneOf([
                    iaa.Noop(),
                    iaa.Multiply((1.0, 1.15)),
                    iaa.ContrastNormalization((1.0, 1.01)),
                ]),
                iaa.Sharpen(alpha=(0.24, 0.26), lightness=(0.44, 0.46)),
            ])),

        ], random_order=False)
        chance = config.epoch / 8
        if chance == 0:
            pass
        elif chance == 1:
            self.aug.add(iaa.Fliplr(1))
        elif chance == 2:
            self.aug.add(iaa.Fliplr(1))
            self.aug.add(iaa.Flipud(1))
        elif chance == 3:
            self.aug.add(iaa.Flipud(1))
        elif chance == 4:
            self.aug.add(iaa.Affine(rotate=90))
        elif chance == 5:
            self.aug.add(iaa.Affine(rotate=90))
            self.aug.add(iaa.Fliplr(1))
        elif chance == 6:
            self.aug.add(iaa.Affine(rotate=90))
            self.aug.add(iaa.Fliplr(1))
            self.aug.add(iaa.Flipud(1))
        elif chance == 7:
            self.aug.add(iaa.Affine(rotate=90))
            self.aug.add(iaa.Flipud(1))
        else:
            raise ValueError("Chance cannot equal to other number other than [0, 7]")


    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

    def to_deterministic(self, n=None):
        self.aug = self.aug.to_deterministic(n)
        return self

class PredictImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Scale({"height": config.AUGMENTATION_RESIZE, "width": config.AUGMENTATION_RESIZE}),
        ], random_order=False)

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

    def to_deterministic(self, n=None):
        self.aug = self.aug.to_deterministic(n)
        return self


class TestImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Scale({"height": config.AUGMENTATION_RESIZE, "width": config.AUGMENTATION_RESIZE}),
        ], random_order=False)
        chance = config.eval_index / 8
        if chance == 0:
            pass
        elif chance == 1:
            self.aug.add(iaa.Fliplr(1))
        elif chance == 2:
            self.aug.add(iaa.Fliplr(1))
            self.aug.add(iaa.Flipud(1))
        elif chance == 3:
            self.aug.add(iaa.Flipud(1))
        elif chance == 4:
            self.aug.add(iaa.Affine(rotate=90))
        elif chance == 5:
            self.aug.add(iaa.Affine(rotate=90))
            self.aug.add(iaa.Fliplr(1))
        elif chance == 6:
            self.aug.add(iaa.Affine(rotate=90))
            self.aug.add(iaa.Fliplr(1))
            self.aug.add(iaa.Flipud(1))
        elif chance == 7:
            self.aug.add(iaa.Affine(rotate=90))
            self.aug.add(iaa.Flipud(1))
        else:
            raise ValueError("Chance cannot equal to other number other than [0, 7]")

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

    def to_deterministic(self, n=None):
        self.aug = self.aug.to_deterministic(n)
        return self


def train_collate(batch):
    """TRASNFORM"""
    new_batch = []
    for id, image_0, labels_0 in batch:
        new_batch.append(transform(id, image_0, labels_0, train=True, val=False))
    batch = new_batch
    return collate(batch)


def val_collate(batch):
    """TRASNFORM"""
    new_batch = []
    for id, image_0, labels_0 in batch:
        new_batch.append(transform(id, image_0, labels_0, train=False, val=True))
    batch = new_batch
    return collate(batch)


def collate(batch):
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def transform(ids, image_0, labels_0, train, val):
    """

    :param ids:
    :param image_0:
    :param labels_0:
    :param train:
    :param val:
    :return:
    """

    if ids is None and labels_0 is None and train is False and val is False:  # predict.py
        image_aug_transform = PredictImgAugTransform().to_deterministic()
        PREDICT_TRANSFORM_IMG = transforms.Compose([
            image_aug_transform,
            lambda x: np.clip(x, a_min=0, a_max=255),
            transforms.ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return PREDICT_TRANSFORM_IMG(image_0)

    """
    Train Data:
            Mean = [0.0804419  0.05262986 0.05474701 0.08270896]
            STD  = [0.0025557  0.0023054  0.0012995  0.00293925]
            STD1 = [0.00255578 0.00230547 0.00129955 0.00293934]
    """
    if not val and train:
        image_aug_transform = AggressiveTrainImgAugTransform().to_deterministic() if config.epoch > 20 else TrainImgAugTransform().to_deterministic()
        TRAIN_TRANSFORM = transforms.Compose([
            image_aug_transform,
            lambda x: np.clip(x, a_min=0, a_max=255),
            transforms.ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        image = TRAIN_TRANSFORM(image_0)
        return (ids, image, labels_0, transforms.ToTensor()(image_0))
    elif not train and val:
        image_aug_transform = TrainImgAugTransform().to_deterministic()
        PREDICT_TRANSFORM_IMG = transforms.Compose([
            image_aug_transform,
            lambda x: np.clip(x, a_min=0, a_max=255),
            transforms.ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        image = PREDICT_TRANSFORM_IMG(image_0)
        return (ids, image, labels_0, transforms.ToTensor()(image_0))
    else:
        raise RuntimeError("ERROR: Cannot be train and validation at the same time.")
