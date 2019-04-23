import collections
import os
import re

import matplotlib as mpl
import torch
import cv2
from tqdm import tqdm

import config
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch._six import string_classes, int_classes
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data.dataloader import numpy_type_map, default_collate
from torchvision.transforms import transforms

from albumentations import (
    HorizontalFlip, CLAHE, ShiftScaleRotate, Blur, GaussNoise, RandomBrightnessContrast, IAASharpen, IAAEmboss, OneOf, Compose, JpegCompression,
    CenterCrop, PadIfNeeded, RandomCrop, RandomGamma, Resize)
# don't import Normalize from albumentations

import tensorboardwriter
from utils.augmentation import AdaptivePadIfNeeded, DoNothing

if os.environ.get('DISPLAY', '') == '':
    print('WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.')
    mpl.use('Agg')
from matplotlib import pyplot as plt


class IMetDataset(data.Dataset):

    def __init__(self, train_csv_dir, test_csv_dir, load_strategy="train", writer=None, id_col = 'id', target_col='target'):
        self.writer = writer
        self.load_strategy = load_strategy
        print("     Reading Data with [test={}]".format(self.load_strategy))

        """Make sure the labels of your dataset is correct"""
        self.train_dataframe = pd.read_csv(train_csv_dir, delimiter=',', encoding="utf-8-sig", engine='python').set_index(id_col).sample(frac=1)
        self.test_dataframe = pd.read_csv(test_csv_dir, delimiter=',', encoding="utf-8-sig", engine='python').set_index(id_col).sample(frac=1)

        self.multilabel_binarizer = MultiLabelBinarizer().fit([list(range(config.TRAIN_NUM_CLASS)),])
        self.labelframe = None

        if self.load_strategy == "train":
            print("Training Dataframe: {}".format(self.train_dataframe.head()))
            self.labelframe = self.multilabel_binarizer.transform([(int(i) for i in str(s).split()) for s in self.train_dataframe[target_col]]).tolist()
            id = self.train_dataframe.index.tolist()

            # """Presudo Labeling"""
            # self.presudo_dataframe = pd.read_csv(config.DIRECTORY_PRESUDO_CSV, delimiter=',', encoding="utf-8-sig", engine='python').set_index(id_col)
            # for index in self.presudo_dataframe.index.tolist():
            #     probability = self.presudo_dataframe.Label[index]
            #     id.append('data/HisCancer_dataset/test/'+index+'.npy')
            #     self.labelframe.append([1-probability, probability])

        elif self.load_strategy == "test" or self.load_strategy == "predict":
            print("Predicting Dataframe: {}".format(self.test_dataframe.head()))
            self.labelframe = self.multilabel_binarizer.transform([(int(i) for i in str(s).split()) for s in self.test_dataframe[target_col]])
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
            ID Size:      {}/{}
            Data Percent:   {}
            Label Size:     {}
            Frame Size:      {}/{}
        """.format(train_csv_dir, test_csv_dir, self.id_len, "?", config.TRAIN_DATA_PERCENT, len(self.labelframe), len(id), "?"))

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


        if config.DEBUG_WRITE_SPLIT_CSV or not os.path.exists(config.DIRECTORY_SPLIT):
            if os.path.exists(config.DIRECTORY_SPLIT):
                os.remove(config.DIRECTORY_SPLIT)
                print("WARNING: the split file '{}' already exist, remove file".format(config.DIRECTORY_SPLIT))

            fold_dict = []
            for fold, (train_index, test_index) in enumerate(mskf.split(X, y)):
                print("#{} TRAIN:{} TEST:{}".format(fold, train_index, test_index))
                x_t = train_index
                # y_t = np.array([y[j] for j in train_index])
                x_e = test_index
                # y_e = np.array([y[j] for j in test_index])

                fold_dict.append([x_t, x_e])

                folded_samplers[fold] = dict()
                folded_samplers[fold]["train"] = SubsetRandomSampler(x_t)
                folded_samplers[fold]["val"] = SubsetRandomSampler(x_e)

                def write_cv_distribution(writer, y_t, y_e):
                    y_t_dict = np.bincount((y_t.astype(np.int8) * np.array(list(range(config.TRAIN_NUM_CLASS)))).flatten())
                    y_e_dict = np.bincount((y_e.astype(np.int8) * np.array(list(range(config.TRAIN_NUM_CLASS)))).flatten())
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

                # write_cv_distribution(self.writer, y_t, y_e)
            np.save(config.DIRECTORY_SPLIT, fold_dict)
        else:
            fold_dict = np.load(config.DIRECTORY_SPLIT)
            pbar = tqdm(fold_dict)
            for fold, items in enumerate(pbar):
                pbar.set_description_str("Creating Folds from Dictionary")
                x_t = items[0]
                x_e = items[1]

                folded_samplers[fold] = dict()
                folded_samplers[fold]["train"] = SubsetRandomSampler(x_t)
                folded_samplers[fold]["val"] = SubsetRandomSampler(x_e)

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

    def get_load_image_by_indice(self, indice):
        id = self.indices_to_id[indice]
        return self.get_load_image_by_id(id)

    def get_load_image_by_id(self, id):
        return np.load(id)

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

def train_aug(term):
    if config.epoch <= config.MODEL_FREEZE_EPOCH +2:
        return Compose([
        HorizontalFlip(p=term % 2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.002, rotate_limit=3, border_mode=cv2.BORDER_CONSTANT, p=0.8),

        # OneOf([CLAHE(clip_limit=2),
        #        IAASharpen(alpha=(0.1, 0.2), lightness=(0.5, 1.)),
        #        IAAEmboss(alpha=(0.1, 0.2), strength=(0.2, 0.7)),
        #        RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
        #        JpegCompression(quality_lower=99, quality_upper=100),
        #        Blur(blur_limit=2),
        #        GaussNoise()], p=0.5),
        RandomGamma(gamma_limit=(90, 110), p=0.8),
        # AdaptivePadIfNeeded(border_mode=cv2.BORDER_CONSTANT),
        PadIfNeeded(config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE, border_mode=cv2.BORDER_CONSTANT),
        Resize(config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE, interpolation=cv2.INTER_CUBIC),
    ])
    elif config.epoch > config.AUGMENTATION_RESIZE_CHANGE_EPOCH:
        return Compose([
            HorizontalFlip(p=term % 2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.002, rotate_limit=3, border_mode=cv2.BORDER_CONSTANT, p=0.8),
            OneOf([CLAHE(clip_limit=2),
                   IAASharpen(alpha=(0.1, 0.2), lightness=(0.5, 1.)),
                   IAAEmboss(alpha=(0.1, 0.2), strength=(0.2, 0.7)),
                   RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
                   JpegCompression(quality_lower=99, quality_upper=100),
                   Blur(blur_limit=2),
                   GaussNoise()], p=0.8),
            RandomGamma(gamma_limit=(90, 110), p=0.8),
            PadIfNeeded(config.AUGMENTATION_RESIZE_CHANGE, config.AUGMENTATION_RESIZE_CHANGE, border_mode=cv2.BORDER_CONSTANT),
            Resize(config.AUGMENTATION_RESIZE_CHANGE, config.AUGMENTATION_RESIZE_CHANGE, interpolation=cv2.INTER_CUBIC),  # 1344
        ])
    else: return Compose([
        HorizontalFlip(p=term % 2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.002, rotate_limit=3, border_mode=cv2.BORDER_CONSTANT, p=0.8),

        OneOf([CLAHE(clip_limit=2),
               IAASharpen(alpha=(0.1, 0.2), lightness=(0.5, 1.)),
               IAAEmboss(alpha=(0.1, 0.2), strength=(0.2, 0.7)),
               RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
               JpegCompression(quality_lower=99, quality_upper=100),
               Blur(blur_limit=2),
               GaussNoise()], p=0.8),
        RandomGamma(gamma_limit=(90, 110), p=0.8),

        # OneOf([
        #        Compose([PadIfNeeded(300, 300, border_mode=cv2.BORDER_REPLICATE), RandomCrop(300, 300)], p=1),
        #        Compose([PadIfNeeded(300, 300, border_mode=cv2.BORDER_REPLICATE), RandomCrop(300, 300)], p=1),
        #        Compose([PadIfNeeded(300, 300, border_mode=cv2.BORDER_REPLICATE), RandomCrop(300, 300)], p=1),
        #        Compose([PadIfNeeded(300, 300, border_mode=cv2.BORDER_REPLICATE), RandomCrop(300, 300)], p=1),
        #        Compose([PadIfNeeded(300, 300, border_mode=cv2.BORDER_REPLICATE), RandomCrop(300, 300)], p=1),
        #        Compose([PadIfNeeded(300, 300, border_mode=cv2.BORDER_REPLICATE), RandomCrop(300, 300)], p=1),
        #        Compose([PadIfNeeded(300, 300, border_mode=cv2.BORDER_REPLICATE), RandomCrop(300, 300)], p=1),
        #        Compose([AdaptivePadIfNeeded(border_mode=cv2.BORDER_CONSTANT),], p=1),
        #        Compose([AdaptivePadIfNeeded(border_mode=cv2.BORDER_CONSTANT),], p=1),
        #        DoNothing(p=1),
        #       ],p=1),
        # 10% force resize
        # 20% black padding to biggest size
        # 70% crop
        PadIfNeeded(config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE, border_mode=cv2.BORDER_CONSTANT),
        Resize(config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE, interpolation=cv2.INTER_CUBIC), #1344
    ])
def eval_aug(term):
    if config.epoch > config.AUGMENTATION_RESIZE_CHANGE_EPOCH:
        return Compose([
            HorizontalFlip(p=term % 2),
            PadIfNeeded(config.AUGMENTATION_RESIZE_CHANGE, config.AUGMENTATION_RESIZE_CHANGE, border_mode=cv2.BORDER_CONSTANT),
            Resize(config.AUGMENTATION_RESIZE_CHANGE, config.AUGMENTATION_RESIZE_CHANGE, interpolation=cv2.INTER_CUBIC),  # 1344
        ])
    else: return Compose([
        HorizontalFlip(p=term % 2),
        PadIfNeeded(config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE, border_mode=cv2.BORDER_CONSTANT),
        Resize(config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE, interpolation=cv2.INTER_CUBIC),  # 1344
    ])
def test_aug(term):
    return Compose([
        HorizontalFlip(p=term % 2),
        PadIfNeeded(config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE, border_mode=cv2.BORDER_CONSTANT),
    ])
def tta_aug(term):
    return train_aug(term)
def train_collate(batch):
    """TRASNFORM"""
    new_batch = []
    for id, image_0, labels_0 in batch:
        if config.global_steps[config.fold] ==1: print(id, image_0.shape, labels_0.shape)
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

def test_collate(batch):
    """TRASNFORM"""
    new_batch = []
    for id, image_0, labels_0 in batch:
        new_batch.append(transform(id, image_0, labels_0, train=False, val=False))
    batch = new_batch
    return collate(batch)

def tta_collate(batch):
    """TRASNFORM"""
    new_batch = []
    for id, image_0, labels_0 in batch:
        new_batch.append(transform(id, image_0, labels_0, train=True, val=True))
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

    REGULARIZATION_TRAINSFORM = transforms.Compose([
            lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB), # and don't put them in strong_aug()
            lambda x: cv2.resize(x,(config.AUGMENTATION_RESIZE,config.AUGMENTATION_RESIZE), interpolation=cv2.INTER_CUBIC),
            lambda x: np.clip(x, a_min=0, a_max=255), # make the image within the range
            transforms.ToTensor(),
        ])

    if not train and not val:
        term = config.eval_index % 8
        TEST_TRANSFORM = transforms.Compose([
            lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB), # and don't put them in strong_aug()
            lambda x: test_aug(term)(image=x), # Yes, you have to use image=xxx
            lambda x: x['image'], # abstract the actual image after the augmentation
            lambda x: np.clip(x, a_min=0, a_max=255), # make the image within the range
            transforms.ToTensor(),
            # Normalize(mean=config.AUGMENTATION_MEAN, std=config.AUGMENTATION_STD), # this needs to be set accordingly
        ])
        image = TEST_TRANSFORM(image_0)
        image_0 = REGULARIZATION_TRAINSFORM(image_0)
        if config.global_steps[config.fold] == 1: print(ids.shape, image.shape, labels_0.shape, image_0.shape)
        return (ids, image, labels_0, image_0)

    """ https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
    If you are enlarging the image, you should prefer to use INTER_LINEAR or INTER_CUBIC interpolation. If you are shrinking the image, you should prefer to use INTER_AREA interpolation.
    Cubic interpolation is computationally more complex, and hence slower than linear interpolation. However, the quality of the resulting image will be higher.
    """
    if not val and train:
        term = config.epoch % 8
        TRAIN_TRANSFORM = transforms.Compose([
            lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB), # and don't put them in strong_aug()
            lambda x: train_aug(term)(image=x), # Yes, you have to use image=xxx
            lambda x: x['image'], # abstract the actual image after the augmentation
            lambda x: np.clip(x, a_min=0, a_max=255), # make the image within the range
            transforms.ToTensor(),
            # Normalize(mean=config.AUGMENTATION_MEAN, std=config.AUGMENTATION_STD), # this needs to be set accordingly
        ])
        image = TRAIN_TRANSFORM(image_0)
        image_0 = REGULARIZATION_TRAINSFORM(image_0)
        if config.global_steps[config.fold] == 1: print(ids.shape, image.shape, labels_0.shape, image_0.shape)
        return (ids, image, labels_0, image_0)
    elif not train and val:
        term = config.eval_index % 8
        PREDICT_TRANSFORM_IMG = transforms.Compose([
            lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB),
            lambda x: eval_aug(term)(image=x),
            lambda x: x['image'],
            lambda x: np.clip(x, a_min=0, a_max=255),
            transforms.ToTensor(),
            # Normalize(mean=config.AUGMENTATION_MEAN, std=config.AUGMENTATION_STD),
        ])
        image = PREDICT_TRANSFORM_IMG(image_0)
        image_0 = REGULARIZATION_TRAINSFORM(image_0)
        if config.global_steps[config.fold] == 1: print(ids.shape, image.shape, labels_0.shape, image_0.shape)
        return (ids, image, labels_0, image_0)
    elif train and val:
        term = config.eval_index % 8
        TTA_TRANSFORM = transforms.Compose([
            lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB), # and don't put them in strong_aug()
            lambda x: tta_aug(term)(image=x), # Yes, you have to use image=xxx
            lambda x: x['image'], # abstract the actual image after the augmentation
            lambda x: np.clip(x, a_min=0, a_max=255), # make the image within the range
            transforms.ToTensor(),
            # Normalize(mean=config.AUGMENTATION_MEAN, std=config.AUGMENTATION_STD), # this needs to be set accordingly
        ])
        image = TTA_TRANSFORM(image_0)
        image_0 = REGULARIZATION_TRAINSFORM(image_0)
        if config.global_steps[config.fold] == 1: print(ids.shape, image.shape, labels_0.shape, image_0.shape)
        return (ids, image, labels_0, image_0)
