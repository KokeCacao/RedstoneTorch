import collections
import os
import re

import matplotlib as mpl
import pydicom
import torch
import cv2
from tqdm import tqdm

import config
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch._six import string_classes, int_classes
from torch.utils import data
from torch.utils.data import SubsetRandomSampler, Sampler
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import transforms

from albumentations import (
    HorizontalFlip, CLAHE, ShiftScaleRotate, Blur, GaussNoise, RandomBrightnessContrast, IAASharpen, IAAEmboss, OneOf, Compose, JpegCompression,
    CenterCrop, PadIfNeeded, RandomCrop, RandomGamma, Resize, IAAPiecewiseAffine)
# don't import Normalize from albumentations

import tensorboardwriter
from utils.augmentation import AdaptivePadIfNeeded, DoNothing, RandomPercentCrop
from utils.encode import rle2mask

if os.environ.get('DISPLAY', '') == '':
    print('WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.')
    mpl.use('Agg')
from matplotlib import pyplot as plt


class SIIMDataset(data.Dataset):

    def __init__(self, train_csv_dir, test_csv_dir, load_strategy, writer, id_col, target_col):
        self.writer = writer
        self.load_strategy = load_strategy
        print("     Reading Data with [test={}]".format(self.load_strategy))

        # TODO: preprocess the dataframe so that no multiple masks
        self.train_dataframe = pd.read_csv(train_csv_dir, delimiter=',', encoding="utf-8-sig", engine='python').set_index(id_col).sample(frac=1)
        self.test_dataframe = pd.read_csv(test_csv_dir, delimiter=',', encoding="utf-8-sig", engine='python').set_index(id_col).sample(frac=1)

        self.multilabel_binarizer = MultiLabelBinarizer().fit([list(range(config.TRAIN_NUM_CLASS)), ])

        if self.load_strategy == "train":
            print("Training Dataframe: {}".format(self.train_dataframe.head()))
            self.labelframe = self.train_dataframe[target_col].tolist()
            id = self.train_dataframe.index.tolist()

            # """Presudo Labeling"""
            # self.presudo_dataframe = pd.read_csv(config.DIRECTORY_PRESUDO_CSV, delimiter=',', encoding="utf-8-sig", engine='python').set_index(id_col)
            # for index in self.presudo_dataframe.index.tolist():
            #     probability = self.presudo_dataframe.Label[index]
            #     id.append('data/HisCancer_dataset/test/'+index+'.npy')
            #     self.labelframe.append([1-probability, probability])

        elif self.load_strategy == "test" or self.load_strategy == "predict":
            # TODO
            print("Predicting Dataframe: {}".format(self.test_dataframe.head()))
            self.labelframe = self.test_dataframe[target_col].tolist()
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

        # print("Indice:{}, Id:{}, Label:{}".format(X[0], self.id[0], y[0]))

        mskf = MultilabelStratifiedKFold(n_splits=fold, random_state=None)
        folded_samplers = dict()

        if config.DEBUG_WRITE_SPLIT_CSV or not os.path.exists(config.DIRECTORY_SPLIT):
            print("Could not (or you chose not to) load DIRECTORY_SPLIT. Creating Split manually.")
            if os.path.exists(config.DIRECTORY_SPLIT):
                os.remove(config.DIRECTORY_SPLIT)
                print("WARNING: the split file '{}' already exist, remove file".format(config.DIRECTORY_SPLIT))

            fold_dict = []

            X = self.indices
            # each instance(x) here must be a list with length greater than 1
            # y = np.array(list([self.get_metadata_by_indice(x)[5], 0] for x in X))  # TODO: verify stratify method using AP/PA
            y = np.array(list([self.get_empty_by_indice(x), 0] for x in X))  # TODO: verify stratify method using AP/PA

            print("X = {} ; y = {}".format(X, y))
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
                    tr = ax.bar(np.arange(len(y_t_dict)) - 0.2, y_t_dict, width=0.4, color='tab:red', log=True)
                    ev = ax.bar(np.arange(len(y_e_dict)) + 0.2, y_e_dict, width=0.4, color='tab:blue', log=True)
                    ax.legend((tr[0], ev[0]), ('trian', 'eval'))
                    ax.set_ylabel('exp', color='tab:blue')
                    for i, v in enumerate(y_t_dict): ax.text(i - 0.2, v + 3, str(v), color='red', fontweight='bold')
                    for i, v in enumerate(y_e_dict): ax.text(i + 0.2, v + 3, str(v), color='blue', fontweight='bold')
                    tensorboardwriter.write_data_distribution(self.writer, F, fold)

                # write_cv_distribution(self.writer, y_t, y_e)
            np.save(config.DIRECTORY_SPLIT, fold_dict)
        else:
            fold_dict = np.load(config.DIRECTORY_SPLIT, allow_pickle=True, encoding="latin1")
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
        return (self.indices_to_id[indice], self.get_load_image_by_indice(indice), self.get_load_label_by_indice(indice), self.get_empty_by_indice(indice))

    def get_load_image_by_indice(self, indice):
        id = self.indices_to_id[indice]
        return self.get_load_image_by_id(id)

    def get_load_image_by_id(self, id):
        # TODO: adjust to png or jpg for now
        # TODO : test if it works
        if ".npy" in id:
            img = np.load(config.DIRECTORY_TRAIN + id, allow_pickle=True, encoding="latin1")
        else:
            ds = pydicom.read_file(config.DIRECTORY_TRAIN + id)  # read dicom image
            img = ds.pixel_array  # get image array

            # return np.load(id, allow_pickle=True, encoding="latin1")
        return np.array(np.stack((img,) * 3, -1))

    def get_load_label_by_indice(self, indice):
        # TODO: process label to picture
        # TODO : test if it works
        if len(self.labelframe) - 1 < indice: return None
        img = np.float32(rle2mask(self.labelframe[indice], config.IMG_SIZE, config.IMG_SIZE))
        img = np.transpose(img) # this dataset mask need to be transposed
        return np.stack((img,) * 3, -1)

    def get_load_label_by_id(self, id):
        # TODO: process label to picture
        # TODO : test if it works
        img = np.float32(rle2mask(self.labelframe[self.id_to_indices[id]], config.IMG_SIZE, config.IMG_SIZE))
        img = np.transpose(img) # this dataset mask need to be transposed
        return np.stack((img,) * 3, -1)

    def get_empty_by_indice(self, indice):
        return int(self.labelframe[indice] == '-1')

    def get_empty_by_id(self, id):
        return int(self.labelframe[self.id_to_indices[id]] == '-1')

    def get_metadata_by_id(self, id):
        ds = pydicom.dcmread(config.DIRECTORY_TRAIN + id)

        # ds.dir() # showDicomTags

        # ds.PatientAge
        # ds.PatientSex
        # ds.PixelSpacing
        # ds.ReferringPhysicianName
        # ds.SeriesDescription
        # ds.ViewPosition
        return int(ds.PatientAge), ds.PatientSex, ds.PixelSpacing, ds.ReferringPhysicianName, ds.SeriesDescription, int(ds.ViewPosition == 'AP')

    def get_metadata_by_indice(self, indice):
        return self.get_metadata_by_id(self.indices_to_id[indice])


def no_aug():
    return Resize(config.AUGMENTATION_RESIZE_CHANGE, config.AUGMENTATION_RESIZE_CHANGE, interpolation=cv2.INTER_CUBIC)


def train_aug(term):
    if config.epoch > config.AUGMENTATION_RESIZE_CHANGE_EPOCH:
        return Compose([
            HorizontalFlip(p=term % 2),

            IAAPiecewiseAffine(scale=(0., 0.02)),
            # AdaptivePadIfNeeded(border_mode=cv2.BORDER_CONSTANT),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-0.2, 0.5), rotate_limit=3, border_mode=cv2.BORDER_CONSTANT, p=0.8),

            OneOf([
                RandomPercentCrop(0.9, 0.9),
                DoNothing(p=1),
            ], p=1),
            Resize(config.AUGMENTATION_RESIZE_CHANGE, config.AUGMENTATION_RESIZE_CHANGE, interpolation=cv2.INTER_CUBIC),  # 1344
        ])
    else:
        return Compose([
            HorizontalFlip(p=term % 2),

            IAAPiecewiseAffine(scale=(0., 0.02)),
            # AdaptivePadIfNeeded(border_mode=cv2.BORDER_CONSTANT),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-0.2, 0.5), rotate_limit=3, border_mode=cv2.BORDER_CONSTANT, p=0.8),

            OneOf([
                RandomPercentCrop(0.9, 0.9),
                DoNothing(p=1),
            ], p=1),
            Resize(config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE, interpolation=cv2.INTER_CUBIC),  # 1344
        ])

def eval_aug(term):
    if config.epoch > config.AUGMENTATION_RESIZE_CHANGE_EPOCH:
        return Compose([
            HorizontalFlip(p=term % 2),
            Resize(config.AUGMENTATION_RESIZE_CHANGE, config.AUGMENTATION_RESIZE_CHANGE, interpolation=cv2.INTER_CUBIC),  # 1344
        ])
    else:
        return Compose([
            HorizontalFlip(p=term % 2),
            Resize(config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE, interpolation=cv2.INTER_CUBIC),  # 1344
        ])


def test_aug(term):
    return Compose([
        HorizontalFlip(p=term % 2),
    ])

def tta_aug(term):
    return Compose([
            HorizontalFlip(p=term % 2),

            IAAPiecewiseAffine(scale=(0., 0.02)),
            # AdaptivePadIfNeeded(border_mode=cv2.BORDER_CONSTANT),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-0.2, 0.5), rotate_limit=3, border_mode=cv2.BORDER_CONSTANT, p=0.8),

            OneOf([
                RandomPercentCrop(0.9, 0.9),
                DoNothing(p=1),
            ], p=1),
            Resize(config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE, interpolation=cv2.INTER_CUBIC),  # 1344
        ])


def train_collate(batch):
    new_batch = []
    for id, image_0, labels_0, empty in batch:
        if config.global_steps[config.fold] == 1: print(id, image_0.shape, labels_0.shape)
        new_batch.append(transform(id, image_0, labels_0, empty, mode="train"))
    batch = new_batch
    return default_collate(batch)


def val_collate(batch):
    new_batch = []
    for id, image_0, labels_0, empty in batch:
        new_batch.append(transform(id, image_0, labels_0, empty, mode="val"))
    batch = new_batch
    return default_collate(batch)


def test_collate(batch):
    new_batch = []
    for id, image_0, labels_0, empty in batch:
        new_batch.append(transform(id, image_0, labels_0, empty, mode="test"))
    batch = new_batch
    return default_collate(batch)


def tta_collate(batch):
    new_batch = []
    for id, image_0, labels_0, empty in batch:
        new_batch.append(transform(id, image_0, labels_0, empty, mode="tta"))
    batch = new_batch
    return default_collate(batch)

def transform(ids, image_0, labels_0, empty, mode):
    REGULARIZATION_TRAINSFORM = transforms.Compose([
        lambda x: (cv2.cvtColor(x[0], cv2.COLOR_BGR2GRAY), cv2.cvtColor(x[1], cv2.COLOR_BGR2GRAY)),  # and don't put them in strong_aug()
        lambda x: no_aug()(image=x[0], mask=x[1]),  # Yes, you have to use image=xxx
        lambda x: (np.clip(x['image'], a_min=0, a_max=255), np.clip(x['mask'], a_min=0, a_max=255)),  # make the image within the range
        lambda x: (torch.from_numpy(np.expand_dims(x[0], axis=0)).float().div(255), torch.from_numpy(np.expand_dims(x[1], axis=0)).float().div(255)),  # for 1 dim gray scale
    ])

    """ https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
    If you are enlarging the image, you should prefer to use INTER_LINEAR or INTER_CUBIC interpolation. If you are shrinking the image, you should prefer to use INTER_AREA interpolation.
    Cubic interpolation is computationally more complex, and hence slower than linear interpolation. However, the quality of the resulting image will be higher.
    """
    if mode == "test":
        term = config.eval_index % 8
        TEST_TRANSFORM = transforms.Compose([
            lambda x: (cv2.cvtColor(x[0], cv2.COLOR_BGR2GRAY), cv2.cvtColor(x[1], cv2.COLOR_BGR2GRAY)),  # and don't put them in strong_aug()
            lambda x: test_aug(term)(image=x[0], mask=x[1]),  # Yes, you have to use image=xxx
            lambda x: (np.clip(x['image'], a_min=0, a_max=255), np.clip(x['mask'], a_min=0, a_max=255)),  # make the image within the range
            lambda x: (torch.from_numpy(np.expand_dims(x[0], axis=0)).float().div(255), torch.from_numpy(np.expand_dims(x[1], axis=0)).float().div(255)),  # for 1 dim gray scale
            # Normalize(mean=config.AUGMENTATION_MEAN, std=config.AUGMENTATION_STD), # this needs to be set accordingly
        ])
        image, labels = TEST_TRANSFORM((image_0, labels_0))
        image_0, labels_0 = REGULARIZATION_TRAINSFORM((image_0, labels_0))
        if config.global_steps[config.fold] == 1: print(ids.shape, image.shape, labels.shape, image_0.shape, labels_0.shape, empty.shape)
        return (ids, image, labels, image_0, labels_0, empty)
    elif mode == "train":
        term = config.epoch % 8
        TRAIN_TRANSFORM = transforms.Compose([
            lambda x: (cv2.cvtColor(x[0], cv2.COLOR_BGR2GRAY), cv2.cvtColor(x[1], cv2.COLOR_BGR2GRAY)),  # and don't put them in strong_aug()
            lambda x: train_aug(term)(image=x[0], mask=x[1]),  # Yes, you have to use image=xxx
            lambda x: (np.clip(x['image'], a_min=0, a_max=255), np.clip(x['mask'], a_min=0, a_max=255)),  # make the image within the range
            lambda x: (torch.from_numpy(np.expand_dims(x[0], axis=0)).float().div(255), torch.from_numpy(np.expand_dims(x[1], axis=0)).float().div(255)),  # for 1 dim gray scale
            # Normalize(mean=config.AUGMENTATION_MEAN, std=config.AUGMENTATION_STD), # this needs to be set accordingly
        ])
        image, labels = TRAIN_TRANSFORM((image_0, labels_0))
        image_0, labels_0 = REGULARIZATION_TRAINSFORM((image_0, labels_0))
        if config.global_steps[config.fold] == 1: print(ids.shape, image.shape, labels.shape, image_0.shape, labels_0.shape, empty.shape)
        return (ids, image, labels, image_0, labels_0, empty)
    elif mode == "val":
        term = config.eval_index % 8
        VAL_TRANSFORM = transforms.Compose([
            lambda x: (cv2.cvtColor(x[0], cv2.COLOR_BGR2GRAY), cv2.cvtColor(x[1], cv2.COLOR_BGR2GRAY)),  # and don't put them in strong_aug()
            lambda x: eval_aug(term)(image=x[0], mask=x[1]),  # Yes, you have to use image=xxx
            lambda x: (np.clip(x['image'], a_min=0, a_max=255), np.clip(x['mask'], a_min=0, a_max=255)),  # make the image within the range
            lambda x: (torch.from_numpy(np.expand_dims(x[0], axis=0)).float().div(255), torch.from_numpy(np.expand_dims(x[1], axis=0)).float().div(255)),  # for 1 dim gray scale
            # Normalize(mean=config.AUGMENTATION_MEAN, std=config.AUGMENTATION_STD),
        ])
        image, labels = VAL_TRANSFORM((image_0, labels_0))
        image_0, labels_0 = REGULARIZATION_TRAINSFORM((image_0, labels_0))
        if config.global_steps[config.fold] == 1: print(ids.shape, image.shape, labels.shape, image_0.shape, labels_0.shape, empty.shape)
        return (ids, image, labels, image_0, labels_0, empty)
    elif mode == "tta":
        term = config.eval_index % 8
        TTA_TRANSFORM = transforms.Compose([
            lambda x: (cv2.cvtColor(x[0], cv2.COLOR_BGR2GRAY), cv2.cvtColor(x[1], cv2.COLOR_BGR2GRAY)),  # and don't put them in strong_aug()
            lambda x: tta_aug(term)(image=x[0], mask=x[1]),  # Yes, you have to use image=xxx
            lambda x: (np.clip(x['image'], a_min=0, a_max=255), np.clip(x['mask'], a_min=0, a_max=255)),  # make the image within the range
            lambda x: (torch.from_numpy(np.expand_dims(x[0], axis=0)).float().div(255), torch.from_numpy(np.expand_dims(x[1], axis=0)).float().div(255)),  # for 1 dim gray scale
            # Normalize(mean=config.AUGMENTATION_MEAN, std=config.AUGMENTATION_STD), # this needs to be set accordingly
        ])
        image, labels = TTA_TRANSFORM((image_0, labels_0))
        image_0, labels_0 = REGULARIZATION_TRAINSFORM((image_0, labels_0))
        if config.global_steps[config.fold] == 1: print(ids.shape, image.shape, labels.shape, image_0.shape, labels_0.shape, empty.shape)
        return (ids, image, labels, image_0, labels_0, empty)
