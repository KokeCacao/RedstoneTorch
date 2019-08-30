import collections
import os
import re

import cv2
import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
from albumentations import (
    HorizontalFlip, CLAHE, ShiftScaleRotate, Blur, GaussNoise, RandomBrightnessContrast, IAASharpen, IAAEmboss, OneOf, Compose, JpegCompression,
    CenterCrop, PadIfNeeded, RandomCrop, RandomGamma, Resize)
from sklearn.preprocessing import MultiLabelBinarizer
from torch._six import string_classes, int_classes
from torch.utils import data
"""OUTDATED IMPORT"""
# from torch.utils.data.dataloader import numpy_type_map, default_collate
from torchvision.transforms import transforms

# don't import Normalize from albumentations
from submission import config

if os.environ.get('DISPLAY', '') == '':
    print('WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.')
    mpl.use('Agg')


class IMetDataset(data.Dataset):

    def __init__(self, train_csv_dir, test_csv_dir, writer=None, id_col = 'id', target_col='target'):
        self.writer = writer

        """Make sure the labels of your dataset is correct"""
        self.test_dataframe = pd.read_csv(test_csv_dir, delimiter=',', encoding="utf-8-sig", engine='python').set_index(id_col).sample(frac=1)

        self.multilabel_binarizer = MultiLabelBinarizer().fit([list(range(config.TRAIN_NUM_CLASS)),])
        self.labelframe = None

        print("Predicting Dataframe: {}".format(self.test_dataframe.head()))
        self.labelframe = self.multilabel_binarizer.transform([(int(i) for i in str(s).split()) for s in self.test_dataframe[target_col]])
        id = ["../input/imet-2019-fgvc6/test/"+s+".png" for s in self.test_dataframe.index.tolist()]

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
        return cv2.imread(id, cv2.IMREAD_COLOR).astype(np.uint8)

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

def test_aug(term):
    return Compose([
        HorizontalFlip(p=term % 2),
        PadIfNeeded(300, 300, border_mode=cv2.BORDER_REPLICATE),
        CenterCrop(300, 300),
        Resize(224, 224, interpolation=cv2.INTER_CUBIC),
    ])
def tta_aug(term):
    return Compose([
        HorizontalFlip(p=term % 2),
        ShiftScaleRotate(shift_limit=0.00625, scale_limit=0.002, rotate_limit=2, p=0.5),

        OneOf([CLAHE(clip_limit=2),
               IAASharpen(alpha=(0.1, 0.2), lightness=(0.5, 1.)),
               IAAEmboss(alpha=(0.1, 0.2), strength=(0.2, 0.7)),
               RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
               JpegCompression(quality_lower=99, quality_upper=100),
               Blur(blur_limit=2),
               GaussNoise()], p=0.5),
        RandomGamma(gamma_limit=(90, 110), p=0.5),

        OneOf([
            Compose([PadIfNeeded(300, 300, border_mode=cv2.BORDER_REPLICATE), RandomCrop(300, 300)], p=1),
            Compose([PadIfNeeded(300, 300, border_mode=cv2.BORDER_REPLICATE), RandomCrop(300, 300)], p=1),
            Compose([PadIfNeeded(300, 300, border_mode=cv2.BORDER_REPLICATE), RandomCrop(300, 300)], p=1),
            Compose([PadIfNeeded(300, 300, border_mode=cv2.BORDER_REPLICATE), RandomCrop(300, 300)], p=1),
            Compose([PadIfNeeded(300, 300, border_mode=cv2.BORDER_REPLICATE), RandomCrop(300, 300)], p=1),
            Compose([PadIfNeeded(300, 300, border_mode=cv2.BORDER_REPLICATE), RandomCrop(300, 300)], p=1),
            Compose([PadIfNeeded(300, 300, border_mode=cv2.BORDER_REPLICATE), RandomCrop(300, 300)], p=1),
            # Compose([AdaptivePadIfNeeded(border_mode=cv2.BORDER_CONSTANT),], p=1),
            # Compose([AdaptivePadIfNeeded(border_mode=cv2.BORDER_CONSTANT),], p=1),
            # DoNothing(p=1),
        ], p=1),
        # 10% force resize
        # 20% black padding to biggest size
        # 70% crop
        Resize(224, 224, interpolation=cv2.INTER_CUBIC),
    ])
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
        # if config.global_steps[config.fold] == 1: print(ids.shape, image.shape, labels_0.shape, image_0.shape)
        return (ids, image, labels_0, image_0)

    if train and val:
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
        # if config.global_steps[config.fold] == 1: print(ids.shape, image.shape, labels_0.shape, image_0.shape)
        return (ids, image, labels_0, image_0)
