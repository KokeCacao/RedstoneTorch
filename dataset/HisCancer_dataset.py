import collections
import os
import re

import matplotlib as mpl
import torch
import cv2
from sklearn.model_selection import KFold

import config
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from torch._six import string_classes, int_classes
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data.dataloader import numpy_type_map, default_collate
from torchvision.transforms import transforms, Normalize

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, JpegCompression
)# don't import Normalize from albumentations

import tensorboardwriter

if os.environ.get('DISPLAY', '') == '':
    print('WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.')
    mpl.use('Agg')
from matplotlib import pyplot as plt


class HisCancerDataset(data.Dataset):

    def __init__(self, train_csv_dir, test_csv_dir, load_strategy="train", writer=None, column='Target'):
        self.writer = writer
        self.load_strategy = load_strategy
        print("     Reading Data with [test={}]".format(self.load_strategy))

        """Make sure the labels of your dataset is correct"""
        self.train_dataframe = pd.read_csv(train_csv_dir, delimiter=',', encoding="utf-8-sig", engine='python').set_index('Id').sample(frac=1)
        self.test_dataframe = pd.read_csv(test_csv_dir, delimiter=',', encoding="utf-8-sig", engine='python').set_index('Id').sample(frac=1)


        self.multilabel_binarizer = MultiLabelBinarizer().fit([list(range(config.TRAIN_NUM_CLASS))])
        self.labelframe = None

        if self.load_strategy == "train":
            print("Training Dataframe: {}".format(self.train_dataframe.head()))
            self.labelframe = self.multilabel_binarizer.transform([(int(i) for i in str(s).split()) for s in self.train_dataframe[column]])
            id = self.train_dataframe.index.tolist()
        elif self.load_strategy == "test" or self.load_strategy == "predict":
            print("Predicting Dataframe: {}".format(self.test_dataframe.head()))
            self.labelframe = self.multilabel_binarizer.transform([(int(i) for i in str(s).split()) for s in self.test_dataframe[column]])
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

    def get_wsl_samples(self, fold=-1):
        def return_tumor_or_not(dic, one_id):
            return dic[one_id]

        def create_dict():
            df = pd.read_csv("~/RedstoneTorch/data/HisCancer_dataset/train.csv")
            result_dict = {}
            for index in range(df.shape[0]):
                one_id = df.iloc[index, 0]
                tumor_or_not = df.iloc[index, 1]
                result_dict[one_id] = int(tumor_or_not)
            return result_dict

        def find_missing(train_ids, cv_ids):
            all_ids = set(pd.read_csv("~/RedstoneTorch/data/HisCancer_dataset/train.csv")['Id'].values)
            wsi_ids = set(train_ids + cv_ids)

            missing_ids = list(all_ids - wsi_ids)
            return missing_ids

        ids = pd.read_csv("~/RedstoneTorch/data/HisCancer_dataset/patch_id_wsi.csv")

        wsi_dict = dict()
        for i in range(ids.shape[0]):
            wsi = ids.iloc[i, 1]
            train_id = ids.iloc[i, 0]
            wsi_array = wsi.split('_')
            number = int(wsi_array[3])
            if wsi_dict.get(number) is None:
                wsi_dict[number] = [train_id]
            else:
                wsi_dict[number].append(train_id)

        wsi_keys = list(wsi_dict.keys())
        np.random.shuffle(wsi_keys)

        folded_samplers = dict()

        train_ids = dict()
        cv_ids = dict()
        kf = KFold(n_splits=fold, random_state=None, shuffle=False)
        for f, (train_index, test_index) in enumerate(kf.split(wsi_keys)):
            print("TRAIN:", train_index, "TEST:", test_index)
            keys_for_train = [wsi_keys[i] for i in train_index]
            keys_for_cv = [wsi_keys[i] for i in test_index]
            train_ids[f] = np.array([wsi_dict[i] for i in keys_for_train]).flatten()
            cv_ids[f] = np.array([wsi_dict[i] for i in keys_for_cv]).flatten()


        dic = create_dict()
        missing_ids = find_missing(train_ids[0], cv_ids[0])
        np.random.shuffle(missing_ids)

        train_missing_ids = dict()
        cv_missing_ids = dict()
        kf = KFold(n_splits=fold, random_state=None, shuffle=False)
        for f, (train_index, test_index) in enumerate(kf.split(missing_ids)):
            print("TRAIN:", train_index, "TEST:", test_index)
            train_missing_ids[f] = np.array(missing_ids[i] for i in train_index)
            cv_missing_ids[f] = np.array(missing_ids[i] for i in test_index)

        for f in range(fold):
            folded_samplers[f]["train"] = SubsetRandomSampler(train_ids[f] + train_missing_ids[f])
            folded_samplers[f]["val"] = SubsetRandomSampler(cv_ids[f] + cv_missing_ids[f])

            print("########## Fold {} ##########".format_map(f))
            train_labels = []
            cv_labels = []
            train_tumor = 0
            for one_id in train_ids[f]:
                temp = return_tumor_or_not(dic, one_id)
                train_tumor += temp
                train_labels.append(temp)
            cv_tumor = 0
            for one_id in cv_ids[f]:
                temp = return_tumor_or_not(dic, one_id)
                cv_tumor += temp
                cv_labels.append(temp)
            total = len(train_ids[f]) + len(cv_ids[f])
            print("Amount of train labels: {}, {}/{}".format(len(train_ids), train_tumor, len(train_ids) - train_tumor))
            print("Amount of cv labels: {}, {}/{}".format(len(cv_ids), cv_tumor, len(cv_ids) - cv_tumor))
            print("Percentage of cv labels: {}".format(len(cv_ids) / total))

        return folded_samplers

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
            write_cv_distribution(self.writer, y_t, y_e)
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
        id = self.indices_to_id[indice]
        return self.get_load_image_by_id(id, suffix)

    def get_load_image_by_id(self, id, suffix=".npy"):
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

def train_aug(term):
    return Compose([
        Transpose(p=term % 2),
        OneOf([CLAHE(clip_limit=2), IAASharpen(), IAAEmboss(), RandomBrightnessContrast(), JpegCompression(), Blur(), GaussNoise()], p=0.5),
        HueSaturationValue(p=0.5),
        ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5),
    ])
def eval_aug(term):
    return Compose([
        Transpose(p=term % 2),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
            JpegCompression(quality_lower=80, quality_upper=100),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.01, rotate_limit=5, p=0.2),
    ])
def test_aug(term):
    return Compose([
        Transpose(p=term % 2),
    ])

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

def test_collate(batch):
    """TRASNFORM"""
    new_batch = []
    for id, image_0, labels_0 in batch:
        new_batch.append(transform(id, image_0, labels_0, train=False, val=False))
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

    if train is False and val is False:
        term = config.eval_index % 8
        TEST_TRANSFORM = transforms.Compose([
            lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB), # and don't put them in strong_aug()
            lambda x: cv2.resize(x,(config.AUGMENTATION_RESIZE,config.AUGMENTATION_RESIZE), interpolation=cv2.INTER_CUBIC),
            lambda x: RandomRotate90().apply(img=x, factor=term % 4), # pull it out from test_aug because test_aug's Compose cannot contain any lambda
            lambda x: test_aug(term)(image=x), # Yes, you have to use image=xxx
            lambda x: x['image'], # abstract the actual image acter the augmentation
            lambda x: np.clip(x, a_min=0, a_max=255), # make the image within the range
            transforms.ToTensor(),
            Normalize(mean=config.AUGMENTATION_MEAN, std=config.AUGMENTATION_STD), # this needs to be set accordingly
        ])
        image = TEST_TRANSFORM(image_0)
        return (ids, image, labels_0, transforms.ToTensor()(image_0))

    """ https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
    If you are enlarging the image, you should prefer to use INTER_LINEAR or INTER_CUBIC interpolation. If you are shrinking the image, you should prefer to use INTER_AREA interpolation.
    Cubic interpolation is computationally more complex, and hence slower than linear interpolation. However, the quality of the resulting image will be higher.
    """
    if not val and train:
        term = config.epoch % 8
        TRAIN_TRANSFORM = transforms.Compose([
            lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB), # and don't put them in strong_aug()
            lambda x: cv2.resize(x,(config.AUGMENTATION_RESIZE,config.AUGMENTATION_RESIZE), interpolation=cv2.INTER_CUBIC),
            lambda x: RandomRotate90().apply(img=x, factor=term % 4),
            lambda x: train_aug(term)(image=x), # Yes, you have to use image=xxx
            lambda x: x['image'], # abstract the actual image acter the augmentation
            lambda x: np.clip(x, a_min=0, a_max=255), # make the image within the range
            transforms.ToTensor(),
            Normalize(mean=config.AUGMENTATION_MEAN, std=config.AUGMENTATION_STD), # this needs to be set accordingly
        ])
        image = TRAIN_TRANSFORM(image_0)
        return (ids, image, labels_0, transforms.ToTensor()(image_0))
    elif not train and val:
        term = config.eval_index % 8
        PREDICT_TRANSFORM_IMG = transforms.Compose([
            lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB),
            lambda x: cv2.resize(x,(config.AUGMENTATION_RESIZE,config.AUGMENTATION_RESIZE), interpolation=cv2.INTER_CUBIC),
            lambda x: RandomRotate90().apply(img=x, factor=term % 4),
            lambda x: eval_aug(term)(image=x),
            lambda x: x['image'],
            lambda x: np.clip(x, a_min=0, a_max=255),
            transforms.ToTensor(),
            Normalize(mean=config.AUGMENTATION_MEAN, std=config.AUGMENTATION_STD),
        ])
        image = PREDICT_TRANSFORM_IMG(image_0)
        return (ids, image, labels_0, transforms.ToTensor()(image_0))
    else:
        raise RuntimeError("ERROR: Cannot be train and validation at the same time.")
