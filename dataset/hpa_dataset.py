import collections
import os
import re

import torch

import cv2
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
from torch._six import string_classes, int_classes
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
from sklearn.preprocessing import MultiLabelBinarizer

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

import config
from utils.encode import to_numpy


class HPAData(data.Dataset):
    """
    Since a multiclass multilabel task is considered,
    there are several things about the model that should
    be pointed out. First, the SOFTMAX MUST NOT BE USED
    as an output layer because it encourages a single label
    prediction. The common output function for multilabel
    tasks is sigmoid. However, combining the sigmoid with
    the loss function (like in BCE with logits loss or in
    Focal loss used in this kernel) allows log(sigmoid)
    optimization of the numerical stability of the loss
    function. Therefore, sigmoid is also removed.

    """

    def __init__(self, csv_dir, load_img_dir=None, img_suffix=".png", test=False, load_preprocessed_dir=None):
        print("Reading Data...")
        self.test = test
        if self.test: print("Prediction Mode Open...")
        self.dataframe = pd.read_csv(csv_dir, engine='python').set_index('Id')
        self.dataframe['Target'] = [(int(i) for i in s.split()) for s in self.dataframe['Target']]
        # self.multilabel_binarizer = MultiLabelBinarizer().fit(self.dataframe['Target'])
        self.multilabel_binarizer = MultiLabelBinarizer()
        self.one_hot_frame = self.multilabel_binarizer.fit_transform(self.dataframe['Target'])

        self.name_label_dict = {
            0: 'Nucleoplasm',
            1: 'Nuclear membrane',
            2: 'Nucleoli',
            3: 'Nucleoli fibrillar center',
            4: 'Nuclear speckles',
            5: 'Nuclear bodies',
            6: 'Endoplasmic reticulum',
            7: 'Golgi apparatus',
            8: 'Peroxisomes',
            9: 'Endosomes',
            10: 'Lysosomes',
            11: 'Intermediate filaments',
            12: 'Actin filaments',
            13: 'Focal adhesion sites',
            14: 'Microtubules',
            15: 'Microtubule ends',
            16: 'Cytokinetic bridge',
            17: 'Mitotic spindle',
            18: 'Microtubule organizing center',
            19: 'Centrosome',
            20: 'Lipid droplets',
            21: 'Plasma membrane',
            22: 'Cell junctions',
            23: 'Mitochondria',
            24: 'Aggresome',
            25: 'Cytosol',
            26: 'Cytoplasmic bodies',
            27: 'Rods & rings'}

        self.load_img_dir = load_img_dir
        self.load_preprocessed_dir = load_preprocessed_dir
        self.img_suffix = img_suffix

        if not self.test:
            self.id = self.dataframe.index.tolist()
            if len(self.id) != len(set(self.id)): print("WARNING: len(self.id) != len(set(self.id))")
            self.id_len = int(len(self.id)*config.TRAIN_DATA_PERCENT)
            self.id = self.id[:self.id_len]
        else:
            self.id = [x.replace(config.DIRECTORY_SUFFIX_IMG, "").replace("_red", "").replace("_green", "").replace("_blue", "").replace("_yellow", "") for x in os.listdir(config.DIRECTORY_TEST)]
            if len(self.id) != len(set(self.id)): print("WARNING: len(self.id) != len(set(self.id))")
            if len(self.dataframe.index.tolist()) != len(set(self.dataframe.index.tolist())): print("WARNING: len(self.dataframe.index.tolist()) != len(set(self.dataframe.index.tolist()))")
            self.id = list(set(self.id) - set(self.dataframe.index.tolist()))
            self.id_len = len(self.id)
        """WARNING: data length and indices depends on the length of images"""
        self.img_len = int(len(os.listdir(self.load_img_dir)) / 4 * config.TRAIN_DATA_PERCENT)
        self.data_len = min(self.img_len, self.id_len)
        if self.img_len != self.id_len and not self.test: raise ResourceWarning("id_len in the csv({}) is not equal to img_len in the folder({}), set data_len to {}".format(self.id_len, self.img_len, self.data_len))
        self.indices = list(range(self.data_len))

        # these parameters will be init by get_sampler
        self.indices_to_id = dict()
        self.id_to_indices = dict()

        self.train_len = None
        self.val_len = None

    def __len__(self):
        return self.data_len

    """
        :param self(data_len)
        :param foldcv_size
        :return folded_sampler
    """

    def get_fold_sampler(self, fold=-1):
        print("     Data Percent: {}".format(config.TRAIN_DATA_PERCENT))
        self.indices_to_id = dict(zip(self.indices, self.id))
        self.id_to_indices = {v: k for k, v in self.indices_to_id.items()}
        print("     Data Size: {}".format(self.data_len))

        data = self.indices[:-(self.data_len % fold)]
        left_over = self.indices[-(self.data_len % fold):]
        cv_size = (len(self.indices) - len(left_over)) / fold

        self.train_len = cv_size * (fold - 1)
        self.val_len = cv_size

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

    # TODO: Get stratified fold instead of random

    def __getitem__(self, indice):
        labels_0 = self.get_load_label_by_indice(indice)
        image_0 = self.get_load_image_by_indice(indice)
        return (self.indices_to_id[indice], image_0, labels_0)

    """CONFIGURATION"""

    # obtain in https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-448-public-lb
    def get_load_image_by_indice(self, indice):
        """

        :param indice: id
        :return: nparray image of (r, g, b, y) from 0~255
        """
        id = self.indices_to_id[indice]
        if config.TRAIN_LOAD_FROM_PREPROCESSED and self.load_preprocessed_dir:
            dir = self.load_preprocessed_dir
            return np.load(os.path.join(dir, id + self.img_suffix))

        dir = self.load_img_dir
        if self.test: dir = config.DIRECTORY_TEST
        colors = ['red', 'green', 'blue', 'yellow']
        flags = cv2.IMREAD_GRAYSCALE
        imgs = [cv2.imread(os.path.join(dir, id + '_' + color + self.img_suffix), flags).astype(np.uint8) for color in colors]
        return np.stack(imgs, axis=-1)

    def get_load_label_by_indice(self, indice):
        """

        :param indice: id
        :return: one hot encoded label
        """
        # return np.float32(self.multilabel_binarizer.transform([self.dataframe['Target'][id]])[0])
        # print(np.float32(self.one_hot_frame[id]))
        return np.float32(self.one_hot_frame[indice])
    def get_load_image_by_id(self, id):
        """

        :param indice: id
        :return: nparray image of (r, g, b, y) from 0~255
        """
        if config.TRAIN_LOAD_FROM_PREPROCESSED and self.load_preprocessed_dir:
            dir = self.load_preprocessed_dir
            return np.load(os.path.join(dir, id + self.img_suffix))

        dir = self.load_img_dir
        if self.test: dir = config.DIRECTORY_TEST
        colors = ['red', 'green', 'blue', 'yellow']
        flags = cv2.IMREAD_GRAYSCALE
        imgs = [cv2.imread(os.path.join(dir, id + '_' + color + self.img_suffix), flags).astype(np.uint8) for color in colors]
        return np.stack(imgs, axis=-1)

    def get_load_label_by_id(self, id):
        """

        :param indice: id
        :return: one hot encoded label
        """
        # return np.float32(self.multilabel_binarizer.transform([self.dataframe['Target'][id]])[0])
        # print(np.float32(self.one_hot_frame[id]))
        return np.float32(self.one_hot_frame[self.id_to_indices[id]])


class TrainImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Scale({"height": config.AUGMENTATION_RESIZE, "width": config.AUGMENTATION_RESIZE}),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.Noop(), iaa.Add((-40, 40)), iaa.EdgeDetect(alpha=(0.0, 0.1)), iaa.Multiply((0.95, 1.05))], iaa.ContrastNormalization((0.95, 1.05))),
            iaa.OneOf([iaa.Noop(), iaa.PiecewiseAffine(scale=(0.00, 0.02)), iaa.Affine(rotate=(-10, 10)), iaa.Affine(shear=(-10, 10))]),
            iaa.CropAndPad(percent=(-0.12, 0)),
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
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.Noop(), iaa.Add((-20, 20)), iaa.EdgeDetect(alpha=(0.0, 0.1)), iaa.Multiply((0.98, 1.02))], iaa.ContrastNormalization((0.99, 1.01))),
            iaa.OneOf([iaa.Noop(), iaa.PiecewiseAffine(scale=(0.00, 0.01)), iaa.Affine(rotate=(-5, 5)), iaa.Affine(shear=(-5, 5))]),
            iaa.CropAndPad(percent=(-0.06, 0)),
        ], random_order=False)

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


def val_collate(batch):
    """TRASNFORM"""
    new_batch = []
    for id, image_0, labels_0 in batch:
        new_batch.append(transform(id, image_0, labels_0, train=False, val=True))
    batch = new_batch

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

    """ https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/69462
    Hi lafoss, 
    just out of interest: How did you calculate these values? I am asking because I did the same a couple of days ago, on the original 512x512 images and got slightly different results, i.e.:
    Means for train image data (originals)
    
    Red average: 0.080441904331346
    Green average: 0.05262986230955176
    Blue average: 0.05474700710311806
    Yellow average: 0.08270895676048498
    
    Means for test image data (originals)
    
    Red average: 0.05908022413399168
    Green average: 0.04532851916280794
    Blue average: 0.040652325092460015
    Yellow average: 0.05923425759572161
    
    Did you resize the images before checking the means? 
    As I say, just out of interest, 
    cheers and thanks, 
    Wolfgang
    """
    if ids is None and labels_0 is None and train is False and val is False: # predict.py
        image_aug_transform = TestImgAugTransform().to_deterministic()
        PREDICT_TRANSFORM_IMG = transforms.Compose([
            image_aug_transform,
            transforms.ToTensor(),
            Normalize(mean=[0.05908022413399168, 0.04532851916280794, 0.040652325092460015, 0.05923425759572161], std=[]),
        ])
        return PREDICT_TRANSFORM_IMG(image_0)

    """
    Train Data:
            Mean = [0.0804419  0.05262986 0.05474701 0.08270896]
            STD  = [0.0025557  0.0023054  0.0012995  0.00293925]
            STD1 = [0.00255578 0.00230547 0.00129955 0.00293934]
    """
    if not val and train:
        image_aug_transform = TrainImgAugTransform().to_deterministic()
        TRAIN_TRANSFORM = {
            'image': transforms.Compose([
                image_aug_transform,
                transforms.ToTensor(),
                Normalize(mean=[0.080441904331346, 0.05262986230955176, 0.05474700710311806, 0.08270895676048498], std=[0.00255578, 0.00230547, 0.00129955, 0.00293934]),
            ]),
        }

        image = TRAIN_TRANSFORM['image'](image_0)
        return (ids, image, labels_0, to_numpy(image))
    elif not train and val:
        image_aug_transform = TestImgAugTransform().to_deterministic()
        PREDICT_TRANSFORM_IMG = transforms.Compose([
            image_aug_transform,
            transforms.ToTensor(),
            Normalize(mean=[0.080441904331346, 0.05262986230955176, 0.05474700710311806, 0.08270895676048498], std=[0.00255578, 0.00230547, 0.00129955, 0.00293934]),
        ])

        image = PREDICT_TRANSFORM_IMG(image_0)
        return (ids, image, labels_0, to_numpy(image))
    else:
        raise RuntimeError("ERROR: Cannot be train and validation at the same time.")
