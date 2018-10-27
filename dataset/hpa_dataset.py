import os

import PIL
import pandas as pd
import numpy as np
import torch
import cv2

from imgaug import augmenters as iaa
from PIL import Image
from torch.utils import data
from torch.utils.data import SubsetRandomSampler, Sampler
from torchvision.transforms import transforms

from utils.encode import get_one_hot, inverse_to_tensor

#  what is pinned memory
#       https://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-fortran/
#


# pytorch custom data
#       http://stackoverflow.com/questions/43441673/trying-to-load-a-custom-dataset-in-pytorch
#       http://forums.fast.ai/t/how-do-you-use-custom-dataset-with-pytorch/2040
#       https://discuss.pytorch.org/t/questions-about-imagefolder/774/6


## https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/3
## https://discuss.pytorch.org/t/questions-about-imagefolder/774
## https://github.com/pytorch/tutorials/issues/78
## https://www.kaggle.com/mratsim/planet-understanding-the-amazon-from-space/starting-kit-for-pytorch-deep-learning
## http://pytorch.org/docs/_modules/torch/utils/data/dataset.html
## https://www.kaggle.com/mratsim/planet-understanding-the-amazon-from-space/starting-kit-for-pytorch-deep-learning/notebook
## https://devhub.io/repos/pytorch-vision
## https://github.com/ClementPinard/FlowNetPytorch/blob/master/balancedsampler.py

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
    def __init__(self, csv_dir, load_img_dir, img_suffix=".png"):
        print("Reading Data...")
        self.dataframe = pd.read_csv(csv_dir, engine='python').set_index('Id')
        self.dataframe['Target'] = [[int(i) for i in s.split()] for s in self.dataframe['Target']]
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
        self.img_suffix = img_suffix

        self.id = self.dataframe.index.tolist()
        """WARNING: data length and indices depends on the length of images"""
        self.data_len = int(len(os.listdir(self.load_img_dir)) / 4)
        self.indices = list(range(self.data_len))

        # these parameters will be init by get_sampler
        self.indices_to_id = dict()

        self.train_len = None
        self.val_len = None
        
        """split"""
        self.train_indices = None
        self.val_indices = None

    def __len__(self):
        return self.data_len


    """
        :param self(data_len)
        :param data_percent
        :param val_percent
        
        :var train_len
        :var val_len
        :var train_indices
        :var val_indices
        
        :return folded_sampler
    """
    def get_split_sampler(self, data_percent=1.0, val_percent=0.05):
        self.indices_to_id = dict(zip(self.indices, self.id))
        print("     Data Size: {}".format(self.data_len))

        val_split = int(np.floor(data_percent * val_percent * self.data_len))
        print("     Validation Size: {}".format(val_split))
        self.val_len = val_split
        data_split = int(np.floor(data_percent * self.data_len))
        print("     Traning Size: {}".format(data_split - val_split))
        self.train_len = data_split - val_split

        self.val_indices = self.indices[:val_split]
        self.train_indices = self.indices[val_split:data_split]

        self.train_len = len(self.train_indices)
        self.val_len = len(self.val_indices)

        train_sampler = HPASubsetRandomSampler(self.train_indices, train=True, val=False)
        validation_sampler = HPASubsetRandomSampler(self.val_indices, train=False, val=True)

        return train_sampler, validation_sampler

    """
        :param self(data_len)
        :param foldcv_size
        :return folded_sampler
    """
    def get_fold_sampler(self, fold=-1):
        self.indices_to_id = dict(zip(self.indices, self.id))
        print("     Data Size: {}".format(self.data_len))

        data = self.indices[:-(self.data_len % fold)]
        left_over = self.indices[-(self.data_len % fold):]
        cv_size = len(self.indices) - len(left_over) / fold

        self.train_len = cv_size*(fold-1)
        self.val_len = cv_size

        print("      Cv_size: {}".format(cv_size))
        print("      Fold: {}".format(fold))

        folded_train_indice = dict()
        folded_val_indice = dict()
        folded_samplers = dict()
        for i in range(fold):
            folded_val_indice[i] = list(set(data[i * cv_size:(i + 1) * cv_size]))
            folded_train_indice[i] = list(set(data[:]) - set(folded_val_indice[i]))
            print("     Fold#{}_train_size: {}".format(i, len(folded_train_indice[i])))
            print("     Fold#{}_val_size: {}".format(i, len(folded_val_indice[i])))
            folded_samplers[i] = {}
            folded_samplers[i]["train"] = HPASubsetRandomSampler(folded_train_indice[i], train=True, val=False)
            folded_samplers[i]["val"] = HPASubsetRandomSampler(folded_val_indice[i], train=False, val=True)

        return folded_samplers

    # TODO: Get stratified fold instead of random

    def __getitem__(self, id, train = True, val = True):
        image_0 = self.get_load_image_by_id(id)
        labels_0 = self.get_load_label_by_id(id)

        if not val and train:
            image_aug_transform = TrainImgAugTransform().to_deterministic()
            TRAIN_TRANSFORM = {
                'image': transforms.Compose([
                    image_aug_transform,
                    transforms.ToTensor(),
                ]),
            }

            image = TRAIN_TRANSFORM['image'](image_0)

            # seq_det = TRAIN_SEQUENCE.to_deterministic()
            # image = seq_det.augment_images(np.array(image))
            # mask = seq_det.augment_images(np.array(mask))

            return (id, image, labels_0, inverse_to_tensor(image))
        elif not train and val:
            image_aug_transform = TrainImgAugTransform().to_deterministic()
            PREDICT_TRANSFORM_IMG = transforms.Compose([
                image_aug_transform,
                transforms.ToTensor()
            ])

            image = PREDICT_TRANSFORM_IMG(image_0)
            return (id, image, labels_0, inverse_to_tensor(image))
        else:
            raise RuntimeError("ERROR: Cannot be train and validation at the same time. Please use SubsetSampler provided by the dataset.")

    """CONFIGURATION"""

    # obtain in https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-448-public-lb
    def get_load_image_by_id(self, id):
        """

        :param id: id
        :return: nparray image of (r, g, b, y) from 0~255
        """
        colors = ['red', 'green', 'blue', 'yellow']
        flags = cv2.IMREAD_GRAYSCALE
        imgs = [cv2.imread(os.path.join(self.load_img_dir, id + '_' + color + self.img_suffix), flags).astype(np.float32) / 1 for color in colors]
        return np.stack(imgs, axis=-1)

    def get_load_label_by_id(self, id):
        """

        :param id: id
        :return: one hot encoded label
        """
        labels = self.dataframe.loc[self.dataframe[id], 'Target']
        return get_one_hot(labels, len(self.name_label_dict))

class TrainImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Scale({"height": 224, "width": 224}),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.Noop(), iaa.Add((-40, 40)), iaa.EdgeDetect(alpha=(0.0, 0.1)), iaa.Multiply((0.95, 1.05))], iaa.ContrastNormalization((0.95, 1.05))),
            iaa.OneOf([iaa.Noop(), iaa.PiecewiseAffine(scale=(0.00, 0.02)), iaa.Affine(rotate=(-10, 10)), iaa.Affine(shear=(-10, 10))]),
            iaa.CropAndPad(percent=(-0.12, 0))
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
            iaa.Scale({"height": 224, "width": 224}),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5)
        ], random_order=False)

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

    def to_deterministic(self, n=None):
        self.aug = self.aug.to_deterministic(n)
        return self

class HPASubsetRandomSampler(Sampler):

    def __init__(self, indices, train=False, val=False):
        self.indices = indices
        if train == val: raise RuntimeError("ERROR val_indices=None")
        else:
            self.train = train
            self.val = val

    def __iter__(self):
        return (self.indices.__getitem__(i, train=self.train, val=self.val) for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)