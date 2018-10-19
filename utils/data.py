import os

import PIL
import pandas as pd
import numpy as np

import torch
from torch import random
from PIL import Image
from skimage import io
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
from torchvision.transforms import transforms

import config


class TGSData(data.Dataset):
    def __init__(self, csv_dir, load_img_dir, load_mask_dir, img_suffix=".png", mask_suffix=".png"):
        print("Reading Data...")
        self.masks_frame = pd.read_csv(csv_dir, index_col=0)
        self.load_img_dir = load_img_dir
        self.load_mask_dir = load_mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix

        self.id = self.get_img_names()
        """WARNING: data length and indices depends on the length of images"""
        self.data_len = len(os.listdir(self.load_img_dir))
        self.indices = list(range(self.data_len))

        # these parameters will be init by get_sampler
        self.indices_to_id = dict()
        self.tran_len = None
        self.val_len = None
        self.train_indices = None
        self.val_indices = None

    def __len__(self):
        return self.data_len

    """CONFIGURATION"""

    def get_img_names(self):
        return (f[:-len(self.img_suffix)].replace("images_original_", "").replace("_groundtruth_(1)_images_", "") for f
                in os.listdir(self.load_img_dir))
        # return (f[:].replace(self.img_suffix, "", 1) for f in os.listdir(self.img_dir))

    def get_sampler(self, data_percent=1.0, val_percent=0.05, data_shuffle=False, train_shuffle=True, val_shuffle=False,
                    seed=19):
        print("     Data Size: {}".format(self.data_len))

        if data_shuffle:
            np.random.seed(seed)
            np.random.shuffle(self.indices)
        self.indices_to_id = dict(zip(self.indices, self.id))

        val_split = int(np.floor(data_percent * val_percent * self.data_len))
        print("     Validation Size: {}".format(val_split))
        self.val_len = val_split
        data_split = int(np.floor(data_percent * self.data_len))
        print("     Traning Size: {}".format(data_split - val_split))
        self.train_len = data_split - val_split

        self.val_indices = self.indices[:val_split]
        if val_shuffle:
            np.random.seed(seed + 1)
            np.random.shuffle(self.val_indices)
            np.random.seed(seed)
        self.train_indices = self.indices[val_split:data_split]
        if train_shuffle:
            np.random.seed(seed + 2)
            np.random.shuffle(self.train_indices)
            np.random.seed(seed)

        self.tran_len = len(self.train_indices)
        self.val_len = len(self.val_indices)

        train_sampler = SubsetRandomSampler(self.train_indices)
        validation_sampler = SubsetRandomSampler(self.val_indices)

        return train_sampler, validation_sampler

    def __getitem__(self, index):
        id = self.indices_to_id.get(index)

        z = self.get_load_z_by_id(id)
        image_0 = self.get_load_image_by_id(id)
        mask_0 = self.get_load_mask_by_id(id)

        """CONFUGURATION
        IMGAUG: https://imgaug.readthedocs.io/en/latest/source/examples_segmentation_maps.html#a-simple-example
        EXAMPLE: https://colab.research.google.com/drive/109vu3F1LTzD1gdVV6cho9fKGx7lzbFll#scrollTo=8q8a2Ha9pnaz
        """

        if index in self.train_indices:
            image_aug_transform = config.ImgAugTransform().to_deterministic()
            TRAIN_TRANSFORM = {
                'image': transforms.Compose([
                    image_aug_transform,
                    lambda x: PIL.Image.fromarray(x),
                    transforms.Resize((224, 224)),
                    # transforms.RandomResizedCrop(224),
                    # transforms.Grayscale(),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean = [0.456, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                ]),
                'mask': transforms.Compose([
                    image_aug_transform,
                    lambda x: PIL.Image.fromarray(x),
                    transforms.Resize((224, 224)),
                    # transforms.CenterCrop(224),
                    transforms.Grayscale(3),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomVerticalFlip(),
                    lambda x: x.convert('L').point(lambda x: 255 if x > 127.5 else 0, mode='1'),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.5, 0.5, 0.5],
                    #                     std=[0.225, 0.225, 0.225]),
                ])
            }

            image = TRAIN_TRANSFORM['image'](image_0)
            mask = TRAIN_TRANSFORM['mask'](mask_0)

            # seq_det = TRAIN_SEQUENCE.to_deterministic()
            # image = seq_det.augment_images(np.array(image))
            # mask = seq_det.augment_images(np.array(mask))

            return (id, z, image, mask, transforms.ToTensor()(image_0), transforms.ToTensor()(mask_0))
        elif index in self.val_indices:
            image = config.PREDICT_TRANSFORM_IMG(image_0)
            mask = config.PREDICT_TRANSFORM_MASK(mask_0)
            return (id, z, image, mask, transforms.ToTensor()(image_0), transforms.ToTensor()(mask_0))
        else:
            return None

    """CONFIGURATION"""

    def get_load_image_by_id(self, id):
        # return Image.open(os.path.join(self.load_img_dir, "images_original_" + id + self.img_suffix)).convert('RGB')
        return Image.open(os.path.join(self.load_img_dir, id + self.img_suffix)).convert('RGB')

    def get_load_mask_by_id(self, id):
        # return Image.open(os.path.join(self.load_mask_dir, "_groundtruth_(1)_images_" + id + self.mask_suffix))
        return Image.open(os.path.join(self.load_mask_dir, id + self.mask_suffix)).convert('RGB')

    def get_load_z_by_id(self, id):
        return self.masks_frame.loc[id, "z"]
