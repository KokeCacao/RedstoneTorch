import os
import pandas as pd
import numpy as np

import torch
from torch import random
from PIL import Image
from skimage import io
from torch.utils import data
from torch.utils.data import SubsetRandomSampler

import config


class TGSData(data.Dataset):
    def __init__(self, csv_dir, load_img_dir, load_mask_dir, img_suffix=".png", mask_suffix=".png", transform=None):
        print("Reading Data...")
        self.masks_frame = pd.read_csv(csv_dir, index_col=0)
        self.load_img_dir = load_img_dir
        self.load_mask_dir = load_mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform

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
        print("Total Size:", self.data_len)

        if data_shuffle:
            np.random.seed(seed)
            np.random.shuffle(self.indices)
        self.indices_to_id = dict(zip(self.indices, self.id))

        val_split = int(np.floor(data_percent * val_percent * self.data_len))
        print("Validation Size: {}".format(val_split))
        self.val_len = val_split
        data_split = int(np.floor(data_percent * self.data_len))
        print("Traning Size: {}".format(data_split - val_split))
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
        image = self.get_load_image_by_id(id)
        mask = self.get_load_mask_by_id(id)

        image = config.TRAIN_TRASNFORM['image'](image)
        mask = config.TRAIN_TRASNFORM['mask'](mask)

        seq_det = config.TRAIN_SEQUENCE.to_deterministic()
        image = seq_det.augment_images([np.array(image)])
        mask = seq_det.augment_images([np.array(mask)])

        return (np.array(id), np.array(z), torch.from_numpy(image), torch.from_numpy(mask))

    """CONFIGURATION"""
    def get_load_image_by_id(self, id):
        # return Image.open(os.path.join(self.load_img_dir, "images_original_" + id + self.img_suffix)).convert('RGB')
        return Image.open(os.path.join(self.load_img_dir, id + self.img_suffix)).convert('RGB')
    def get_load_mask_by_id(self, id):
        # return Image.open(os.path.join(self.load_mask_dir, "_groundtruth_(1)_images_" + id + self.mask_suffix))
        return Image.open(os.path.join(self.load_mask_dir, id + self.mask_suffix)).convert('RGB')
    def get_load_z_by_id(self, id):
        return self.masks_frame.loc[id, "z"]

    # removed due to big data
    # def get_all_sample(self, ids, seed=19):
    #     random.manual_seed(seed)
    #     # item_name = self.masks_frame.iloc[idx, 0]
    #     images = []
    #     masks = []
    #     id_depth = self.masks_frame.to_dict('list') # {'id': dis, 'z': depths}
    #     id_depth['z'] = torch.Tensor(id_depth['z'])
    #
    #     z_mean = id_depth['z'].mean()
    #     z_std = id_depth['z'].std()
    #     image_mean = 0.0
    #     image_std = 0.0
    #     mask_mean = 0.0
    #     mask_std = 0.0
    #
    #     i = 0
    #
    #     id_list = []
    #     for id in ids:
    #         id_list.append(id)
    #
    #         image = self.get_transformed_image_by_id(id)
    #
    #         if self.transform:
    #             image = self.transform['image'](image)
    #
    #         # mask_name = os.path.join(self.mask_dir, id + self.mask_suffix)
    #         mask = self.get_transformed_mask_by_id(id)
    #
    #
    #         if self.transform:
    #             mask = self.transform['mask'](mask)
    #
    #         # depth = self.masks_frame.iloc[id, 1].astype('float')
    #         # switch from io to PIL
    #         # image = io.imread(img_name)
    #         # masks = self.masks_frame.iloc[idx, 1:].as_matrix()
    #         # WARNING: I don't know if I keep reshape
    #         # masks = masks.astype('float').reshape(-1, 2)
    #         images.append(image)
    #         image_mean = image_mean + image.mean()
    #         image_std = np.math.sqrt(image_std ** 2 + image.mean() ** 2)
    #         # depths.append(depth)
    #         masks.append(mask)
    #         mask_mean = mask_mean + mask.mean()
    #         mask_std = np.math.sqrt(mask_std ** 2 + mask.mean() ** 2)
    #
    #         i=i+1
    #
    #     # print (images)
    #     # image_mask = {'image': images, 'mask': masks}
    #
    #     id_depth['image'] = images
    #     id_depth['mask'] = masks
    #     id_depth['id'] = id_list # use image id instead of id from .cvs because of image augmentation
    #
    #     self.sample = id_depth
    #
    #     self.means = {'id': None, 'z': z_mean/i, 'image': image_mean/i, 'mask': mask_mean/i}
    #     self.stds = {'id': None, 'z': z_std, 'image': image_std, 'mask': mask_std}
    #
    #     return self.sample

    # implemented in the __getitem__()
    # def all_transform(self, sample):
    #     id, z, image, mask = sample.items()
    #     # # z cannot be transformed using pytorch.vision
    #     # z = np.asarray(z[1], dtype='float32')
    #     # z = self.transform['depth'](z)
    #     z = z[1]
    #     image = self.transform['image'](image[1])
    #     mask = self.transform['mask'](mask[1])
    #
    #     return {'id': id, 'z': z, 'image': image, 'mask': mask}
