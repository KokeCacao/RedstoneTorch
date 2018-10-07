import os
import pandas as pd
import numpy as np

import torch
from torch import random
from PIL import Image
from skimage import io
from torch.utils import data
from torch.utils.data import SubsetRandomSampler


class TGSData(data.Dataset):
    # def __init__(self, root, train=True, transform=None, mask_transform=None):
    #     self.root = root
    #     self.transform = transform
    #     self.mask_transform = mask_transform
    #     self.train = train  # training set or test set
    #
    #     if not self._check_exists():
    #         raise RuntimeError('Dataset not found.' +
    #                            ' You can use download=True to download it')
    #
    #     if self.train:
    #         self.train_data, self.train_labels = torch.load(
    #             os.path.join(root, self.processed_folder, self.training_file))
    #     else:
    #         self.test_data, self.test_labels = torch.load(os.path.join(root, self.processed_folder, self.test_file))
    #
    # def __getitem__(self, index):
    #     if self.train:
    #         img, mask = self.train_data[index], self.train_labels[index]
    #     else:
    #         img, mask = self.test_data[index], self.test_labels[index]
    #
    #     # doing this so that it is consistent with all other datasets
    #     # to return a PIL Image
    #     img = Image.fromarray(img.numpy(), mode='L')
    #
    #     if self.transform is not None:
    #         img = self.transform(img)
    #
    #     if self.mask_transform is not None:
    #         mask = self.mask_transform(mask)
    #
    #     return img, mask
    #
    # def __len__(self):
    #     return 0;

    def __init__(self, csv_dir, img_dir, mask_dir, img_suffix=".png", mask_suffix=".png", transform=None):
        print("Reading Data...")
        self.masks_frame = pd.read_csv(csv_dir)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform

        self.data_len = len(self.masks_frame)
        self.train_len = 0
        self.val_len = 0

        self.sample = None

        self.means = None
        self.stds = None

    def __len__(self):
        return self.data_len

    def get_img_names(self):
        return (f[:-len(self.img_suffix)].replace("images_original_","").replace("_groundtruth_(1)_images_","") for f in os.listdir(self.img_dir))
        # return (f[:].replace(self.img_suffix, "", 1) for f in os.listdir(self.img_dir))

    def get_data(self):
        if (self.sample == None): print("WARNING: Tyring to load empty data!")
        return self.sample

    def get_sampler(self, all_id, data_percent=1.0, val_percent=0.05, data_shuffle = False, train_shuffle=True, val_shuffle=False, seed=19):
        self.sample = self.get_all_sample(all_id)
        print ("Loading sample:")
        print ("Sample size: {} samples".format(self.data_len))

        print ("Sample mean of depth: {}".format(self.means.get("z").float()))
        print ("Sample mean of image: {}".format(self.means.get("image").float()))
        print ("Sample mean of masks: {}".format(self.means.get("mask").float()))
        print ("Sample std of depth: {}".format(self.stds.get("z").float()))
        print ("Sample std of image: {}".format(self.stds.get("image")))
        print ("Sample std of masks: {}".format(self.stds.get("mask")))
        # print ("Data Structure of Sample: {'id': id(size), 'z': z(size), 'image': image(size, 225, 225), 'mask': mask(size, 225, 225)}")

        # ABANDONED: get indice
        # train_size = int(val_percent * len(self.sample))
        # test_size = len(self.sample) - train_size
        # train_dataset, test_dataset = torch.utils.data.random_split(self.sample, [train_size, test_size])

        # get indice
        dataset_size = self.data_len
        print("Total Size:", dataset_size)
        indices = list(range(dataset_size))

        if data_shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)

        val_split = int(np.floor(data_percent * val_percent * dataset_size))
        print("Validation Size: {}".format(val_split))
        self.val_len = val_split
        data_split = int(np.floor(data_percent * dataset_size))
        print("Traning Size: {}".format(data_split-val_split))
        self.train_len = data_split-val_split

        val_indices = indices[:val_split]
        if val_shuffle:
            np.random.seed(seed+1)
            np.random.shuffle(val_indices)
            np.random.seed(seed)
        train_indices = indices[val_split:data_split]
        if train_shuffle:
            np.random.seed(seed+2)
            np.random.shuffle(train_indices)
            np.random.seed(seed)

        self.tran_len = len(train_indices)
        self.val_len = len(val_indices)

        train_sampler = SubsetRandomSampler(train_indices)
        validation_sampler = SubsetRandomSampler(val_indices)


        return train_sampler, validation_sampler

    def __getitem__(self, index):
        print ("loading new data for the batch...")
        z = self.sample['z'][index]
        image = self.sample['image'][index]
        mask = self.sample['mask'][index]
        return ((z, image), mask)

    def get_image_by_id(self, id):
        return Image.open(os.path.join(self.img_dir, "images_original_" + id + self.img_suffix)).convert('RGB')

    def get_mask_by_id(self, id):
        return Image.open(os.path.join(self.img_dir, "_groundtruth_(1)_images_" + id + self.mask_suffix))

    def get_all_sample(self, ids, seed=19):
        random.manual_seed(seed)
        # item_name = self.masks_frame.iloc[idx, 0]
        images = []
        masks = []
        id_depth = self.masks_frame.to_dict('list') # {'id': dis, 'z': depths}
        id_depth['z'] = torch.Tensor(id_depth['z'])

        z_mean = id_depth['z'].mean()
        z_std = id_depth['z'].std()
        image_mean = 0.0
        image_std = 0.0
        mask_mean = 0.0
        mask_std = 0.0

        i = 0

        for id in ids:
            image = self.get_image_by_id(id)

            if self.transform:
                image = self.transform['image'](image)

            # mask_name = os.path.join(self.mask_dir, id + self.mask_suffix)
            mask = self.get_mask_by_id(id)


            if self.transform:
                mask = self.transform['mask'](mask)

            # depth = self.masks_frame.iloc[id, 1].astype('float')
            # switch from io to PIL
            # image = io.imread(img_name)
            # masks = self.masks_frame.iloc[idx, 1:].as_matrix()
            # WARNING: I don't know if I keep reshape
            # masks = masks.astype('float').reshape(-1, 2)
            images.append(image)
            image_mean = image_mean + image.mean()
            image_std = np.math.sqrt(image_std ** 2 + image.mean() ** 2)
            # depths.append(depth)
            masks.append(mask)
            mask_mean = mask_mean + mask.mean()
            mask_std = np.math.sqrt(mask_std ** 2 + mask.mean() ** 2)

            i=i+1

        # print (images)
        # image_mask = {'image': images, 'mask': masks}

        id_depth['image'] = images
        id_depth['mask'] = masks

        self.sample = id_depth

        self.means = {'id': None, 'z': z_mean/i, 'image': image_mean/i, 'mask': mask_mean/i}
        self.stds = {'id': None, 'z': z_std, 'image': image_std, 'mask': mask_std}

        return self.sample

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

