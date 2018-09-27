import os

import torch
from PIL import Image
from torch.utils import data


class MNIST(data.Dataset):
    def __init__(self, root, train=True, transform=None, mask_transform=None):
        self.root = root
        self.transform = transform
        self.mask_transform = mask_transform
        self.train = train  # training set or test set

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(os.path.join(root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        if self.train:
            img, mask = self.train_data[index], self.train_labels[index]
        else:
            img, mask = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return img, mask

    def __len__(self):
        return 0;