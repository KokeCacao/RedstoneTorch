import numpy as np
from torch.utils.data import Sampler

# https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/101429
class BalanceClassSampler(Sampler):
    def __init__(self, indices, label, length=None, replace=True):
        self.indices = indices
        self.label = label
        self.replace = replace

        if length is None: length = len(indices)
        self.length = length
        print("Using BalanceClassSampler(length={}, replace={})".format(self.length, self.replace))

    def __iter__(self):
        # pos_index = np.where(self.label==1)[0]
        # neg_index = np.where(self.label==0)[0]

        # See this line: y = np.array(list([self.get_empty_by_indice(x), 0] for x in X))
        pos_index = np.where(self.label==1)[0]
        neg_index = np.where(self.label==0)[0]

        # print("There are {} pos, and {} neg".format(len(pos_index), len(neg_index)))
        # 15177 pos, and 10443 neg

        half = self.length//2 + 1
        pos = np.random.choice(pos_index, half, replace=self.replace)
        neg = np.random.choice(neg_index, half, replace=self.replace)
        l = np.stack([pos,neg]).T
        l = l.reshape(-1)
        l = l[:self.length]
        l = self.indices[l]
        return iter(l)

    def __len__(self):
        return self.length