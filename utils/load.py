#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import pandas as pd
import numpy as np
from PIL import Image

from .utils import resize_and_crop, normalize, hwc_to_chw, rle_encode

def get_ids(dir):
    """Returns a list of the ids in the directory"""
    # return (f[:-4] for f in os.listdir(dir))
    return (f[:] for f in os.listdir(dir))


# for each id, get (id, sub-id)
def split_ids_for_augmentation(ids, n=1):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i) for i in range(n) for id in ids)

# scale = resize to what percent
def augmentation(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        # yield get_square(im, pos)
        yield im

def get_imgs_depths_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = augmentation(ids, dir_img, '.png', scale)
    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    csv_depths = pd.read_csv('../data/depths.csv', index_col='id')
    depths = csv_depths.loc[dir_img, 'z']

    masks = augmentation(ids, dir_mask, '.png', scale)
    masks_rle_encoded = map(rle_encode, masks)

    return zip(imgs_normalized, depths, masks_rle_encoded)

# def get_full_img_and_mask(id, dir_img, dir_mask):
#     im = Image.open(dir_img + id + '.jpg')
#     mask = Image.open(dir_mask + id + '_mask.gif')
#     return np.array(im), np.array(mask)
