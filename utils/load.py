#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import pandas as pd
import numpy as np
import torchvision
from PIL import Image

from .utils import resize_and_crop, normalize, hwc_to_chw, rle_encode

def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:].replace(".png", "", 1) for f in os.listdir(dir))


# for each id, get (id, sub-id)
def split_ids_for_augmentation(ids, n=1):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i) for i in range(n) for id in ids)

# scale = resize to what percent
def augmentation_image(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        # yield get_square(im, pos)
        yield im

def augmentation_depth(ids, dir):
    """From a list of tuples, returns the correct cropped img"""
    csv_depths = pd.read_csv(dir, index_col='id')
    for id, pos in ids:
        depth = csv_depths.loc[id, 'z']
        # yield get_square(im, pos)
        yield depth

def get_imgs_depths_and_masks(ids, dir_img, dir_depth, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = augmentation_image(ids, dir_img, '.png', scale)
    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    depths = augmentation_depth(ids, dir_depth)

    masks = augmentation_image(ids, dir_mask, '.png', scale)
    # masks_rle_encoded = map(rle_encode, masks)

    return zip(imgs_normalized, depths, masks)

# def get_full_img_and_mask(id, dir_img, dir_mask):
#     im = Image.open(dir_img + id + '.jpg')
#     mask = Image.open(dir_mask + id + '_mask.gif')
#     return np.array(im), np.array(mask)
