import random

from albumentations import DualTransform
from albumentations.augmentations import functional as F
import cv2

class DoNothing(DualTransform):
    """Do Nothing

    Args:
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask

    Image types:
        uint8, float32

    """
    def __init__(self, always_apply=False, p=1.0):
        super(DoNothing, self).__init__(always_apply, p)

    def apply(self, img, **params):
        return img

class AdaptivePadIfNeeded(DualTransform):
    """Pad the smallest side to the biggest side

    Args:
        p (float): probability of applying the transform. Default: 1.0.
        value (list of ints [r, g, b]): padding value if border_mode is cv2.BORDER_CONSTANT.

    Targets:
        image, mask

    Image types:
        uint8, float32

    """

    def __init__(self, border_mode=cv2.BORDER_REFLECT_101,
                 value=[0, 0, 0], always_apply=False, p=1.0):
        super(AdaptivePadIfNeeded, self).__init__(always_apply, p)
        self.border_mode = border_mode
        self.value = value

    def apply(self, img, **params):
        size = max(img.shape)
        return F.pad(img, min_height=size, min_width=size,
                     border_mode=self.border_mode, value=self.value)

class RandomPercentCrop(DualTransform):
    """Crop a random part of the input.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, percent_height=0.8, percent_width=0.8, always_apply=False, p=1.0):
        super(RandomPercentCrop, self).__init__(always_apply, p)
        self.percent_height = percent_height
        self.percent_width = percent_width

    def apply(self, img, h_start=0, w_start=0, **params):
        height, width = img.shape[:2]
        return F.random_crop(img, int(self.percent_height*float(height)), int(self.percent_width*float(width)), h_start, w_start)

    def get_params(self):
        return {'h_start': random.random(),
                'w_start': random.random()}