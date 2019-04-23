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