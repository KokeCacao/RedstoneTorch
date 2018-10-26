import numpy as np
import pydensecrf.densecrf as dcrf
from torchvision.transforms import transforms


def dense_crf(img, output_probs):
    h = output_probs.shape[0]
    w = output_probs.shape[1]

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    U = -np.log(output_probs)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=20, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)

    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q

def rle_encode(img):
    if len(img.shape) != 2 or img.shape[0] == 1:
        print("WARNING: The Image shape is {}, expected (H, W).".format(img.shape))
    pixels = img.flatten(order='F')
    if 255 in img:
        print("WARNING: The Image Start with non-binary value. Expected 0 or 1, got {}.".format(pixels[0]))
        print("Here is an example of the image: {}".format(img))
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return ' '.join(str(x) for x in runs)

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes, dtype=np.float)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])

def inverse_to_tensor(tensor):
    return [x * 255 for x in tensor.numpy()]


def tensor_to_PIL(tensor):
    """

    :param tensor: tensor with shape of (1, 1, 1) or (3, 1, 1)
    :return: PIL image with shape of (3, 1, 1)
    """
    image = tensor.cpu().clone()
    if image.size()[0] == 1: image = image.repeat(3, 1, 1)  # from gray sacale to RGB
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image