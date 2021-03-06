import numpy as np
# import pydensecrf.densecrf as dcrf
from PIL import Image
from torchvision.transforms import transforms


# def dense_crf(img, output_probs):
#     h = output_probs.shape[0]
#     w = output_probs.shape[1]
#
#     output_probs = np.expand_dims(output_probs, 0)
#     output_probs = np.append(1 - output_probs, output_probs, axis=0)
#
#     d = dcrf.DenseCRF2D(w, h, 2)
#     U = -np.log(output_probs)
#     U = U.reshape((2, -1))
#     U = np.ascontiguousarray(U)
#     img = np.ascontiguousarray(img)
#
#     d.setUnaryEnergy(U)
#
#     d.addPairwiseGaussian(sxy=20, compat=3)
#     d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)
#
#     Q = d.inference(5)
#     Q = np.argmax(np.array(Q), axis=0).reshape((h, w))
#
#     return Q


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

def mask2rle(img, width, height, max_color=1):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == max_color:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel+=1

    return " ".join(rle)

def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    """WARNING: This function should only be used in SIIM dataset because it constains .T() transformation here"""
    if rle == '-1': return mask.reshape(width, height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]
    """WARNING: This function should only be used in SIIM dataset because it constains .T() transformation here"""
    return mask.reshape(width, height)

# def get_one_hot(targets, nb_classes):
#     res = np.eye(nb_classes, dtype=np.float)[np.array(targets).reshape(-1)]
#     return res.reshape(list(targets.shape) + [nb_classes])

def to_numpy(tensor):
    return 225 * tensor.numpy()


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


def tensor_to_np_four_channel_transarant(image):
    """

    :param tensor: tensor with channel of (r, g, b, y), shape of (4, W, H)
    :return: transparant image with mask in tensor[1], the output will put the cannel layer the last layer
    """
    # red = tensor[0]
    # green = tensor[1]
    # blue = tensor[2]
    # yellow = tensor[3]
    ndarray = 1.0 * np.array([image[0], image[2], image[3], image[1]])
    return ndarray.transpose((1, 2, 0))


def tensor_to_np_four_channel_drop(image):
    """

    :param tensor: tensor with channel of (r, g, b, y), shape of (4, W, H)
    :return: drop tensor[1], the output will put the cannel layer the last layer
    """
    ndarray = 1.0 * np.array([image[0], image[2], image[3]])
    return ndarray.transpose((1, 2, 0))


def tensor_to_np_three_channel_with_green(image):
    """

    :param tensor: tensor with channel of (r, g, b, y), shape of (4, W, H)
    :return: drop tensor[1], the output will put the cannel layer the last layer
    """
    ndarray = 1.25 * np.array([0.5 * image[0] + 0.25 * image[3], 0.5 * image[1] + 0.25 * image[3], 0.5 * image[2]])
    return ndarray.transpose((1, 2, 0))


def tensor_to_np_three_channel_without_green(image):
    """

    :param tensor: tensor with channel of (r, g, b, y), shape of (4, W, H)
    :return: drop tensor[1], the output will put the cannel layer the last layer
    """
    ndarray = 1.25 * np.array([0.5 * image[0] + 0.25 * image[3], 0.25 * image[3], 0.5 * image[2]])
    return ndarray.transpose((1, 2, 0))


def ndarray_to_PIL(ndarray):
    return Image.fromarray(ndarray.astype('uint8'), 'RGB')


def save_as_npy(from_dir, to_dir):
    img = Image.open(from_dir)
    data = np.array(img, dtype='uint8')
    np.save(to_dir, data)


def np_three_channel_with_green(image, shape, green_intensity=1, other_intensity=1):
    """

    :param image: ndarray with channel of (r, g, b, y), shape of (W, H, 4)
    :return: np_three_channel_with_green for mpl display
    """
    image = image.transpose((2, 0, 1))
    img = np.zeros(shape)
    img[0] = other_intensity*(0.5*image[0] + 0.25*image[3])/0.75
    img[1] = (green_intensity*0.5*image[1] + 0*image[3])/0.75
    img[2] = other_intensity*image[2]
    return np.stack(img/255., axis=-1)