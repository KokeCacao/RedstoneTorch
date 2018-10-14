import os
import sys
import numpy as np
from datetime import datetime

from PIL import Image

import torch

from optparse import OptionParser

from torchvision.transforms import transforms

import config
from predict import predict
from unet.unet_model import UNetResNet
from unet import UNet


def submit(net, gpu):
    """Used for Kaggle submission: predicts and encode all test images"""
    directory_list = os.listdir(config.DIRECTORY_TEST)
    with open(config.DIRECTORY_TEST + "SUBMISSION" + config.PREDICTION_TAG + ".csv", 'a') as f:
        f.write('img,rle_mask\n')
        for index, img_name in enumerate(directory_list):
            print('{} --- {}/{}'.format(img_name, index, len(directory_list)))

            img = Image.open(config.DIRECTORY_TEST + img_name).convert('RGB')
            img = config.PREDICT_TRANSFORM(img)

            mask_pred = predict(net, img, gpu)
            masks_pred_pil = config.PREDICT_TRANSFORM(tensor_to_PIL(mask_pred))
            masks_pred_np = np.array(masks_pred_pil)

            enc = rle_encode(masks_pred_np)
            f.write('{},{}\n'.format(img_name, ' '.join(map(str, enc))))
            masks_pred_pil.save(config.DIRECTORY_TEST + "predicted/" + img_name + ".png")

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    if image.size()[0] == 1: image = image.repeat(3, 1, 1) # from gray sacale to RGB
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image

def get_args():
    parser = OptionParser()
    parser.add_option('-g', '--gpu', dest='gpu', default="", help='use cuda, please put all gpu id here')
    parser.add_option('-c', '--load', dest='load', default=False, help='load file model')
    parser.add_option('-p', '--dir_prefix', dest='dir_prefix', default="", help='the root directory')
    parser.add_option('-t', '--tag', dest='tag', default="", help='tag for tensorboard-log')

    (options, args) = parser.parse_args()
    return options


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
# def rle_encode(mask_image):
#     pixels = mask_image.flatten()
#     # We avoid issues with '1' at the start or end (at the corners of
#     # the original image) by setting those pixels to '0' explicitly.
#     # We do not expect these to be non-zero for an accurate mask,
#     # so this should not harm the score.
#     pixels[0] = 0
#     pixels[-1] = 0
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
#     runs[1::2] = runs[1::2] - runs[:-1:2]
#     return runs

def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


if __name__ == '__main__':
    args = get_args()
    if args.load != False:
        config.TRAIN_LOAD = args.load
        config.PREDICTION_LOAD_TAG = config.TRAIN_LOAD.replace("/", "-").replace(".pth", "-")
    if args.tag != "": config.PREDICTION_TAG = str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-") + "-" + args.tag

    print("Current Directory: " + str(os.getcwd()))
    print("====================================")
    print("Loading Neuronetwork...")
    net = UNetResNet(encoder_depth=50, num_classes=1, num_filters=32, dropout_2d=0.2,
                 pretrained=True, is_deconv=True)
    if config.TRAIN_GPU != "": net = torch.nn.DataParallel(net, device_ids=[int(i) for i in config.TRAIN_GPU.split(",")])

    if config.TRAIN_LOAD:
        net.load_state_dict(torch.load(config.TRAIN_LOAD))
        print('Model loaded from {}'.format(config.TRAIN_LOAD))

    torch.manual_seed(config.TRAIN_SEED)
    if config.TRAIN_GPU != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = config.TRAIN_GPU
        print('Using GPU: [' + config.TRAIN_GPU + ']')
        torch.cuda.manual_seed_all(config.TRAIN_SEED)
        net.cuda()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        submit(net=net, gpu=config.TRAIN_GPU)
    except KeyboardInterrupt as e:
        print(e)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
# t1 = time.time()
# pred_dict = {idx: rle_encode(      np.round(      downsample(preds_test[i]) > threshold_best    )      ) for i, idx in enumerate(tqdm_notebook(test_df.index.values))}
# t2 = time.time()
#
# print(f"Usedtime = {t2-t1} s")

"""
python submit --load --tag

"""