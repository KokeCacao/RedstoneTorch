import os
import sys
import matplotlib as mpl
import numpy as np

from tqdm import tqdm
from PIL import Image

if os.environ.get('DISPLAY','') == '':
    print('WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.')
    mpl.use('Agg')
from matplotlib import pyplot as plt

import torch

from optparse import OptionParser

from torchvision.transforms import transforms

import config
from predict import predict
from unet.unet_model import UNetResNet
from tensorboardX import SummaryWriter


def submit(net, writer):
    torch.no_grad()
    """Used for Kaggle submission: predicts and encode all test images"""
    directory_list = os.listdir(config.DIRECTORY_TEST)
    if not os.path.exists(config.DIRECTORY_TEST + "predicted/" + config.PREDICTION_TAG):
        os.makedirs(config.DIRECTORY_TEST + "predicted/" + config.PREDICTION_TAG)
    if os.path.exists(config.DIRECTORY_TEST + "predicted/" + config.PREDICTION_TAG + ".csv"):
        os.remove(config.DIRECTORY_TEST + "predicted/" + config.PREDICTION_TAG + ".csv")
    with open(config.DIRECTORY_TEST + "predicted/" + config.PREDICTION_TAG + ".csv", 'a') as f:
        f.write('id,rle_mask\n')
        for index, img_name in enumerate(tqdm(directory_list)):
            if config.DIRECTORY_SUFFIX_IMG not in img_name: continue

            img = Image.open(config.DIRECTORY_TEST + img_name).convert('RGB')
            img_n = config.PREDICT_TRANSFORM_IMG(img).unsqueeze(0) # add N

            mask_pred = predict(net, img_n).squeeze(0) # reduce N
            """if config.TRAIN_GPU: """
            masks_pred_pil = config.PREDICT_TRANSFORM_BACK(mask_pred) # reduce C from 3 to 1
            masks_pred_np = np.array(masks_pred_pil)

            enc = rle_encode(masks_pred_np)
            f.write('{},{}\n'.format(img_name.replace(config.DIRECTORY_SUFFIX_MASK, ""), enc))

            if index % 100 == 0:
                F = plt.figure()
                plt.subplot(131)
                plt.imshow(img)
                plt.title("Image_Real")
                plt.grid(False)
                plt.subplot(132)
                plt.imshow(masks_pred_pil)
                plt.title("Result")
                plt.grid(False)
                plt.subplot(133)
                plt.imshow(config.tensor_to_PIL(mask_pred))
                plt.title("Predicted")
                plt.grid(False)
                writer.add_figure(config.PREDICTION_TAG + "/" + str(img_name), F, global_step=index)
            if config.PREDICTION_SAVE_IMG: masks_pred_pil.save(config.DIRECTORY_TEST + "predicted/" + config.PREDICTION_TAG + "/" + img_name)



def get_args():
    parser = OptionParser()
    parser.add_option('-g', '--gpu', dest='gpu', default="", help='use cuda, please put all gpu id here')
    parser.add_option('-c', '--load', dest='load', default=False, help='load file model')
    parser.add_option('-p', '--dir_prefix', dest='dir_prefix', default="", help='the root directory')
    parser.add_option('-t', '--tag', dest='tag', default="", help='tag for tensorboard-log')

    (options, args) = parser.parse_args()
    return options

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join(map(str, run_lengths))

def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate(([0], pixels, [0]))
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(map(str, runs))

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
#     return ' '.join(map(str, runs))

# # ref.: https://www.kaggle.com/stainsby/fast-tested-rle
# def rle_encode(img):
#     '''
#     img: numpy array, 1 - mask, 0 - background
#     Returns run length as string formated
#     '''
#     pixels = img.flatten(order = 'F')
#     pixels[0] = 0 # add to prevent bug
#     pixels[-1] = 0 # add to prevent bug
#     pixels = np.concatenate([[0], pixels, [0]])
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
#     runs[1::2] -= runs[::2]
#     return ' '.join(str(x) for x in runs)


if __name__ == '__main__':
    args = get_args()
    config.TRAIN_LOAD = args.load
    config.PREDICTION_LOAD_TAG = config.TRAIN_LOAD.replace("/", "-").replace(".pth", "-")
    config.PREDICTION_TAG = config.TRAIN_LOAD.replace("tensorboard-", "").replace("-checkpoints-", "-").replace(".pth", "").replace("tensorboard/", "").replace("/checkpoints/", "-").replace(".pth", "")
    if args.tag != "": config.PREDICTION_TAG = config.PREDICTION_TAG + "-" + args.tag
    writer = SummaryWriter(config.DIRECTORY_TEST + "predicted/tensorboard")
    print("Copy this line to command: " + "python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/" + config.DIRECTORY_TEST + "predicted/tensorboard" + " --port=6006")


    print("Current Directory: " + str(os.getcwd()))
    print("Download Model here: " + "ResUnet/" + config.DIRECTORY_TEST + "predicted/" + config.PREDICTION_TAG + ".csv")
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
        submit(net=net, writer=writer)
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
python submit.py --load tensorboard/2018-10-13-19-53-02-729361-test/checkpoints/CP36.pth --tag bronze-here
download: ResUnet/data/test/images/predicted/SUBMISSION-2018-10-14-11-53-06-571963-bronze-here.csv
ResUnet/data/test/images/predicted/2018-10-14-05-12-51-616453-bronze-here/78a68dece6.png


python submit.py --load tensorboard/2018-10-13-19-53-02-991722-success-music3/checkpoints/CP50.pth --tag second
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/tensorboard/2018-10-14-13-32-48-325330-train-predict --port=6006

download: ResUnet/data/test/images/predicted/SUBMISSION-2018-10-14-11-53-06-571963-bronze-here.csv
ResUnet/data/test/images/predicted/2018-10-14-05-12-51-616453-bronze-here/78a68dece6.png


python submit.py --load tensorboard/2018-10-17-17-00-26-568369-wednesday-aft/checkpoints/CP13.pth --tag 'submit'
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/data/test/images/predicted/tensorboard --port=6006
Download: ResUnet/data/test/images/predicted/2018-10-17-17-00-26-568369-wednesday-aft-CP13-submit.csv

python submit.py --load tensorboard/2018-10-17-19-47-01-207026-wednesday-eve/checkpoints/CP7.pth --tag 'submit'
Download: ResUnet/data/test/images/predicted/2018-10-17-19-47-01-207026-wednesday-eve-CP7-submit.csv

python submit.py --load tensorboard/2018-10-17-19-47-01-207026-wednesday-eve/checkpoints/CP7.pth --tag 'submit2'
"""