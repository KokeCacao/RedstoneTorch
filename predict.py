import os
import sys
import matplotlib as mpl
import torch
import config
import numpy as np

from tqdm import tqdm
from PIL import Image

import utils.encode
from utils.encode import rle_encode

if os.environ.get('DISPLAY', '') == '':
    print('WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.')
    mpl.use('Agg')
from matplotlib import pyplot as plt

from optparse import OptionParser
from model.resunet.resunet_model import UNetResNet
from tensorboardX import SummaryWriter

def predict(net, image):
    if config.TRAIN_GPU_ARG: image = image.cuda()

    if image.mean() < config.PREDICTION_DARK_THRESHOLD:
        """WARNING: Encounter Dark Image"""
        return torch.zeros((image.size()[0], 1, image.size()[2], image.size()[3])).cuda()
    """Need to repeat three times because the net will automatically reduce C when the Cs are the same"""
    masks_pred = net(image)

    del image
    if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
    return masks_pred


def submit(net, writer):
    torch.no_grad()
    """Used for Kaggle submission: predicts and encode all test images"""
    directory_list = os.listdir(config.DIRECTORY_TEST)
    if not os.path.exists(config.DIRECTORY_TEST + "predicted/" + config.PREDICTION_TAG):
        os.makedirs(config.DIRECTORY_TEST + "predicted/" + config.PREDICTION_TAG)
    if os.path.exists(config.DIRECTORY_TEST + "predicted/" + config.PREDICTION_TAG + ".csv"):
        os.remove(config.DIRECTORY_TEST + "predicted/" + config.PREDICTION_TAG + ".csv")
        print("WARNING: delete file '{}'".format(config.DIRECTORY_TEST + "predicted/" + config.PREDICTION_TAG + ".csv"))
    with open(config.DIRECTORY_TEST + "predicted/" + config.PREDICTION_TAG + ".csv", 'a') as f:
        f.write('id,rle_mask\n')
        for index, img_name in enumerate(tqdm(directory_list)):
            if config.DIRECTORY_SUFFIX_IMG not in img_name: continue

            img = Image.open(config.DIRECTORY_TEST + img_name).convert('RGB')
            img_n = config.PREDICT_TRANSFORM_IMG(img).unsqueeze(0)  # add N

            mask_pred = predict(net, img_n).squeeze(0)  # reduce N
            """if config.TRAIN_GPU: """
            masks_pred_pil = config.PREDICT_TRANSFORM_BACK(mask_pred)  # return gray scale PIL
            masks_pred_np = np.where(np.asarray(masks_pred_pil, order="F") > config.EVAL_CHOSEN_THRESHOLD, 1, 0)  # return tensor with (H, W) - proved

            enc = rle_encode(masks_pred_np)
            f.write('{},{}\n'.format(img_name.replace(config.DIRECTORY_SUFFIX_MASK, ""), enc))

            if index % 100 == 0:
                F = plt.figure()
                plt.subplot(221)
                plt.imshow(img)
                plt.title("Image_Real")
                plt.grid(False)
                plt.subplot(222)
                plt.imshow(masks_pred_pil)
                plt.title("Result")
                plt.grid(False)
                plt.subplot(223)
                plt.imshow(utils.encode.tensor_to_PIL(mask_pred))
                plt.title("Predicted")
                plt.grid(False)
                plt.subplot(224)
                plt.imshow(Image.fromarray(np.where(masks_pred_np > config.EVAL_CHOSEN_THRESHOLD, 255, 0), mode="L"))
                plt.title("Encoded")
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

def load_checkpoint(net, load_path):
    if load_path and os.path.isfile(load_path):
        print("=> Loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        if 'state_dict' not in checkpoint:
            net.load_state_dict(checkpoint)
            print("=> Loaded only the model")
            return
        config.epoch = checkpoint['epoch']
        config.global_step = checkpoint['global_step']
        net.load_state_dict(checkpoint['state_dict'])
        print("=> Loaded checkpoint 'epoch = {}' (global_step = {})".format(config.epoch, config.global_step))
    else:
        print("=> Nothing loaded")


def load_args():
    args = get_args()
    config.TRAIN_LOAD = args.load
    config.PREDICTION_LOAD_TAG = config.TRAIN_LOAD.replace("/", "-").replace(".pth", "-")
    config.PREDICTION_TAG = config.TRAIN_LOAD.replace("tensorboard-", "").replace("-checkpoints-", "-").replace(".pth", "").replace("tensorboard/", "").replace("/checkpoints/", "-").replace(".pth", "")
    if args.tag != "": config.PREDICTION_TAG = config.PREDICTION_TAG + "-" + args.tag


if __name__ == '__main__':
    load_args()
    writer = SummaryWriter("tensorboard/" + config.TRAIN_TAG)
    print("=> Tensorboard: " + "python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/tensorboard/" + config.TRAIN_TAG + " --port=6006")
    print("=> Current Directory: " + str(os.getcwd()))
    print("=> Download Model here: " + "ResUnet/" + config.DIRECTORY_TEST + "predicted/" + config.PREDICTION_TAG + ".csv")

    net = UNetResNet(encoder_depth=50, num_classes=1, num_filters=32, dropout_2d=0.2,
                     pretrained=True, is_deconv=True)
    if config.TRAIN_GPU_ARG != "": net = torch.nn.DataParallel(net, device_ids=[int(i) for i in config.TRAIN_GPU_ARG.split(",")])

    load_checkpoint(net, config.TRAIN_LOAD)

    if config.TRAIN_GPU_ARG != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = config.TRAIN_GPU_ARG  # default
        print('=> Using GPU: [' + config.TRAIN_GPU_ARG + ']')
        net.cuda()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        submit(net=net, writer=writer)
    except KeyboardInterrupt as e:
        print(e)
        writer.close()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

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

python submit.py --load tensorboard/2018-10-17-19-47-01-207026-wednesday-eve/checkpoints/CP7.pth --tag 'submit5'
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/data/test
/images/predicted/tensorboard --port=6006
ResUnet/data/test/images/predicted/2018-10-17-19-47-01-207026-wednesday-eve-CP7-submita.csv

python submit.py --load tensorboard/2018-10-17-19-47-01-207026-wednesday-eve/checkpoints/CP73.pth



python submit.py --load tensorboard/2018-10-19-04-42-36-919742-thursday-a/checkpoints/thursday-a-CP0.pth
"""
