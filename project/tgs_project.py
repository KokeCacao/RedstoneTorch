import os
import sys
from datetime import datetime

import torch
from torch.utils import data as data

import config
from dataset.tgs import TGSData
from loss import loss as L
from loss.iou import iou_score
from model.resunet.resunet_model import UNetResNet
from train import writer, save_checkpoint, load_checkpoint
from utils.load import cuda

import matplotlib as mpl
import numpy as np
import operator
from torch.autograd import Variable

from utils.encode import tensor_to_PIL

if os.environ.get('DISPLAY', '') == '':
    print('WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.')
    mpl.use('Agg')
from matplotlib import pyplot as plt

class TGSProject():
    def __init__(self):
        net = UNetResNet(encoder_depth=50, num_classes=1, num_filters=32, dropout_2d=0.2,
                         pretrained=True, is_deconv=True)  # don't init weights, don't give depth
        if config.TRAIN_GPU_ARG: net = torch.nn.DataParallel(net, device_ids=config.TRAIN_GPU_LIST)

        self.optimizer = torch.optim.Adam(params=net.parameters(), lr=config.MODEL_LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.MODEL_WEIGHT_DEFAY)  # all parameter learnable
        load_checkpoint(net, self.optimizer, config.DIRECTORY_LOAD)
        self.net = cuda(net)




    def run(self):
        try:
            self.train(net=self.net,
                       optimizer=self.optimizer,
                       epochs=config.MODEL_EPOCHS,
                       batch_size=config.MODEL_BATCH_SIZE,
                       val_percent=config.TRAIN_VAL_PERCENT,
                       gpu=config.TRAIN_GPU_ARG,
                       data_percent=config.TRAIN_DATA_PERCENT
                       )
        except KeyboardInterrupt as e:
            print(e)
            writer.close()
            save_checkpoint(self.net.state_dict(), self.optimizer.state_dict(), interupt=True)
            print("To Resume: python train.py --tag 'default' --load " + config.DIRECTORY_CHECKPOINT + "INTERUPT-" + config.versiontag + "-" + config.DIRECTORY_CP_NAME.format(config.epoch))
            print("Or: python train.py --tag 'default' --load " + config.DIRECTORY_CHECKPOINT + config.versiontag + "-" + config.DIRECTORY_CP_NAME.format(config.epoch - 1))
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    def train(self, net,
              optimizer,
              epochs,
              batch_size,
              val_percent,
              gpu,
              data_percent
              ):
        tgs_data = TGSData(config.DIRECTORY_CSV, config.DIRECTORY_IMG, config.DIRECTORY_MASK, config.DIRECTORY_SUFFIX_IMG, config.DIRECTORY_SUFFIX_MASK)

        train_sampler, validation_sampler = tgs_data.get_split_sampler(data_percent=data_percent, val_percent=val_percent)

        train_loader = data.DataLoader(tgs_data, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=config.TRAIN_NUM_WORKER)
        validation_loader = data.DataLoader(tgs_data, batch_size=batch_size, sampler=validation_sampler, shuffle=False, num_workers=config.TRAIN_NUM_WORKER)

        # print('''
        # Starting training:
        #     Epochs: {}
        #     Batch size: {}
        #     Learning rate: {}
        #     Training size: {}
        #     Validation size: {}
        #     Checkpoints: {}
        #     CUDA: {}
        #     Momentum: {}
        #     Weight_decay: {}
        # '''.format(epochs, batch_size, lr, tgs_data.train_len, tgs_data.val_len, str(save_cp), str(gpu), momentum, weight_decay))

        # optimizer = optim.SGD(net.parameters(),
        #                       lr=lr,
        #                       momentum=momentum,
        #                       weight_decay=weight_decay)
        # optimizer = torch.optim.Adam(params=[
        #             # {'params': net.parameters()},
        #             # {'params': net.module.dropout_2d},
        #             # {'params': net.module.pool},
        #             # {'params': net.module.relu},
        #             {'params': net.module.conv1.parameters(), 'lr': 0.0001},
        #             {'params': net.module.conv2.parameters(), 'lr': 0.0004},
        #             {'params': net.module.conv3.parameters(), 'lr': 0.0006},
        #             {'params': net.module.conv4.parameters(), 'lr': 0.0008},
        #             {'params': net.module.conv5.parameters(), 'lr': 0.0009},
        #             {'params': net.module.center.parameters(), 'lr': 0.001},
        #             {'params': net.module.dec5.parameters(), 'lr': 1e-3},
        #             {'params': net.module.dec4.parameters(), 'lr': 1e-3},
        #             {'params': net.module.dec3.parameters(), 'lr': 1e-3},
        #             {'params': net.module.dec2.parameters(), 'lr': 1e-3},
        #             {'params': net.module.dec1.parameters(), 'lr': 1e-3},
        #             {'params': net.module.dec0.parameters(), 'lr': 1e-3},
        #             {'params': net.module.final.parameters(), 'lr': 0.0015}], lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay) # all parameter learnable

        train_begin = datetime.now()
        for epoch in range(epochs):
            epoch_begin = datetime.now()
            print('Starting epoch {}/{} - total of {}'.format(epoch + 1, epochs, config.epoch))

            epoch_loss = 0
            epoch_iou = 0

            # batch size should < 4000 due to the amount of data avaliable
            for batch_index, (id, z, image, true_mask, image_0, true_mask_0) in enumerate(train_loader, 0):

                config.global_step = config.global_step + 1

                if gpu != "":
                    # z = z.cuda()
                    image = image.cuda()
                    true_mask = true_mask.cuda()

                """
                Input: N, 1, H, W
                Output: N, 1, H, W
                """
                masks_pred = net(image)

                iou = iou_score(masks_pred, true_mask).mean()
                epoch_iou = epoch_iou + iou

                if epochs < 1e5:
                    loss = torch.nn.BCELoss()(torch.sigmoid(masks_pred).view(-1), true_mask.view(-1))
                else:
                    loss = L.lovasz_hinge(masks_pred, true_mask, ignore=None)

                epoch_loss += loss.item()

                now = datetime.now()
                train_duration = now - train_begin
                epoch_duration = now - epoch_begin
                print("SinceTrain:{}, Since Epoch:{}".format(train_duration, epoch_duration))
                print('{0}({8})# Epoch - {1:.6f}% ({2}/{3})batch ({4:}/{5:})data - TrainLoss: {6:.6f}, IOU: {7:.6f}'.format(epochs + 1,
                                                                                                                            (100 * (batch_index + 1.0) * batch_size) / tgs_data.train_len,
                                                                                                                            batch_index + 1,
                                                                                                                            tgs_data.train_len / batch_size,
                                                                                                                            (batch_index + 1) * batch_size,
                                                                                                                            tgs_data.train_len,
                                                                                                                            loss.item(),
                                                                                                                            iou, config.epoch))
                writer.add_scalars('loss/batch_training', {'Epoch': epochs + 1, 'TrainLoss': loss.item(), 'IOU': iou}, config.global_step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                del id, z, image, true_mask
                if gpu != "": torch.cuda.empty_cache()  # release gpu memory
            config.epoch = config.epoch + 1
            print('{}# Epoch finished ! Loss: {}, IOU: {}'.format(epochs + 1, epoch_loss / (batch_index + 1), epoch_iou / (batch_index + 1)))
            save_checkpoint(state_dict=net.state_dict(), optimizer_dict=optimizer.state_dict())
            # validation
            if config.TRAIN_GPU_ARG != "": torch.cuda.empty_cache()  # release gpu memory
            if config.TRAIN_VALIDATION:
                val_dice = eval_net(net, validation_loader, visualization=config.DISPLAY_VISUALIZATION, writer=writer, epoch_num=epochs + 1)
                print('Validation Dice Coeff: {}'.format(val_dice))
                writer.add_scalars('loss/epoch_validation', {'Validation': val_dice}, epochs + 1)
            if config.DISPLAY_HISTOGRAM:
                for i, (name, param) in enumerate(net.named_parameters()):
                    print("Calculating Histogram #{}".format(i))
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epochs + 1)
            if config.TRAIN_GPU_ARG != "": torch.cuda.empty_cache()  # release gpu memory




def eval_net(net, validation_loader, visualization, writer, epoch_num=0):
    thresold_dict = dict()
    """Evaluation without the densecrf with the dice coefficient"""
    # total_loss = 0
    total_ious = np.array([])

    for batch_index, (id, z, image, true_mask, image_0, true_mask_0) in enumerate(validation_loader, 0):

        if config.TRAIN_GPU_ARG:
            image = image.cuda()
            true_mask = true_mask.cuda()

        masks_pred = net(image)
        """return: shape(N, iou)"""
        ious = iou_score(masks_pred, true_mask, threshold=0.5)
        if config.EVAL_IF_THRESHOLD_TEST:
            for threshold in config.EVAL_TRY_THRESHOLD:
                iou_temp = iou_score(masks_pred, true_mask, threshold).mean()
                threshold_pre = thresold_dict.get(threshold)
                if threshold_pre != None:
                    threshold_pre = threshold_pre.append(iou_temp)
                else:
                    threshold_pre = [iou_temp]
                thresold_dict[threshold] = threshold_pre
        total_ious = np.concatenate((total_ious, np.array(ious).flatten()), axis=None)
        # iou = ious.mean().float()

        if visualization and batch_index == 0:
            writer.add_pr_curve("loss/epoch_validation_image", true_mask, masks_pred, global_step=epoch_num)
            for index, input_id in enumerate(id):
                F = plt.figure()

                plt.subplot(321)
                plt.imshow(tensor_to_PIL(image_0[index]))
                plt.title("Image_Real")
                plt.grid(False)

                plt.subplot(322)
                plt.imshow(tensor_to_PIL(image[index]))
                plt.title("Image_Trans")
                plt.grid(False)

                plt.subplot(323)
                plt.imshow(tensor_to_PIL(true_mask_0[index]))
                plt.title("Mask_Real")
                plt.grid(False)

                plt.subplot(324)
                plt.imshow(tensor_to_PIL(true_mask[index]))
                plt.title("Mask_Trans")
                plt.grid(False)

                # plt.subplot(325)
                # plt.imshow(ImageChops.difference(tensor_to_PIL(true_mask[index]), tensor_to_PIL(masks_pred[index])))
                # plt.title("Error: {}".format(ious[index]))
                # plt.grid(False)

                plt.subplot(325)
                if config.TRAIN_GPU_ARG:
                    plt.imshow(tensor_to_PIL((masks_pred[index] > Variable(torch.Tensor([config.EVAL_CHOSEN_THRESHOLD])).cuda()).float() * 1))
                else:
                    pass  # TODO
                plt.title("Error: {}".format(ious[index]))
                plt.grid(False)

                plt.subplot(326)
                plt.imshow(tensor_to_PIL(masks_pred[index]))
                plt.title("Predicted")
                plt.grid(False)
                writer.add_figure("image/epoch_validation/" + str(index), F, global_step=config.global_step)
        del id, z, image, true_mask
        if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()  # release gpu memory

    threshold_dict_mean = dict()
    for key, item in thresold_dict.items():
        item = np.mean(item)
        threshold_dict_mean[key] = item
        writer.add_scalars('val/threshold', {'Thresold': item}, key * 100)

    writer.add_scalars('val/max_threshold_val', {'MaxThresold': np.max(threshold_dict_mean.values())}, config.global_step)
    writer.add_scalars('val/max_threshold', {'MaxThresold': max(threshold_dict_mean.items(), key=operator.itemgetter(1))[0]}, config.global_step)

    writer.add_histogram("iou", total_ious, config.global_step)
    return total_ious.mean()



"""
Good Models

2018-10-07-23-40-34-439264-different-lr 21Epoch -> python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/tensorboard/2018-10-07-23-40-34-439264-different-lr --port=6006 -> IOU0.59, loss0.21, but no pattern
2018-10-08-23-24-27-715364-load-different-lr -> overfit

Don't augment image too much, but you can zoom in...
python train.py --epochs 300 --batch-size 32 --learning-rate 0.01 --dir_prefix '' --data_percent 1.00 --gpu "0,1" --visualization "True" --tag "fast-train" -> gray pictures

Adjust smaller batch size, and keep learning rate slower
python train.py --epochs 300 --batch-size 16 --learning-rate 0.005 --dir_prefix '' --data_percent 1.00 --gpu "0,1" --visualization "True" --tag "fast-train" -> First Epoch good, but bad after the first
python train.py --epochs 300 --batch-size 16 --learning-rate 0.001 --dir_prefix '' --data_percent 1.00 --gpu "0,1" --visualization "True" --tag "adjust-train" --load tensorboard/2018-10-10-02-14-05-405869-fast-train/checkpoints/CP1.pth
python train.py --epochs 300 --batch-size 16 --learning-rate 0.001 --dir_prefix '' --data_percent 1.00 --gpu "0,1" --visualization "True" --tag "adjust-train2" --load tensorboard/2018-10-10-03-02-43-871959-adjust-train/checkpoints/CP5.pth
python train.py --epochs 300 --batch-size 16 --learning-rate 0.001 --dir_prefix '' --data_percent 1.00 --gpu "0,1" --visualization "True" --tag "adjust-train3" --load tensorboard/2018-10-10-10-29-55-491693-adjust-train2/checkpoints/CP1.pth

Train the strange Model first using new images and 0.001 learning rate, with better showcase of error. Becareful that the model probably already see the validation data! But it is not great in terms of visualized prediction.
python train.py --epochs 300 --batch-size 16 --learning-rate 0.001 --dir_prefix '' --data_percent 1.00 --gpu "0,1" --visualization "True" --tag "adjust-train3" --load tensorboard/2018-10-07-23-40-34-439264-different-lr/checkpoints/CP21.pth

Now train more epoch of the second model -> overfit
python train.py --epochs 300 --batch-size 16 --learning-rate 0.001 --dir_prefix '' --data_percent 1.00 --gpu "0,1" --visualization "True" --tag "adjust-train4" --load tensorboard/2018-10-10-10-29-55-491693-adjust-train2/checkpoints/CP1.pth

reduce weight decay, decrease learning rate. The CP3.pth is good, others are overfitting
python train.py --epochs 300 --batch-size 16 --learning-rate 0.0008 --dir_prefix '' --data_percent 1.00 --gpu "0,1" --visualization "True" --tag "adjust-train5" --load tensorboard/2018-10-10-19-59-21-422178-adjust-train4/checkpoints/CP2.pth

Different Augmentation
python train.py --tag "diff-aug3" --load tensorboard/2018-10-13-13-41-28-633198-test-success/checkpoints/CP1.pth
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/tensorboard/2018-10-13-15-02-56-313421-diff-aug3 --port=6006
cp: tensorboard/2018-10-13-15-02-56-066021-test/checkpoints/CP21.pth

Try not change brightness of mask, add global step
python train.py --tag "success-music" --load tensorboard/2018-10-13-15-02-56-066021-test/checkpoints/CP21.pth
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/tensorboard/2018-10-13-18-46-22-184141-success-music --port=6006



====================================
python train.py --tag "success-music2" --load tensorboard/2018-10-13-18-46-21-934969-test/checkpoints/CP2.pth
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/tensorboard/2018-10-13-19-21-11-323191-success-music2 --port=6006
===================================
python train.py --tag "success-music3" --load tensorboard/2018-10-13-18-46-21-934969-test/checkpoints/CP2.pth
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/tensorboard/2018-10-13-19-53-02-991722-success-music3 --port=6006

python train.py --tag "success-submit5" --load tensorboard/2018-10-13-19-53-02-991722-success-music3/checkpoints/CP73.pth
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/tensorboard/2018-10-14-12-29-43-331445-success-submit5 --port=6006



python train.py --tag "new-day"
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/tensorboard/2018-10-14-14-05-57-388044-new-day2 --port=6006

# NEW EXPERIMENT
python train.py --tag "tuesday-night"
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/tensorboard/2018-10-17-00-00-33-668670-tuesday-night --port=6006

python train.py --tag "tuesday-night" --load tensorboard/2018-10-17-00-00-33-668670-tuesday-night/checkpoints/CP5.pth
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/tensorboard/2018-10-17-00-53-07-003683-tuesday-night --port=6006

python train.py --tag "wednesday-aft" --load tensorboard/2018-10-17-00-53-07-003683-tuesday-night/checkpoints/CP71.pth
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/tensorboard/2018-10-17-17-00-26-568369-wednesday-aft --port=6006

python train.py --tag "wednesday-eve" --load tensorboard/2018-10-17-17-00-26-568369-wednesday-aft/checkpoints/CP13.pth
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/tensorboard/2018-10-17-19-47-01-207026-wednesday-eve --port=6006

python train.py --tag "thursday-eve" --load tensorboard/2018-10-17-19-47-01-207026-wednesday-eve/checkpoints/CP73.pth
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/tensorboard/2018-10-17-19-47-01-207026-wednesday-eve --port=6006


python train.py --tag "thursday-final" --load tensorboard/2018-10-19-02-11-20-325481-thursday-eve/checkpoints/INTERUPT-CP0.pth
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/tensorboard/2018-10-19-02-11-20-325481-thursday-eve --port=6006 


python train.py --tag "test" --load tensorboard/2018-10-19-03-56-22-480073-thursday-final/checkpoints/INTERUPT-thursday-final-CP0.pth


g.mul_(beta1).add_(1 - beta1, grad)
RuntimeError: Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor for argument #4 'other'
k1412042720@ml-k80-3:~/ResUnet$ python train.py --tag "thursday-a" --load tensorboard/2018-10-17-19-47-01-207026-wednesday-eve/checkpoints/CP73.pth
WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.
=> Tensorboard: python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/tensorboard/2018-10-17-19-47-01-207026-wednesday-eve --port=6006

python train.py --tag "thursday-a" --load tensorboard/2018-10-19-04-09-09-838061-thursday-a/checkpoints/thursday-a-CP0.pth

"""

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
from tensorboardwriter import SummaryWriter

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


# def load_args():
#     args = get_args()
#     config.DIRECTORY_LOAD = args.load
#     config.PREDICTION_LOAD_TAG = config.DIRECTORY_LOAD.replace("/", "-").replace(".pth", "-")
#     config.PREDICTION_TAG = config.DIRECTORY_LOAD.replace("tensorboard-", "").replace("-checkpoints-", "-").replace(".pth", "").replace("tensorboard/", "").replace("/checkpoints/", "-").replace(".pth", "")
#     if args.tag: config.PREDICTION_TAG = config.PREDICTION_TAG + "-" + args.tag


if __name__ == '__main__':
    load_args()
    writer = SummaryWriter("tensorboard/" + config.PROJECT_TAG)
    print("=> Tensorboard: " + "python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/tensorboard/" + config.PROJECT_TAG + " --port=6006")
    print("=> Current Directory: " + str(os.getcwd()))
    print("=> Download Model here: " + "ResUnet/" + config.DIRECTORY_TEST + "predicted/" + config.PREDICTION_TAG + ".csv")

    net = UNetResNet(encoder_depth=50, num_classes=1, num_filters=32, dropout_2d=0.2,
                     pretrained=True, is_deconv=True)
    if config.TRAIN_GPU_ARG: net = torch.nn.DataParallel(net, device_ids=[int(i) for i in config.TRAIN_GPU_ARG.split(",")])

    load_checkpoint(net, config.DIRECTORY_LOAD)

    if config.TRAIN_GPU_ARG:
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
