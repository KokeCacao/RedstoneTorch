import sys
import os
from optparse import OptionParser

import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
from torch import optim
from torchvision import transforms

from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids_for_augmentation, split_train_val, get_imgs_depths_and_masks, batch

# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize(224),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

dir_prefix = 'drive/My Drive/ML/Pytorch-UNet/'
dir_img = dir_prefix + 'data/train/images/'
dir_mask = dir_prefix + 'data/train/masks/'
dir_depth = dir_prefix + 'data/depths.csv'
dir_checkpoint = dir_prefix + 'checkpoints/'
validation = True


def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=96,
              weight_init=0.01):


    # get (id, sub-id)
    ids = get_ids(dir_img)
    ids = split_ids_for_augmentation(ids, 2)

    # iddataset['train'] are ids of tranning data
    # iddataset['val'] are ids of validation data
    iddataset = split_train_val(ids, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
        Weight_init: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu), weight_init))

    N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = torch.nn.BCELoss()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        # reset the generators(augmentation)
        train = get_imgs_depths_and_masks(iddataset['train'], dir_img, dir_depth, dir_mask, img_scale)
        val = get_imgs_depths_and_masks(iddataset['val'], dir_img, dir_depth, dir_mask, img_scale)

        epoch_loss = 0

        # create batch
        # batch size should < 4000 due to the amount of data avaliable
        num = 0
        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([x[0] for x in b]).astype(np.float32)
            depths = np.array([x[1] for x in b]).astype(np.double)
            true_masks = np.array([x[2] for x in b])

            imgs = torch.from_numpy(imgs)
            depths = torch.from_numpy(depths)
            true_masks = torchvision.transforms.ToTensor()(true_masks)

            if gpu:
                imgs = imgs.cuda()
                depths = depths.cuda()
                true_masks = true_masks.cuda()

            # train
            masks_pred = net(imgs, depths)
            masks_probs = torch.sigmoid(masks_pred)
            # true_masks = torch.sigmoid(true_masks) # add this, IDK why loss negative

            # stretch result to one dimension
            masks_probs_flat = masks_probs.view(-1)
            print ("predicted:", masks_probs_flat)
            true_masks_flat = true_masks.view(-1)
            print ("true:", true_masks_flat)

            loss = criterion(masks_probs_flat, true_masks_flat)
            print("loss=", loss.item())
            epoch_loss += loss.item()
            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num = num+1
        print('Epoch finished ! Loss: {}'.format(epoch_loss / num))

        # validation
        if validation:
            val_dice = eval_net(net, val, gpu)
            print('Validation Dice Coeff: {}'.format(val_dice))

        # save parameter
        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=96, help='downscaling factor of the images')
    parser.add_option('-w', '--weight_init', dest='weight_init', default=0.01,
                      type='float', help='weight initialization number')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    # init artgs
    args = get_args()

    # 3 channels: 3 form image, 1 mask
    # 1 classes: separate salt and others
    unet = UNet(n_channels=3, n_classes=1)

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.001)
    unet.apply(init_weights)

    if args.load:
        unet.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        unet.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=unet,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale,
                  weight_init=args.weight_init)
    except KeyboardInterrupt:
        torch.save(unet.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
