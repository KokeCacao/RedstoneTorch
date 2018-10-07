import sys
import os
from optparse import OptionParser

import torch
import torch.nn as nn
import torch.utils.data as data
from torch import optim
from torchvision import transforms

from eval import eval_net, iou_score
from unet import UNet
from resunet.resunet_model import ResUNet
from unet.unet_model import UNetResNet
from utils.data import TGSData
from datetime import datetime, date
from tensorboardX import SummaryWriter
from utils import lovasz_losses as L


# dir_prefix = 'drive/My Drive/ML/Pytorch-UNet/'
img_suffix = ".png"
mask_suffix = ".png"
validation = True


transform = {
    # 'depth': transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5], [0.5])
    # ]),
    'image': transforms.Compose([
        transforms.Resize((224,224)),
        # transforms.RandomResizedCrop(224),
        # transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.456, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ]),
    'mask': transforms.Compose([
        transforms.Resize((224,224)),
        # transforms.CenterCrop(224),
        # transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.225, 0.225, 0.225]),
        lambda x: x>0,
        lambda x: x.float(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.225, 0.225, 0.225]),
        lambda x: (x/4.4444)+0.5
    ])
}
# transform = {
#     # 'depth': transforms.Compose([
#     #     transforms.ToTensor(),
#     #     transforms.Normalize([0.5], [0.5])
#     # ]),
#     'image': transforms.Compose([
#         transforms.Resize((224,224)),
#         # transforms.RandomResizedCrop(224),
#         transforms.Grayscale(),
#         # transforms.RandomHorizontalFlip(),
#         # transforms.RandomVerticalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean = [0.5], std = [0.225])
#     ]),
#     'mask': transforms.Compose([
#         transforms.Resize((224,224)),
#         # transforms.CenterCrop(224),
#         transforms.Grayscale(),
#         # transforms.RandomHorizontalFlip(),
#         # transforms.RandomVerticalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean = [0.5], std = [0.225]),
#         lambda x: x>0,
#         lambda x: x.float()
#     ])
# }

def train_net(net,
              epochs=5,
              batch_size=10,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              weight_init=0.01,
              data_percent=0.181818182,
              momentum=0.9,
              weight_decay=0.0005,
              seed=19):

    tgs_data = TGSData(dir_depth, dir_img, dir_mask, img_suffix, mask_suffix, transform)

    train_sampler, validation_sampler = tgs_data.get_sampler(tgs_data.get_img_names(), data_percent=data_percent, val_percent=val_percent, data_shuffle = False, train_shuffle=True, val_shuffle=False, seed=seed)

    random = 23
    # print("debug-image:",random, "is", tgs_data.get_data()['image'][random])
    # print("debug-z:",random, "is", tgs_data.get_data()['z'][random])
    # print("debug-mask:",random, "is", tgs_data.get_data()['mask'][random])

    print("Id Size: {}".format(len(tgs_data.get_data()['id'])))
    print("Z Size: {}".format(len(tgs_data.get_data()['z'])))
    print("Image Size: {}".format(len(tgs_data.get_data()['image'])))
    print("Mask Size: {}".format(len(tgs_data.get_data()['mask'])))
    zip_data = list(zip(tgs_data.get_data()['id'], tgs_data.get_data()['z'], tgs_data.get_data()['image'], tgs_data.get_data()['mask']))
    # x_data = list(zip(tgs_data.get_data()['id'], tgs_data.get_data()['z'], tgs_data.get_data()['image']))
    # y_data = list(zip(tgs_data.get_data()['id'], tgs_data.get_data()['mask']))
    print("Zip-Data Size: {}".format(len(zip_data)))

    # if args.visualization:
    #     visual_id = tgs_data.get_data()['id'][:10]
    #     # visual_z = tgs_data.get_data()['z'][:10].float()
    #     visual_image = torch.tensor(tgs_data.get_data()['image'][:10])
    #     visual_mask = torch.tensor(tgs_data.get_data()['mask'][:10])
    #     writer.add_embedding(visual_image.view(10), metadata="image_"+visual_id, label_img=visual_image.unsqueeze(1))
    #     writer.add_embedding(visual_mask.view(10), metadata="mask_"+visual_id, label_img=visual_mask.unsqueeze(1))

    train_loader = data.DataLoader(zip_data, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=0)
    validation_loader = data.DataLoader(zip_data, batch_size=batch_size, sampler=validation_sampler, shuffle=False, num_workers=0)

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
        Momentum: {}
        Weight_decay: {}
    '''.format(epochs, batch_size, lr, tgs_data.train_len, tgs_data.val_len, str(save_cp), str(gpu), weight_init, momentum, weight_decay))

    # optimizer = optim.SGD(net.parameters(),
    #                       lr=lr,
    #                       momentum=momentum,
    #                       weight_decay=weight_decay)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay) # all parameter learnable
    train_begin = datetime.now()
    for epoch_index, epoch in enumerate(range(epochs)):
        epoch_begin = datetime.now()
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        epoch_loss = 0
        epoch_iou = 0

        # batch size should < 4000 due to the amount of data avaliable
        for batch_index, (id, z, image, true_mask) in enumerate(train_loader, 0):

            if gpu != "": #trying to use cuda 1 to prevent out of memory
                # z = z.cuda()
                image = image.cuda(0)
                true_mask = true_mask.cuda(0)

            # train

            # masks_pred = net(image, z)
            masks_pred = net(image)
            # true_masks = torch.sigmoid(true_masks) # add this, IDK why loss negative

            # calculating iou
            iou = iou_score(masks_pred, true_mask).mean().float()
            epoch_iou = epoch_iou + iou
            # print("iou:", iou.mean())

            # calculating loss
            # loss
            if epoch_index < 1e5: loss = torch.nn.BCELoss()(torch.sigmoid(masks_pred).view(-1), true_mask.view(-1))
            else: loss = L.lovasz_hinge(masks_pred, true_mask, ignore=None)



            epoch_loss += loss.item()

            now = datetime.now()
            train_duration = now - train_begin
            epoch_duration = now - epoch_begin
            print("SinceTrain:{}, Since Epoch:{}".format(train_duration, epoch_duration))
            print('{0}# Epoch - {1:.6f}% ({2}/{3})batch ({4:}/{5:})data - TrainLoss: {6:.6f}, IOU: {7:.6f}'.format(epoch_index+1,
                                                                                                     (100*(batch_index+1.0)*batch_size)/tgs_data.train_len,
                                                                                                     batch_index+1,
                                                                                                     tgs_data.train_len/batch_size,
                                                                                                     (batch_index+1)*batch_size,
                                                                                                     tgs_data.train_len,
                                                                                                     loss.item(), iou))
            writer.add_scalars('loss/batch_training', {'Epoch': epoch_index+1, 'TrainLoss': loss.item(), 'IOU': iou}, epoch_index*batch_size+(batch_index+1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('{}# Epoch finished ! Loss: {}, IOU: {}'.format(epoch_index+1, epoch_loss/(batch_index+1), epoch_iou/(batch_index+1)))
        # validation
        if gpu != "": torch.cuda.empty_cache() # release gpu memory
        if validation:
            val_dice = eval_net(net, validation_loader, gpu, tgs_data, visualization=args.visualization, writer=writer, epoch_num=epoch_index+1)
            print('Validation Dice Coeff: {}'.format(val_dice))
            writer.add_scalars('loss/epoch_validation', {'Validation': val_dice}, epoch_index+1)
        if args.visualization:
            for name, param in net.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch_index+1)
        # save parameter
        if save_cp:
            if not os.path.exists(dir_checkpoint):
                os.makedirs(dir_checkpoint)
            torch.save(net.state_dict(), dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(dir_checkpoint + 'CP{}.pth'.format(epoch + 1)))
        if gpu != "": torch.cuda.empty_cache()  # release gpu memory

def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int', help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10, type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1, type='float', help='learning rate')
    # parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default="", help='use cuda, please put all gpu id here')
    parser.add_option('-g', '--gpu', dest='gpu', default="", help='use cuda, please put all gpu id here')
    parser.add_option('-c', '--load', dest='load', default=False, help='load file model')
    parser.add_option('-w', '--weight_init', dest='weight_init', default=0.01, type='float', help='weight initialization number')
    parser.add_option('-v', '--val_percent', dest='val_percent', default=0.05, type='float', help='percent for validation')
    parser.add_option('-p', '--dir_prefix', dest='dir_prefix', default='', help='the root directory')
    parser.add_option('-d', '--data_percent', dest='data_percent', default=1.0, type='float', help='the root directory')
    parser.add_option('-i', '--visualization', dest='visualization', action='store_true', default="False", help='visualization the data')
    parser.add_option('-t', '--tag', dest='tag', default="", help='tag for tensorboard-log')

    (options, args) = parser.parse_args()
    return options

def log_data(file_name, data):
    with open(file_name+".txt", "a+") as file:
        file.write(data+"\n")

if __name__ == '__main__':
    # init artgs
    args = get_args()
    args.tag = str(datetime.now()).replace(" ","-").replace(".","-").replace(":","-") + "-" + args.tag
    dir_prefix = args.dir_prefix
    dir_img = dir_prefix + 'data/train/images/output/' #augmentation
    dir_mask = dir_prefix + 'data/train/masks/output/' #augmentation
    dir_depth = dir_prefix + 'data/depths.csv'
    dir_checkpoint = dir_prefix + "tensorboard/" + args.tag + '/checkpoints/'
    writer = SummaryWriter("tensorboard/" + args.tag)
    print("Current Directory: " + str(os.getcwd()))
    print("====================================")
    print("Copy this line to command: " + "python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/tensorboard/" + args.tag + " --port=6006")

    # 3 channels: 3 form image, 1 mask
    # 1 classes: separate salt and others


    # net = ResUNet(n_channels=3, n_classes=1)
    # net = UNet(n_channels=3, n_classes=1)
    net = UNetResNet(encoder_depth=50, num_classes=1, num_filters=32, dropout_2d=0.2,
                 pretrained=True, is_deconv=True) #don't init weights, don't give depth
    if args.gpu != "": net = torch.nn.DataParallel(net, device_ids=[int(i) for i in args.gpu.split(",")])
    

    # def init_weights(m):
    #     if type(m) == nn.Linear:
    #         torchp.nn.init.xavier_uniform(m.weight)
    #         m.bias.data.fill_(args.weight_init)


    # print("Initializing Weights...")
    # net.apply(init_weights)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    torch.manual_seed(19)
    if args.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu #default
        print('Using GPU: [' + args.gpu + ']')
        torch.cuda.manual_seed_all(19)
        net.cuda()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  val_percent=args.val_percent,
                  gpu=args.gpu,
                  weight_init=args.weight_init,
                  data_percent=0.181818182*args.data_percent,
                  seed=19)
    except KeyboardInterrupt as e:
        print(e)
        torch.save(net.state_dict(), dir_checkpoint + 'INTERUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    writer.export_scalars_to_json("tensorboard/" + args.tag + "/all_scalars.json")
    writer.close()
# python train.py --epochs 5 --batch-size 32 --learning-rate 0.001 --dir_prefix '' --data_percent 0.01 --gpu "0,1" --visualization "True" --tag "test"

# python train.py --epochs 300 --batch-size 32 --learning-rate 0.001 --dir_prefix '' --data_percent 1.00 --gpu "0,1" --visualization "True" --tag "plot-test" --load tensorboard/2018-10-05-03-05-24-773432-first-train/checkpoints/CP16.pth

# python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=ResUnet/tensorboard/2018-10-05-03-05-24-773432-plot-test --port=6006