import os
import sys
from datetime import datetime

import torch
from torch.utils import data as data

import config
from dataset.hpa import HPAData
from eval import iou_score, eval_net
from loss import loss as L
from model.proteinet.proteinet_model import se_resnext101_32x4d
from train import writer, save_checkpoint, load_checkpoint, cuda


class HPAroject():
    def __init__(self):
        net = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        if config.TRAIN_GPU_ARG != "": net = torch.nn.DataParallel(net, device_ids=config.TRAIN_GPU_LIST)

        self.optimizer = torch.optim.Adam(params=net.parameters(), lr=config.MODEL_LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.MODEL_WEIGHT_DEFAY)  # all parameter learnable
        load_checkpoint(net, self.optimizer, config.TRAIN_LOAD)
        self.net = cuda(net)

    def run(self):
        try:
            self.train(net=self.net,
                       optimizer=self.optimizer,
                       epochs=config.MODEL_EPOCHS,
                       batch_size=config.MODEL_BATCH_SIZE,
                       fold=config.MODEL_FOLD)
        except KeyboardInterrupt as e:
            print(e)
            writer.close()
            save_checkpoint(self.net.state_dict(), self.optimizer.state_dict(), interupt=True)
            print("To Resume: python train.py --tag 'default' --load " + config.DIRECTORY_CHECKPOINT + "INTERUPT-" + config.tag + "-" + config.DIRECTORY_CP_NAME.format(config.epoch))
            print("Or: python train.py --tag 'default' --load " + config.DIRECTORY_CHECKPOINT + config.tag + "-" + config.DIRECTORY_CP_NAME.format(config.epoch - 1))
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    def train(self, net,
              optimizer,
              epochs,
              batch_size,
              fold):
        hpa_data = HPAData(config.DIRECTORY_CSV, config.DIRECTORY_IMG, )
        folded_samplers = hpa_data.get_fold_sampler(fold=fold)

        for f in range(fold):
            train_sampler = folded_samplers[f]["train"]
            validation_sampler = folded_samplers[f]["val"]
            train_loader = data.DataLoader(hpa_data, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=config.TRAIN_NUM_WORKER)
            validation_loader = data.DataLoader(hpa_data, batch_size=batch_size, sampler=validation_sampler, shuffle=False, num_workers=config.TRAIN_NUM_WORKER)

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
                val_dice = eval_net(net, validation_loader, gpu=gpu, visualization=config.DISPLAY_VISUALIZATION, writer=writer, epoch_num=epochs + 1)
                print('Validation Dice Coeff: {}'.format(val_dice))
                writer.add_scalars('loss/epoch_validation', {'Validation': val_dice}, epochs + 1)
            if config.DISPLAY_HISTOGRAM:
                for i, (name, param) in enumerate(net.named_parameters()):
                    print("Calculating Histogram #{}".format(i))
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epochs + 1)
            if config.TRAIN_GPU_ARG != "": torch.cuda.empty_cache()  # release gpu memory

