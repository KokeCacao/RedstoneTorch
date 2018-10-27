import os
import sys
from datetime import datetime

import tensorboardwriter
import torch
import numpy as np
from torch.utils import data

import config
from dataset.hpa_dataset import HPAData

from loss.focal import FocalLoss
from net.proteinet.proteinet_model import se_resnext101_32x4d_modified

import matplotlib as mpl

from utils.load import save_checkpoint_fold, load_checkpoint_all_fold, cuda
from utils import encode

if os.environ.get('DISPLAY', '') == '':
    print('WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.')
    mpl.use('Agg')
from matplotlib import pyplot as plt


class HPAProject:
    def __init__(self, writer):
        self.writer = writer
        self.train_begin = None
        self.epoch_begin = None
        self.fold_begin = None

        self.optimizers = []
        self.nets = []
        for fold in range(config.MODEL_FOLD):
            print("     Creating Fold: #{}".format(fold))
            net = se_resnext101_32x4d_modified(num_classes=28, pretrained='imagenet')
            if config.TRAIN_GPU_ARG: net = torch.nn.DataParallel(net, device_ids=config.TRAIN_GPU_LIST)
            optimizer = torch.optim.Adam(params=net.parameters(), lr=config.MODEL_LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.MODEL_WEIGHT_DEFAY)
            self.optimizers = self.optimizers.append(optimizer)  # all parameter learnable
            cuda(net)
            self.nets = self.nets.append(net)
        load_checkpoint_all_fold(self.nets, self.optimizers, config.DIRECTORY_LOAD)

        # TODO: load 10 model together, save 10 model

        self.dataset = HPAData(config.DIRECTORY_CSV, config.DIRECTORY_IMG)
        self.folded_samplers = self.dataset.get_fold_sampler(fold=config.MODEL_FOLD)

    def run(self):
        try:

            self.train_begin = datetime.now()
            for epoch in range(config.MODEL_EPOCHS):
                self.step_epoch(nets=self.nets,
                                optimizers=self.optimizers,
                                batch_size=config.MODEL_BATCH_SIZE
                                )




        except KeyboardInterrupt as e:
            print(e)
            self.writer.close()
            print("To Resume: python train.py --versiontag 'test' --projecttag " + config.PROJECT_TAG + "--loadfile " + config.lastsave)
            print("Or: python train.py --tag 'default' --load " + config.DIRECTORY_CHECKPOINT + config.versiontag + "-" + config.DIRECTORY_CP_NAME.format(config.epoch - 1))
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    def step_epoch(self,
                   nets,
                   optimizers,
                   batch_size
                   ):
        self.epoch_begin = datetime.now()
        config.epoch = config.epoch + 1

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

        epoch_evaluations = np.array([])
        for fold, (net, optimizer) in enumerate(zip(nets, optimizers)):
            fold_evaluation = self.step_fold(fold, net, optimizer, batch_size)
            epoch_evaluations = np.concatenate((epoch_evaluations, [fold_evaluation.fold_losses]), axis=None)

            """DISPLAY"""
            best_id, best_loss, best_pred = fold_evaluation.best()
            worst_id, worst_loss, worst_pred = fold_evaluation.worst()
            best_img = self.dataset.get_load_image_by_id(best_id)
            best_label = self.dataset.get_load_label_by_id(best_id)
            worst_img = self.dataset.get_load_image_by_id(worst_id)
            worst_label = self.dataset.get_load_label_by_id(worst_id)
            tensorboardwriter.write_best_img(self.writer, img=best_img, label=best_label, id=best_id, loss=best_loss, pred=best_pred, fold=fold)
            tensorboardwriter.write_worst_img(self.writer, img=worst_img, label=worst_label, id=worst_id, loss=worst_loss, pred=worst_pred, fold=fold)

        """SAVE"""
        save_checkpoint_fold([x.state_dict() for x in nets], [x.state_dict() for x in optimizers])

        """DISPLAY"""
        tensorboardwriter.write_eval_loss(self.writer, {"EpochLoss": epoch_evaluations.mean(), "EpochSTD": epoch_evaluations.std()}, config.epoch)
        tensorboardwriter.write_loss_distribution(self.writer, epoch_evaluations.flatten(), config.epoch)

    def step_fold(self, fold, net, optimizer, batch_size):
        self.fold_begin = datetime.now()
        config.fold = fold

        train_sampler = self.folded_samplers[config.fold]["train"]
        validation_sampler = self.folded_samplers[config.fold]["val"]
        train_loader = data.DataLoader(self.dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=config.TRAIN_NUM_WORKER)
        validation_loader = data.DataLoader(self.dataset, batch_size=batch_size, sampler=validation_sampler, shuffle=False, num_workers=config.TRAIN_NUM_WORKER)

        epoch_loss = 0

        for batch_index, (ids, image, labels_0, image_for_display) in enumerate(train_loader, 0):
            """TRAIN NET"""
            config.global_steps[fold] = config.global_steps[fold] + 1
            if config.TRAIN_GPU_ARG: image = image.cuda()
            predict = net(image)
            loss = FocalLoss()(predict=predict, target=labels_0)
            epoch_loss = epoch_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """OUTPUT"""
            train_duration = self.fold_begin - self.train_begin
            epoch_duration = self.fold_begin - self.epoch_begin
            print("""
                            SinceTrain: {}
                            SinceEpoch: {}
                            Epoch: {}
                            Fold: {}
                            GlobalStep: {}
                            BatchIndex: {}
                        """.format(train_duration, epoch_duration, config.epoch, config.fold, config.global_steps[fold], batch_index))
            tensorboardwriter.write_loss(self.writer, {'Epoch' + '-f' + str(config.fold): config.epoch, 'TrainLoss' + '-f' + str(config.fold): loss.item(), 'IOU' + '-f' + str(config.fold): 0}, config.global_steps[fold])

            """CLEAN UP"""
            del ids, image, labels_0, image_for_display
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()  # release gpu memory

        print("""
            Epoch: {}
            EpochLoss: {}
        """.format(config.epoch, epoch_loss / (batch_index + 1)))
        if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()

        evaluation = HPAEvaluation()
        loss = evaluation.eval(net, validation_loader)
        print('Validation Dice Coeff: {}'.format(loss))
        # if config.DISPLAY_HISTOGRAM:
        #     for i, (name, param) in enumerate(net.named_parameters()):
        #         print("Calculating Histogram #{}".format(i))
        #         writer.add_histogram(name, param.clone().cpu().data.numpy(), config.epoch)
        if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()  # release gpu memory
        return evaluation


class HPAEvaluation:
    def __init__(self):
        """
        loss of one fold
        self.fold_losses = [
            eval_epoch -> [... losses of one batch...]
            eval_epoch -> [... losses of one batch...]
            eval_epoch -> [... losses of one batch...]
            eval_epoch -> [... losses of one batch...]
            eval_epoch -> [... losses of one batch...]
            eval_epoch -> [... losses of one batch...]
        ]
        """
        self.fold_losses = np.array([])
        self.best_id = None
        self.worst_id = None
        self.best_loss = None
        self.worst_loss = None
        self.best_pred = None
        self.worst_pred = None

    def eval(self, nets, validation_loader):

        """Evaluation without the densecrf with the dice coefficient"""
        epoch_losses = np.array([])
        epoch_dict = []  # list of fold losses with id

        for fold, net in enumerate(nets):
            fold_dict = dict()
            pred_dict = dict()
            for batch_index, (ids, image, labels_0, image_for_display) in enumerate(validation_loader, 0):

                """CALCULATE LOSS"""
                if config.TRAIN_GPU_ARG: image = image.cuda()
                predict = net(image)
                loss = FocalLoss()(predict=predict, target=labels_0)
                print("DEBUG: ", loss.item().shape)
                epoch_losses = np.concatenate((epoch_losses, np.array(loss.item()).flatten()), axis=None)
                for id, loss_item in zip(ids, loss.item()): fold_dict[id] = loss_item
                for id, pred in zip(ids, predict): pred_dict[id] = pred

                """EVALUATE LOSS"""
                min_loss = min(fold_dict.values())
                min_key = min(fold_dict, key=fold_dict.get)
                if min_loss < self.best_loss:
                    self.best_loss = min_loss
                    self.best_id = min_key
                    self.best_pred = pred_dict[min_key]
                max_loss = max(fold_dict.values())
                max_key = max(fold_dict, key=fold_dict.get)
                if max_loss > self.worst_loss:
                    self.worst_loss = max_loss
                    self.worst_id = max_key
                    self.worst_pred = pred_dict[max_key]

                """DISPLAY"""
                if config.DISPLAY_VISUALIZATION and batch_index == 0 and config.fold == 0: self.display(fold, ids, image, image_for_display, labels_0, predict, loss)

                """CLEAN UP"""
                del ids, image, labels_0, image_for_display
                if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
            epoch_dict = epoch_dict.append(fold_dict)

        self.fold_losses = np.concatenate((self.fold_losses, [epoch_losses]), axis=None)
        return epoch_losses.mean()

    def __int__(self):
        return self.mean()

    def mean(self, axis=0):
        return self.fold_losses.mean(axis)

    def std(self, axis=0):
        return self.fold_losses.std(axis)

    def best(self):
        return (self.best_id, self.best_loss, self.best_pred)

    def worst(self):
        return (self.worst_id, self.worst_loss, self.worst_pred)

    def display(self, fold, ids, transfered, untransfered, label, predicted, loss):
        tensorboardwriter.write_pr_curve(self.writer, label, predicted, config.global_steps[fold])
        for index, input_id in enumerate(ids):
            F = plt.figure()

            plt.subplot(321)
            plt.imshow(encode.tensor_to_np_four_channel_transarant(untransfered[index]))
            plt.title("Image_Real")
            plt.grid(False)

            plt.subplot(322)
            plt.imshow(encode.tensor_to_np_four_channel_transarant(transfered[index]))
            plt.title("Image_Trans")
            plt.grid(False)

            plt.subplot(323)
            plt.imshow(encode.tensor_to_np_four_channel_drop(untransfered[index]))
            plt.title("Mask_Real; label:{}".format(label[index]))
            plt.grid(False)

            plt.subplot(324)
            plt.imshow(encode.tensor_to_np_four_channel_drop(transfered[index]))
            plt.title("Mask_Trans; loss:{}".format(loss[index]))
            plt.grid(False)
            tensorboardwriter.write_image(self.writer, F, config.global_steps[fold])

    def get_epoch_loss_across_fold(self):
        return self.fold_losses.mean()

class HPAPrediction:
    pass