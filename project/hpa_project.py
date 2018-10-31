import itertools
import os
import sklearn
import sys
from datetime import datetime

import matplotlib as mpl
import numpy as np
import torch
import pandas as pd
from torch.utils import data
from tqdm import tqdm

import config
import tensorboardwriter
from dataset.hpa_dataset import HPAData, train_collate, val_collate, transform
from gpu import gpu_profile
from loss.f1 import competitionMetric, f1_macro
from loss.focal import FocalLoss, FocalLoss_reduced, Focal_Loss_from_git
from net.proteinet.proteinet_model import se_resnext101_32x4d_modified
from utils import encode
from utils.load import save_checkpoint_fold, load_checkpoint_all_fold, cuda, load_checkpoint_all_fold_without_optimizers, save_onnx

if os.environ.get('DISPLAY', '') == '':
    print('WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.')
    mpl.use('Agg')
from matplotlib import pyplot as plt


class HPAProject:
    """"."""

    """"TODO"""
    # TODO: Data pre processing - try normalize data mean and std (https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/18) Save as ".npy" with dtype = "uint8". Before augmentation, convert back to float32 and normalize them with dataset mean/std.
    # TODO: Ask your biology teacher about yellow channel
    # TODO: cosine (https://github.com/SeuTao/Kaggle_TGS2018_4th_solution/blob/master/loss/cyclic_lr.py)
    # TODO: try global average pooling instead of averag epooling
    # TODO: try set f1 to 0 when 0/0; (7 missing classes in LB) / (28 total classes) = 0.25, and if the organizer is interpreting 0/0 as 0
    # TODO: try to process RBY first, and then concat Green layer
    # TODO: freeze loaded layer, check if the layers loaded correctly (ie. I want to load as much as I can)
    # TODO: Zero padding in augmentation
    # TODO: Yes, using SGD with cosine annealing schedule. Also used Adadelta to start training, Padam for mid training, and SGD at the end. Then I freeze parts of the model and train the other layers. My current leading model is 2.3M params. Performs great locally, but public LB is 45% lower. (https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/69462#412909)
    # TODO: Better augmentation

    """"TESTINGS"""
    # TODO: test visualize your network
    # TODO: test f1 score output and curve

    """"READINGS"""
    # TODO: https://www.proteinatlas.org/learn/dictionary/cell/microtubule+organizing+center+3; https://www.proteinatlas.org/learn/dictionary/cell



    def __init__(self, writer):
        self.writer = writer
        self.train_begin = None
        self.epoch_begin = None
        self.fold_begin = None

        self.optimizers = []
        self.nets = []
        for fold in range(config.MODEL_FOLD):
            print("     Creating Fold: #{}".format(fold))
            net = se_resnext101_32x4d_modified(num_classes=config.TRAIN_NUMCLASS, pretrained='imagenet')
            net.apply(lambda m: torch.nn.init.kaiming_normal_(m, mode='fan_in', nonlinearity='relu'))
            if config.TRAIN_GPU_ARG: net = torch.nn.DataParallel(net, device_ids=config.TRAIN_GPU_LIST)

            self.optimizers.append(torch.optim.Adam(params=net.parameters(), lr=config.MODEL_LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.MODEL_WEIGHT_DEFAY))  # all parameter learnable
            self.nets.append(cuda(net))
        load_checkpoint_all_fold(self.nets, self.optimizers, config.DIRECTORY_LOAD)

        self.dataset = HPAData(config.DIRECTORY_CSV, config.DIRECTORY_IMG)
        self.folded_samplers = self.dataset.get_fold_sampler(fold=config.MODEL_FOLD)

        self.run()
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

        evaluation = HPAEvaluation(self.writer)
        for fold, (net, optimizer) in enumerate(zip(nets, optimizers)):
            self.step_fold(fold, net, optimizer, batch_size, evaluation)

        """DISPLAY"""
        best_id, best_loss = evaluation.best()
        worst_id, worst_loss = evaluation.worst()
        for fold, (best_id, best_loss, worst_id, worst_loss) in enumerate(zip(best_id, best_loss, worst_id, worst_loss)):
            best_img = self.dataset.get_load_image_by_id(best_id)
            best_label = self.dataset.get_load_label_by_id(best_id)
            worst_img = self.dataset.get_load_image_by_id(worst_id)
            worst_label = self.dataset.get_load_label_by_id(worst_id)
            tensorboardwriter.write_best_img(self.writer, img=best_img, label=best_label, id=best_id, loss=best_loss, fold=fold)
            tensorboardwriter.write_worst_img(self.writer, img=worst_img, label=worst_label, id=worst_id, loss=worst_loss, fold=fold)

        """LOSS"""
        f1 = f1_macro(evaluation.epoch_pred, evaluation.epoch_label).mean()
        f2 = sklearn.metrics.f1_score(evaluation.epoch_label > 0.5, evaluation.epoch_pred > 0.5, average='macro')
        print("F1 by sklearn = ".format(f2))
        tensorboardwriter.write_loss(self.writer, {"EpochLoss": f1}, config.epoch)

        """THRESHOLD"""
        if config.EVAL_IF_THRESHOLD_TEST:
            best_threshold = 0.0
            best_val = 0.0
            pbar = tqdm(config.EVAL_TRY_THRESHOLD)
            for threshold in pbar:
                score = f1_macro(evaluation.epoch_pred, evaluation.epoch_label, thresh=threshold).mean()
                if score > best_val:
                    best_threshold = threshold
                    best_val = score
                pbar.set_description("Threshold: {}; F1: {}".format(threshold, score))
            print("BestThreshold: {}, F1: {}".format(best_threshold, best_val))

        """SAVE"""
        save_checkpoint_fold([x.state_dict() for x in nets], [x.state_dict() for x in optimizers])

        """DISPLAY"""
        tensorboardwriter.write_eval_loss(self.writer, {"EpochLoss": evaluation.mean(), "EpochSTD": evaluation.std()}, config.epoch)
        tensorboardwriter.write_loss_distribution(self.writer, np.array(list(itertools.chain.from_iterable(evaluation.epoch_losses))).flatten(), config.epoch)

    def step_fold(self, fold, net, optimizer, batch_size, evaluation):
        self.fold_begin = datetime.now()
        config.fold = fold

        train_sampler = self.folded_samplers[config.fold]["train"]
        validation_sampler = self.folded_samplers[config.fold]["val"]
        train_loader = data.DataLoader(self.dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=config.TRAIN_NUM_WORKER, collate_fn=train_collate)
        validation_loader = data.DataLoader(self.dataset, batch_size=batch_size, sampler=validation_sampler, shuffle=False, num_workers=config.TRAIN_NUM_WORKER, collate_fn=val_collate)

        epoch_loss = 0

        for batch_index, (ids, image, labels_0, image_for_display) in enumerate(train_loader, 0):
            """UPDATE LR"""
            state = optimizer.state_dict()
            state['state']['lr'] = config.TRAIN_COSINE(config.global_steps)
            optimizer.load_state_dict(state)

            """TRAIN NET"""
            config.global_steps[fold] = config.global_steps[fold] + 1
            if config.TRAIN_GPU_ARG:
                image = image.cuda()
                labels_0 = labels_0.cuda()
            predict = net(image)
            predict = torch.nn.Softmax(predict)

            """LOSS"""
            loss = Focal_Loss_from_git(labels_0, predict)
            epoch_loss = epoch_loss + loss.flatten().mean()
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            loss = loss.detach().cpu().numpy()
            f1 = f1_macro(predict, labels_0).mean()

            """OUTPUT"""
            train_duration = self.fold_begin - self.train_begin
            epoch_duration = self.fold_begin - self.epoch_begin
            print("""SinceTrain: {}; SinceEpoch: {}; Epoch: {}; Fold: {}; GlobalStep: {}; BatchIndex: {}/{}; Loss: {}; F1: {}""".format(train_duration, epoch_duration, config.epoch, config.fold, config.global_steps[fold], batch_index, len(train_sampler)/config.MODEL_BATCH_SIZE, loss.flatten().mean(), f1))
            tensorboardwriter.write_loss(self.writer, {'Epoch/' + str(config.fold): config.epoch, 'TrainLoss/' + str(config.fold): loss.flatten().mean(),  'F1Loss/' + str(config.fold): f1}, config.global_steps[fold])

            """CLEAN UP"""
            del ids, image, labels_0, image_for_display
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()  # release gpu memory

        print("""
            Epoch: {}
            EpochLoss: {}
        """.format(config.epoch, epoch_loss / (batch_index + 1)))
        if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()

        loss = np.array(list(fold.values() for fold in evaluation.eval_fold(net, validation_loader).epoch_dict)).mean()
        print('Validation Dice Coeff: {}'.format(loss))
        # if config.DISPLAY_HISTOGRAM:
        #     for i, (name, param) in enumerate(net.named_parameters()):
        #         print("Calculating Histogram #{}".format(i))
        #         writer.add_histogram(name, param.clone().cpu().data.numpy(), config.epoch)
        if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()  # release gpu memory


class HPAEvaluation:
    def __init__(self, writer):
        self.writer = writer
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
        self.epoch_losses = [] # [loss.flatten()]
        self.epoch_dict = np.array([]) # [fold_loss_dict]
        self.f1_losses = np.array([])

        self.best_id = np.array([])
        self.worst_id = np.array([])
        self.best_loss = np.array([])
        self.worst_loss = np.array([])

        self.epoch_pred = np.array([])
        self.epoch_label = np.array([])

    def eval_epoch(self, nets=None, validation_loaders=None):

        if nets != None and validation_loaders != None:
            for fold, (net, validation_loader) in enumerate(zip(nets, validation_loaders)):
                self.eval_fold(net, validation_loader)
        return self

    def eval_fold(self, net, validation_loader):
        fold_loss_dict = dict()
        predict_total = np.array([])
        label_total = np.array([])
        for batch_index, (ids, image, labels_0, image_for_display) in enumerate(validation_loader, 0):
            """CALCULATE LOSS"""
            if config.TRAIN_GPU_ARG:
                image = image.cuda()
                labels_0 = labels_0.cuda()
            predict = net(image)
            predict = torch.nn.Softmax(predict)

            """LOSS"""
            loss = Focal_Loss_from_git(labels_0, predict).detach().cpu().numpy()
            self.epoch_losses.append(loss.flatten())
            for id, loss_item in zip(ids, loss.flatten()): fold_loss_dict[id] = loss_item
            np.append(self.f1_losses, f1_macro(predict, labels_0).mean())

            predict_total = np.concatenate((predict_total, predict.detach().cpu()), axis=0)
            label_total = np.concatenate((label_total, labels_0.cpu()), axis=0)

            """EVALUATE LOSS"""
            min_loss = min(fold_loss_dict.values())
            min_key = min(fold_loss_dict, key=fold_loss_dict.get)
            if min_loss < self.best_loss:
                np.append(self.best_loss, min_loss)
                np.append(self.best_id, min_key)
            max_loss = max(fold_loss_dict.values())
            max_key = max(fold_loss_dict, key=fold_loss_dict.get)
            if max_loss > self.worst_loss:
                np.append(self.worst_loss, max_loss)
                np.append(self.worst_id, max_key)

            """DISPLAY"""
            if config.DISPLAY_VISUALIZATION and batch_index == 0 and config.fold == 0: self.display(config.fold, ids, image, image_for_display, labels_0, predict, loss)

            """CLEAN UP"""
            del ids, image, labels_0, image_for_display, predict, loss
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
            if config.DEBUG_TRAISE_GPU: gpu_profile(frame=sys._getframe(), event='line', arg=None)
        """LOSS"""
        f1 = f1_macro(predict_total, label_total).mean()
        tensorboardwriter.write_loss(self.writer, {"FoldLoss/" + str(config.fold): np.array(fold_loss_dict.values()).mean(), "FoldF1/" + str(config.fold): f1}, config.global_steps[-1])
        tensorboardwriter.write_pr_curve(self.writer, label_total, predict_total, config.global_steps[-1])
        self.epoch_pred = np.concatenate((self.epoch_pred, predict_total), axis=0)
        self.epoch_label = np.concatenate((self.epoch_label, label_total), axis=0)
        del predict_total, label_total

        self.epoch_dict = np.concatenate((self.epoch_dict, [fold_loss_dict]), axis=0)
        del fold_loss_dict


        if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
        return self


    def __int__(self):
        return self.mean()

    def mean(self, axis=None):
        if axis == None: return np.array(list(itertools.chain.from_iterable(self.epoch_losses))).mean()
        print("WARNING: self.epoch_losse may have different shape according to different shape of loss caused by different batch. Numpy cannot take the mean of it is baches shapes are different.")
        return np.array(self.epoch_losses).mean(axis)

    def std(self, axis=None):
        if axis == None: return np.array(list(itertools.chain.from_iterable(self.epoch_losses))).std()
        print("WARNING: self.epoch_losse may have different shape according to different shape of loss caused by different batch. Numpy cannot take the mean of it is baches shapes are different.")
        return np.array(self.epoch_losses).std(axis)

    def f1_mean(self):
        return self.f1_losses.mean()

    def best(self):
        return (self.best_id, self.best_loss)

    def worst(self):
        return (self.worst_id, self.worst_loss)

    def display(self, fold, ids, transfereds, untransfereds, labels, predicteds, losses):
        tensorboardwriter.write_pr_curve(self.writer, labels, predicteds, config.global_steps[fold])

        for index, (id, transfered, untransfered, label, predicted, loss) in enumerate(zip(ids, transfereds, untransfereds, labels, predicteds, losses)):
            if index != 0: continue

            F = plt.figure()

            plt.subplot(321)
            plt.imshow(encode.tensor_to_np_four_channel_transarant(untransfered))
            plt.title("Image_Real")
            plt.grid(False)

            plt.subplot(322)
            plt.imshow(encode.tensor_to_np_four_channel_transarant(transfered))
            plt.title("Image_Trans")
            plt.grid(False)

            plt.subplot(323)
            plt.imshow(encode.tensor_to_np_four_channel_drop(untransfered))
            plt.title("Mask_Real; label:{}".format(label))
            plt.grid(False)

            plt.subplot(324)
            plt.imshow(encode.tensor_to_np_four_channel_drop(transfered))
            plt.title("Mask_Trans; loss:{}".format(loss))
            plt.grid(False)
            tensorboardwriter.write_image(self.writer, F, config.global_steps[fold])

class HPAPrediction:
    def __init__(self, writer, threshold = 0.5):
        self.threshold = threshold
        self.nets = []
        for fold in range(config.MODEL_FOLD):
            print("     Creating Fold: #{}".format(fold))
            net = se_resnext101_32x4d_modified(num_classes=config.TRAIN_NUMCLASS, pretrained='imagenet')
            if config.TRAIN_GPU_ARG: net = torch.nn.DataParallel(net, device_ids=config.TRAIN_GPU_LIST)

            self.nets.append(cuda(net))
        load_checkpoint_all_fold_without_optimizers(self.nets, config.DIRECTORY_LOAD)
        if config.DISPLAY_SAVE_ONNX:
            for net in self.nets:
                save_onnx(net, (config.MODEL_BATCH_SIZE, 4, config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE), config.DIRECTORY_LOAD+"-"+str(net)+".onnx")

        self.dataset = HPAData(config.DIRECTORY_CSV, config.DIRECTORY_IMG, test=True)

        self.run()
    def run(self):
        torch.no_grad()
        """Used for Kaggle submission: predicts and encode all test images"""
        for fold, net in enumerate(self.nets):
            save_path = config.DIRECTORY_LOAD + "-" + config.PREDICTION_TAG + "-" + str(fold) + ".csv"

            if os.path.exists(save_path):
                os.remove(save_path)
                print("WARNING: delete file '{}'".format(save_path))

            with open(save_path, 'a') as f:
                f.write('Id,Predicted\n')
                pbar = tqdm(self.dataset.id)
                for index, id in enumerate(pbar):
                    input = self.dataset.get_load_image_by_id(id)
                    input = transform(ids=None, image_0=input, labels_0=None, train=False, val=False).unsqueeze(0)

                    if config.TRAIN_GPU_ARG: input = input.cuda()
                    predict = (net(input).detach().cpu().numpy() > self.threshold).astype(np.int16)
                    encoded = self.dataset.multilabel_binarizer.inverse_transform(predict)
                    encoded = list(encoded[0])


                    f.write('{},{}\n'.format(id, " ".join(str(x) for x in encoded)))
                    pbar.set_description("Fold: {}; Index: {}; Out: {}".format(fold, index, encoded))
                    del id, input, predict, encoded
                    if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()

            """ORGANIZE"""
            f1 = pd.read_csv(config.DIRECTORY_SAMPLE_CSV)
            f1.drop('Predicted', axis=1, inplace=True)
            f2 = pd.read_csv(save_path)
            f1 = f1.merge(f2, left_on='Id', right_on='Id', how='outer')
            os.remove(save_path)
            f1.to_csv(save_path, index=False)