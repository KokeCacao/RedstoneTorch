import itertools
import operator
import os
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch.nn import BCELoss
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm

import config
import tensorboardwriter
from dataset.hpa_dataset import HPAData, train_collate, val_collate
from gpu import gpu_profile
from loss.f1 import f1_macro, Differenciable_F1
from loss.focal import FocalLoss_Sigmoid
from net.proteinet.proteinet_model import se_resnext101_32x4d_modified
from utils import encode, load
from utils.load import save_checkpoint_fold, load_checkpoint_all_fold, cuda, load_checkpoint_all_fold_without_optimizers, save_onnx
from visualization.gradcam import GradCam
from visualization.guided_backprop import GuidedBackprop
from visualization.guided_gradcam import guided_grad_cam
from visualization.misc_functions import save_gradient_images, convert_to_grayscale

if os.environ.get('DISPLAY', '') == '':
    print('WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.')
    mpl.use('Agg')
from matplotlib import pyplot as plt


class HPAProject:
    def __init__(self, writer):
        self.writer = writer

        self.optimizers = []
        self.nets = []
        self.lr_schedulers = []
        for fold in range(config.MODEL_FOLD):
            if fold not in config.MODEL_TRAIN_FOLD:
                print("     Skipping Fold: #{}".format(fold))
            else:
                print("     Creating Fold: #{}".format(fold))
                net = se_resnext101_32x4d_modified(num_classes=config.TRAIN_NUMCLASS, pretrained='imagenet')
                if config.TRAIN_GPU_ARG: net = torch.nn.DataParallel(net, device_ids=config.TRAIN_GPU_LIST)

                # self.optimizers.append(torch.optim.Adam(params=net.parameters(), lr=config.MODEL_INIT_LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.MODEL_WEIGHT_DEFAY))
                optimizer = torch.optim.Adadelta(params=net.features.parameters(), lr=config.MODEL_INIT_LEARNING_RATE, rho=0.9, eps=1e-6, weight_decay=config.MODEL_WEIGHT_DEFAY)
                self.optimizers.append(optimizer)
                self.nets.append(net)
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2 * int(27964.8 / config.MODEL_BATCH_SIZE), verbose=False, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
                self.lr_schedulers.append(lr_scheduler)

                # for name, param in net.named_parameters():
                #     if param.requires_grad:
                #         print (name)
        load_checkpoint_all_fold(self.nets, self.optimizers, config.DIRECTORY_LOAD)

        print(self.nets[0])
        if config.DISPLAY_SAVE_ONNX and config.DIRECTORY_LOAD: save_onnx(self.nets[0], (config.MODEL_BATCH_SIZE, 4, config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE), config.DIRECTORY_LOAD + ".onnx")

        self.dataset = HPAData(config.DIRECTORY_CSV, load_img_dir=config.DIRECTORY_PREPROCESSED_IMG, img_suffix=config.DIRECTORY_PREPROCESSED_SUFFIX_IMG, load_strategy="train", load_preprocessed_dir=True, writer=self.writer)
        self.folded_samplers = self.dataset.get_stratified_samplers(fold=config.MODEL_FOLD)

        self.run()

    def run(self):
        try:
            for epoch in range(config.MODEL_EPOCHS):
                self.step_epoch(nets=self.nets,
                                optimizers=self.optimizers,
                                lr_schedulers=self.lr_schedulers,
                                batch_size=config.MODEL_BATCH_SIZE
                                )
                if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
                """SAVE"""
                save_checkpoint_fold([x.state_dict() for x in self.nets], [x.state_dict() for x in self.optimizers])
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()

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
                   lr_schedulers,
                   batch_size
                   ):
        config.epoch = config.epoch + 1

        evaluation = HPAEvaluation(self.writer, self.dataset.multilabel_binarizer)
        for fold, (net, optimizer, lr_scheduler) in enumerate(zip(nets, optimizers, lr_schedulers)):
            """Switch Optimizers"""
            if config.epoch == 50:
                optimizer = torch.optim.SGD(net.parameters(), lr=config.MODEL_INIT_LEARNING_RATE, momentum=config.MODEL_MOMENTUM, dampening=0, weight_decay=config.MODEL_WEIGHT_DEFAY, nesterov=False)
                tensorboardwriter.write_text(self.writer, "Switch to torch.optim.SGD, weight_decay={}, momentum={}".format(config.MODEL_WEIGHT_DEFAY, config.MODEL_MOMENTUM), config.global_steps[fold])
            net = net.cuda()
            optimizer = load.move_optimizer_to_cuda(optimizer)
            self.step_fold(fold, net, optimizer, lr_scheduler, batch_size)
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
            val_loss, val_f1 = evaluation.eval_fold(net, data.DataLoader(self.dataset,
                                                                         batch_size=batch_size,
                                                                         shuffle=False,
                                                                         sampler=self.folded_samplers[config.fold]["val"],
                                                                         batch_sampler=None,
                                                                         num_workers=config.TRAIN_NUM_WORKER,
                                                                         collate_fn=val_collate,
                                                                         pin_memory=False,
                                                                         drop_last=False,
                                                                         timeout=0,
                                                                         worker_init_fn=None,
                                                                         ))
            print("""
                ValidLoss: {}, ValidF1: {}
            """.format(val_loss, val_f1))
            net = net.cpu()
            optimizer = load.move_optimizer_to_cpu(optimizer)  # 3299Mb
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()  # 1215Mb

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
        f1_2 = metrics.f1_score((evaluation.epoch_label > 0.5).astype(np.byte), (evaluation.epoch_pred > 0.5).astype(np.byte), average='macro')  # sklearn does not automatically import matrics.
        f1_dict = dict(("Class-{}".format(i), x) for i, x in enumerate(metrics.f1_score((evaluation.epoch_label > 0.5).astype(np.byte), (evaluation.epoch_pred > 0.5).astype(np.byte), average=None)))
        f1_dict.update({"EvalF1": f1, "Sklearn": f1_2})
        max_names = max(f1_dict.items(), key=operator.itemgetter(1))
        min_names = min(f1_dict.items(), key=operator.itemgetter(1))
        print("""
            F1 by sklearn = {}
            Max = {}, socre = {}
            Min = {}, score = {}
        """.format(f1_2, max_names[0], max_names[1], min_names[0], min_names[1]))
        tensorboardwriter.write_epoch_loss(self.writer, f1_dict, config.epoch)
        tensorboardwriter.write_pred_distribution(self.writer, evaluation.epoch_pred.flatten(), config.epoch)

        """THRESHOLD"""
        if config.EVAL_IF_THRESHOLD_TEST:
            best_threshold = 0.0
            best_val = 0.0

            best_threshold_dict = np.zeros(28)
            best_val_dict = np.zeros(28)

            pbar = tqdm(config.EVAL_TRY_THRESHOLD)
            for threshold in pbar:
                score = f1_macro(evaluation.epoch_pred, evaluation.epoch_label, thresh=threshold).mean()
                tensorboardwriter.write_threshold(self.writer, -1, score, threshold * 1000.0, config.fold)
                if score > best_val:
                    best_threshold = threshold
                    best_val = score
                pbar.set_description("Threshold: {}; F1: {}".format(threshold, score))

                for c in range(28):
                    score = metrics.f1_score(evaluation.epoch_label[:][c], (evaluation.epoch_pred[:][c] > threshold))
                    tensorboardwriter.write_threshold(self.writer, c, score, threshold * 1000.0, config.fold)
                    if score > best_val_dict[c]:
                        best_threshold_dict[c] = threshold
                        best_val_dict[c] = score

            tensorboardwriter.write_best_threshold(self.writer, -1, best_val, best_threshold, config.epoch, config.fold)
            for c in range(28): tensorboardwriter.write_best_threshold(self.writer, c, best_val_dict[c], best_threshold_dict[c], config.epoch, config.fold)

        """HISTOGRAM"""
        if config.DISPLAY_HISTOGRAM:
            tensorboardwriter.write_eval_loss(self.writer, {"EvalFocalMean": evaluation.mean(), "EvalFocalSTD": evaluation.std()}, config.epoch)
            tensorboardwriter.write_loss_distribution(self.writer, np.array(list(itertools.chain.from_iterable(evaluation.epoch_losses))).flatten(), config.epoch)

        """CLEAN UP"""
        del evaluation

    def step_fold(self, fold, net, optimizer, lr_scheduler, batch_size):
        config.fold = fold

        epoch_loss = 0
        epoch_f1 = 0

        # pin_memory: https://blog.csdn.net/tsq292978891/article/details/80454568
        train_loader = data.DataLoader(self.dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       sampler=self.folded_samplers[config.fold]["train"],
                                       batch_sampler=None,
                                       num_workers=config.TRAIN_NUM_WORKER,
                                       collate_fn=train_collate,
                                       pin_memory=True,
                                       drop_last=False,
                                       timeout=0,
                                       worker_init_fn=None,
                                       )
        pbar = tqdm(train_loader)
        train_len = len(train_loader) + 1e-10

        print("Set Model Trainning mode to trainning=[{}]".format(net.train().training))
        for batch_index, (ids, image, labels_0, image_for_display) in enumerate(pbar):
            # 1215MB -> 4997MB = 3782

            # """UPDATE LR"""
            # if config.global_steps[fold] == 2 * 46808 / 32 - 1: print("Perfect Place to Stop")
            # optimizer.state['lr'] = config.TRAIN_TRY_LR_FORMULA(config.global_steps[fold]) if config.TRAIN_TRY_LR else config.TRAIN_COSINE(config.global_steps[fold])

            """TRAIN NET"""
            config.global_steps[fold] = config.global_steps[fold] + 1
            if config.TRAIN_GPU_ARG:
                image = image.cuda()
                labels_0 = labels_0.cuda()
            logits_predict = net(image)
            sigmoid_predict = torch.sigmoid(logits_predict)

            """LOSS"""
            focal = FocalLoss_Sigmoid(alpha=0.25, gamma=5, eps=1e-7)(labels_0, logits_predict)
            f1, precise, recall = Differenciable_F1(beta=1)(labels_0, logits_predict)
            bce = BCELoss()(sigmoid_predict, labels_0)
            positive_bce = BCELoss(weight=labels_0 * 20 + 1)(sigmoid_predict, labels_0)
            # [1801.5 / 12885, 1801.5 / 1254, 1801.5 / 3621, 1801.5 / 1561, 1801.5 / 1858, 1801.5 / 2513, 1801.5 / 1008, 1801.5 / 2822, 1801.5 / 53, 1801.5 / 45, 1801.5 / 28, 1801.5 / 1093, 1801.5 / 688, 1801.5 / 537, 1801.5 / 1066, 1801.5 / 21, 1801.5 / 530, 1801.5 / 210, 1801.5 / 902, 1801.5 / 1482, 1801.5 / 172, 1801.5 / 3777, 1801.5 / 802, 1801.5 / 2965, 1801.5 / 322, 1801.5 / 8228, 1801.5 / 328, 1801.5 / 11] / (1801.5 / 11)
            weighted_bce = BCELoss(weight=torch.Tensor(
                [8.53705860e-04, 8.77192982e-03, 3.03783485e-03, 7.04676489e-03,
                 5.92034446e-03, 4.37723836e-03, 1.09126984e-02, 3.89794472e-03,
                 2.07547170e-01, 2.44444444e-01, 3.92857143e-01, 1.00640439e-02,
                 1.59883721e-02, 2.04841713e-02, 1.03189493e-02, 5.23809524e-01,
                 2.07547170e-02, 5.23809524e-02, 1.21951220e-02, 7.42240216e-03,
                 6.39534884e-02, 2.91236431e-03, 1.37157107e-02, 3.70994941e-03,
                 3.41614907e-02, 1.33689840e-03, 3.35365854e-02, 1.00000000e+00]
            ).cuda())(sigmoid_predict, labels_0)
            if config.epoch < 10:
                loss = bce
            else:
                loss = f1 + weighted_bce
            if config.epoch == 10: tensorboardwriter.write_text(self.writer, "Switch to f1", config.global_steps[fold])
            """BACKPROP"""
            lr_scheduler.step((precise.detach().cpu().numpy().mean() + recall.detach().cpu().numpy().mean()) / 2, epoch=config.global_steps[fold])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """DETATCH"""
            focal = focal.detach().cpu().numpy().mean()
            f1 = f1.detach().cpu().numpy().mean()
            precise = precise.detach().cpu().numpy().mean()
            recall = recall.detach().cpu().numpy().mean()
            bce = bce.detach().cpu().numpy().mean()
            positive_bce = positive_bce.detach().cpu().numpy().mean()
            weighted_bce = weighted_bce.detach().cpu().numpy().mean()
            loss = loss.detach().cpu().numpy().mean()
            labels_0 = labels_0.cpu().numpy()
            logits_predict = logits_predict.detach().cpu().numpy()
            sigmoid_predict = sigmoid_predict.detach().cpu().numpy()
            # print(image)

            """SUM"""
            epoch_loss = epoch_loss + loss.mean()
            epoch_f1 = epoch_f1 + f1.mean()
            # f1 = f1_macro(logits_predict, labels_0).mean()

            """DISPLAY"""
            tensorboardwriter.write_memory(self.writer, "train")

            left = self.dataset.multilabel_binarizer.inverse_transform((np.expand_dims((np.array(labels_0).sum(0) < 1).astype(np.byte), axis=0)))[0]
            label = np.array(self.dataset.multilabel_binarizer.inverse_transform(labels_0)[0])
            pred = np.array(self.dataset.multilabel_binarizer.inverse_transform(logits_predict > 0.5)[0])
            pbar.set_description_str("(E{}-F{}) Stp:{} Label:{} Pred:{} Left:{}".format(config.epoch, config.fold, int(config.global_steps[fold]), label, pred, left))
            # pbar.set_description_str("(E{}-F{}) Stp:{} Focal:{:.4f} F1:{:.4f} lr:{:.4E} BCE:{:.2f}|{:.2f}".format(config.epoch, config.fold, int(config.global_steps[fold]), focal, f1, optimizer.param_groups[0]['lr'], weighted_bce, bce))
            # pbar.set_description_str("(E{}-F{}) Stp:{} Y:{}, y:{}".format(config.epoch, config.fold, int(config.global_steps[fold]), labels_0, logits_predict))

            tensorboardwriter.write_loss(self.writer, {'Epoch/{}'.format(config.fold): config.epoch,
                                                       'LearningRate{}/{}'.format(optimizer.__class__.__name__, config.fold): optimizer.param_groups[0]['lr'],
                                                       'Loss/{}'.format(config.fold): loss,
                                                       'F1/{}'.format(config.fold): f1,
                                                       'Focal/{}'.format(config.fold): focal,
                                                       'PositiveBCE/{}'.format(config.fold): positive_bce,
                                                       'WeightedBCE/{}'.format(config.fold): weighted_bce,
                                                       'BCE/{}'.format(config.fold): bce,
                                                       'Precision/{}'.format(config.fold): precise,
                                                       'Recall/{}'.format(config.fold): recall,
                                                       'PredictProbability/{}'.format(config.fold): logits_predict.mean(),
                                                       'LabelProbability/{}'.format(config.fold): labels_0.mean(),
                                                       }, config.global_steps[fold])

            """CLEAN UP"""
            del ids, image, image_for_display
            del focal, f1, precise, recall, bce, positive_bce, weighted_bce, loss, labels_0, logits_predict, sigmoid_predict
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()  # release gpu memory
        del train_loader, pbar

        train_loss = epoch_loss / train_len
        epoch_f1 = epoch_f1 / train_len
        print("""
            Epoch: {}, Fold: {}
            TrainLoss: {}, TrainF1: {}
        """.format(config.epoch, config.fold, train_loss, epoch_f1))

        del train_loss

        # if config.DISPLAY_HISTOGRAM:
        #     for i, (name, param) in enumerate(net.named_parameters()):
        #         print("Calculating Histogram #{}".format(i))
        #         writer.add_histogram(name, param.clone().cpu().data.numpy(), config.epoch)
        if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()  # release gpu memory


class HPAEvaluation:
    def __init__(self, writer, binarlizer):
        self.writer = writer
        self.binarlizer = binarlizer
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
        if config.DISPLAY_HISTOGRAM: self.epoch_losses = []  # [loss.flatten()]
        self.mean_losses = []
        # self.epoch_dict = np.array([]) # [fold_loss_dict]
        self.f1_losses = np.array([])

        self.best_id = None
        self.worst_id = None
        self.best_loss = None
        self.worst_loss = None

        self.epoch_pred = None
        self.epoch_label = None

    def eval_epoch(self, nets=None, validation_loaders=None):

        if nets != None and validation_loaders != None:
            for fold, (net, validation_loader) in enumerate(zip(nets, validation_loaders)):
                self.eval_fold(net, validation_loader)
        return self

    def eval_fold(self, net, validation_loader):
        focal_losses = np.array([])
        predict_total = None
        label_total = None

        self.best_id = []
        self.worst_id = []
        self.best_loss = []
        self.worst_loss = []

        pbar = tqdm(itertools.chain(validation_loader, validation_loader, validation_loader, validation_loader))
        print("Set Model Trainning mode to trainning=[{}]".format(net.eval().training))
        for batch_index, (ids, image, labels_0, image_for_display) in enumerate(pbar):
            """CALCULATE LOSS"""
            if config.TRAIN_GPU_ARG:
                image = image.cuda()
                labels_0 = labels_0.cuda()
            logits_predict = net(image)
            sigmoid_predict = torch.sigmoid(logits_predict)

            """LOSS"""
            focal = FocalLoss_Sigmoid(alpha=0.25, gamma=5, eps=1e-7)(labels_0, logits_predict)
            f1, precise, recall = Differenciable_F1(beta=1)(labels_0, logits_predict)
            # bce = BCELoss()(sigmoid_predict, labels_0)
            # positive_bce = BCELoss(weight=labels_0*20+1)(sigmoid_predict, labels_0)
            # weighted_bce = BCELoss(weight=torch.Tensor([1801.5/12885, 1801.5/1254, 1801.5/3621, 1801.5/1561, 1801.5/1858, 1801.5/2513, 1801.5/1008, 1801.5/2822, 1801.5/53, 1801.5/45, 1801.5/28, 1801.5/1093, 1801.5/688, 1801.5/537, 1801.5/1066, 1801.5/21, 1801.5/530, 1801.5/210, 1801.5/902, 1801.5/1482, 1801.5/172, 1801.5/3777, 1801.5/802, 1801.5/2965, 1801.5/322, 1801.5/8228, 1801.5/328, 1801.5/11]).cuda())(torch.sigmoid(logits_predict), labels_0)
            # loss = f1 + bce.sum()

            """EVALUATE LOSS"""
            focal = focal.detach()
            focal_min = focal.min()
            focal_max = focal.max()
            focal_min_id = (focal == focal_min).nonzero().view(1)[0]
            focal_max_id = (focal == focal_max).nonzero().view(1)[0]
            focal_min = focal_min.cpu().numpy()
            focal_max = focal_max.cpu().numpy()
            focal_min_id = ids[focal_min_id.cpu().numpy()]
            focal_max_id = ids[focal_max_id.cpu().numpy()]
            self.best_loss = np.append(self.best_loss, focal_min)
            self.worst_loss = np.append(self.worst_loss, focal_max)
            self.best_id = np.append(self.best_id, focal_min_id)
            self.worst_id = np.append(self.worst_id, focal_max_id)

            """DETATCH"""
            focal = focal.cpu().numpy()
            focal_mean = focal.mean()
            f1 = f1.detach().cpu().numpy()
            precise = precise.detach().cpu().numpy().mean()
            recall = recall.detach().cpu().numpy().mean()
            # bce = bce.detach().cpu().numpy().mean()
            # positive_bce = positive_bce.detach().cpu().numpy().mean()
            # loss = loss.detach().cpu().numpy()
            labels_0 = labels_0.cpu().numpy()
            image = image.cpu().numpy()
            image_for_display = image_for_display.numpy()
            logits_predict = logits_predict.detach().cpu().numpy()
            sigmoid_predict = sigmoid_predict.detach().cpu().numpy()

            """SUM"""
            # np.append(self.f1_losses, f1_macro(sigmoid_predict, labels_0).mean())
            np.append(self.f1_losses, f1.mean())
            np.append(focal_losses, focal_mean)

            """PRINT"""
            # label = np.array(self.dataset.multilabel_binarizer.inverse_transform(labels_0)[0])
            # pred = np.array(self.dataset.multilabel_binarizer.inverse_transform(sigmoid_predict>0.5)[0])
            # pbar.set_description_str("(E{}-F{}) Stp:{} Label:{} Pred:{} Left:{}".format(int(config.global_steps[fold]), label, pred, left))
            pbar.set_description("Focal:{} F1:{}".format(focal.mean(), f1.mean()))
            # if config.DISPLAY_HISTOGRAM: self.epoch_losses.append(focal.flatten())
            predict_total = np.concatenate((predict_total, sigmoid_predict), axis=0) if predict_total is not None else sigmoid_predict
            label_total = np.concatenate((label_total, labels_0), axis=0) if label_total is not None else labels_0

            """DISPLAY"""
            tensorboardwriter.write_memory(self.writer, "train")
            if config.DISPLAY_VISUALIZATION and batch_index < max(1, config.MODEL_BATCH_SIZE / 32): self.display(config.fold, ids, image, image_for_display, labels_0, sigmoid_predict, focal)

            """CLEAN UP"""
            del ids, image, image_for_display
            del focal, f1, precise, recall, labels_0, logits_predict, sigmoid_predict
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
            if config.DEBUG_TRAISE_GPU: gpu_profile(frame=sys._getframe(), event='line', arg=None)
        del pbar
        """LOSS"""
        f1 = f1_macro(predict_total, label_total).mean()
        tensorboardwriter.write_eval_loss(self.writer, {"FoldFocal/{}".format(config.fold): focal_losses.mean(), "FoldF1/{}".format(config.fold): f1}, config.epoch)
        tensorboardwriter.write_pr_curve(self.writer, label_total, predict_total, config.epoch, config.fold)
        self.epoch_pred = np.concatenate((self.epoch_pred, predict_total), axis=0) if self.epoch_pred is not None else predict_total
        self.epoch_label = np.concatenate((self.epoch_label, label_total), axis=0) if self.epoch_label is not None else label_total
        del predict_total, label_total

        if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
        mean_loss = focal_losses.mean()
        del focal_losses
        self.mean_losses.append(mean_loss)
        if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
        return mean_loss, f1

    def __int__(self):
        return self.mean()

    def mean(self, axis=None):
        # if axis == None: return np.array(list(itertools.chain.from_iterable(self.epoch_losses))).mean()
        # print("WARNING: self.epoch_losse may have different shape according to different shape of loss caused by different batch. Numpy cannot take the mean of it is baches shapes are different.")
        # return np.array(self.epoch_losses).mean(axis)
        return np.array(self.mean_losses).mean()

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
        # tensorboardwriter.write_pr_curve(self.writer, labels, predicteds, config.global_steps[fold], fold)

        for index, (id, transfered, untransfered, label, predicted, loss) in enumerate(zip(ids, transfereds, untransfereds, labels, predicteds, losses)):
            if index != 0: continue

            label = self.binarlizer.inverse_transform(np.expand_dims(np.array(label).astype(np.byte), axis=0))[0]
            predict = self.binarlizer.inverse_transform(np.expand_dims((predicted > 0.5).astype(np.byte), axis=0))[0]

            F = plt.figure()

            plt.subplot(321)
            # print(encode.tensor_to_np_three_channel_without_green(untransfered))
            plt.imshow(encode.tensor_to_np_three_channel_without_green(untransfered), vmin=0, vmax=1)
            plt.title("Image_Real; pred:{}".format(predict))
            plt.grid(False)

            plt.subplot(322)
            plt.imshow(encode.tensor_to_np_three_channel_without_green(transfered), vmin=0, vmax=1)
            plt.title("Image_Trans")
            plt.grid(False)

            plt.subplot(323)
            plt.imshow(encode.tensor_to_np_three_channel_with_green(untransfered), vmin=0, vmax=1)
            plt.title("Mask_Real; label:{}".format(label))
            plt.grid(False)

            plt.subplot(324)
            plt.imshow(encode.tensor_to_np_three_channel_with_green(transfered), vmin=0, vmax=1)
            plt.title("Mask_Trans; f1:{}".format(loss))
            plt.grid(False)
            tensorboardwriter.write_image(self.writer, "e{}-{}-{}".format(config.epoch, fold, id), F, config.epoch)


class HPAPrediction:
    def __init__(self, writer):
        self.thresholds = config.PREDICTION_CHOSEN_THRESHOLD
        self.writer = writer
        self.nets = []
        for fold in range(config.MODEL_FOLD):
            if fold not in config.MODEL_TRAIN_FOLD:
                print("     Junping Fold: #{}".format(fold))
            else:
                print("     Creating Fold: #{}".format(fold))
                net = se_resnext101_32x4d_modified(num_classes=config.TRAIN_NUMCLASS, pretrained='imagenet')

                """ONNX"""
                if config.DISPLAY_SAVE_ONNX and config.DIRECTORY_LOAD: save_onnx(net, (config.MODEL_BATCH_SIZE, 4, config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE), config.DIRECTORY_LOAD + ".onnx")

                if config.TRAIN_GPU_ARG: net = torch.nn.DataParallel(net, device_ids=config.TRAIN_GPU_LIST)
                self.nets.append(cuda(net))
        load_checkpoint_all_fold_without_optimizers(self.nets, config.DIRECTORY_LOAD)

        # self.test_dataset = HPAData(config.DIRECTORY_CSV, load_img_dir=config.DIRECTORY_TEST, img_suffix=config.DIRECTORY_SUFFIX_IMG, load_strategy="test", load_preprocessed_dir=False)
        self.test_dataset = HPAData(config.DIRECTORY_CSV, load_img_dir=config.DIRECTORY_PREPROCESSED_IMG, img_suffix=config.DIRECTORY_PREPROCESSED_SUFFIX_IMG, load_strategy="test", load_preprocessed_dir=True)

        self.run()

    def run(self):
        torch.no_grad()
        """Used for Kaggle submission: predicts and encode all test images"""
        for fold, net in enumerate(self.nets):
            print("Start fold: {}".format(fold))
            # Get params
            target_example = 0  # Snake
            (original_image, prep_img, target_class, file_name_to_export, pretrained_model) = \
                get_params(target_example)

            # Grad cam
            gcv2 = GradCam(net, target_layer=11)
            # Generate cam mask
            cam = gcv2.generate_cam(prep_img, target_class)
            print('Grad cam completed')

            # Guided backprop
            GBP = GuidedBackprop(net)
            # Get gradients
            guided_grads = GBP.generate_gradients(prep_img, target_class)
            print('Guided backpropagation completed')

            # Guided Grad cam
            cam_gb = guided_grad_cam(cam, guided_grads)
            save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')
            grayscale_cam_gb = convert_to_grayscale(cam_gb)
            save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')
            print('Guided grad cam completed')

class HPAPreprocess:
    def __init__(self, calculate=False, expected_img_size=(4, 512, 512)):
        self.expected_img_size = expected_img_size
        self.calculate = calculate  # 6item/s when turn off calculation, 6item/s when turn on, 85item/s when loaded in memory (80 save + 85to_np = 6 load)
        if not os.path.exists(config.DIRECTORY_PREPROCESSED_IMG):
            os.makedirs(config.DIRECTORY_PREPROCESSED_IMG)
        mean, std, std1 = self.run(HPAData(config.DIRECTORY_CSV, load_img_dir=config.DIRECTORY_IMG, img_suffix=".png", load_strategy="train", load_preprocessed_dir=False))
        print("""
        Train Data:
            Mean = {}
            STD  = {}
            STD1 = {}
        """.format(mean, std, std1))
        mean, std, std1 = self.run(HPAData(config.DIRECTORY_CSV, load_img_dir=config.DIRECTORY_TEST, img_suffix=".png", load_strategy="test", load_preprocessed_dir=False))
        print("""
        Test Data:
            Mean = {}
            STD  = {}
            STD1 = {}
        """.format(mean, std, std1))

    def transform_video(self):
        vidcap = cv2.VideoCapture('big_buck_bunny_720p_5mb.mp4')
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite("frame-{}.png".format(count), image)
            success, image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1

    def get_mean(self, dataset, save=False, overwrite=False):
        pbar = tqdm(dataset.id)
        length = len(pbar)
        sum = [0, 0, 0, 0]
        for id in pbar:
            img = dataset.get_load_image_by_id(id).astype(np.uint8)
            if self.expected_img_size != img.size: raise ValueError("Expected image size:{} is not equal to actual image size:{}".format(self.expected_img_size, img.size))
            img_mean = np.stack((img.astype(np.float32).mean(0).mean(0)) / 255.)
            sum = sum + img_mean
            pbar.set_description("{} Sum:[{:.2f},{:.2f},{:.2f},{:.2f}]".format(id, img_mean[0], img_mean[1], img_mean[2], img_mean[3]))
            if not os.path.exists(config.DIRECTORY_PREPROCESSED_IMG + id + ".npy") and save and overwrite:
                np.save(config.DIRECTORY_PREPROCESSED_IMG + id + ".npy", img)
            elif save and not overwrite:
                pbar.set_description("Pass: {}".format(id))
                continue
        mean = sum / length
        print("     Mean = {}".format(mean))
        return mean

    def get_std(self, dataset, mean):
        pbar = tqdm(dataset.id)
        length = len(pbar)
        sum_variance = [0, 0, 0, 0]
        for id in pbar:
            img = dataset.get_load_image_by_id(id).astype(np.uint8)
            if self.expected_img_size != img.size: raise ValueError("Expected image size:{} is not equal to actual image size:{}".format(self.expected_img_size, img.size))
            img_mean = np.stack((img.astype(np.float32).mean(0).mean(0)) / 255.)
            img_variance = (img_mean - mean) ** 2
            sum_variance = sum_variance + img_variance

            pbar.set_description("{} Var:[{:.2f},{:.2f},{:.2f},{:.2f}]".format(id, img_variance[0], img_variance[1], img_variance[2], img_variance[3]))
        std = (sum_variance / length) ** 0.5
        std1 = (sum_variance / (length - 1)) ** 0.5
        print("     STD  = {}".format(std))
        print("     STD1 = {}".format(std1))
        return mean, std, std1

    def normalize(self, dataset, mean, std, save=True, overwrite=False):
        """normalize and save data
        Not recomanded because uint8 can be load faster than float32
        """
        pbar = tqdm(dataset.id)
        length = len(pbar)
        for id in pbar:
            img = (dataset.get_load_image_by_id(id).astype(np.float32) / 225. - mean) / std
            if self.expected_img_size != img.size: raise ValueError("Expected image size:{} is not equal to actual image size:{}".format(self.expected_img_size, img.size))
            pbar.set_description("{}".format(id))
            if not os.path.exists(config.DIRECTORY_PREPROCESSED_IMG + id + ".npy") and save and overwrite:
                np.save(config.DIRECTORY_PREPROCESSED_IMG + id + ".npy", img)
            elif save and not overwrite:
                pbar.set_description("Pass: {}".format(id))
                continue

    def run(self, dataset):
        pbar = tqdm(dataset.id)
        length = len(pbar)
        sum = [0, 0, 0, 0]
        sum_variance = [0, 0, 0, 0]
        mean = [0, 0, 0, 0]
        for id in pbar:
            if os.path.exists(config.DIRECTORY_PREPROCESSED_IMG + id + ".npy") and not self.calculate:
                pbar.set_description("Pass: {}".format(id))
                continue
            img = dataset.get_load_image_by_id(id).astype(np.uint8)
            if self.expected_img_size != img.size: raise ValueError("Expected image size:{} is not equal to actual image size:{}".format(self.expected_img_size, img.size))
            # print(img.shape) # (512, 512, 4)
            if self.calculate:
                img_mean = np.stack((img.astype(np.float32).mean(0).mean(0)) / 255.)
                sum = sum + img_mean
                pbar.set_description("{} Sum:[{:.2f},{:.2f},{:.2f},{:.2f}]".format(id, img_mean[0], img_mean[1], img_mean[2], img_mean[3]))

            if not os.path.exists(config.DIRECTORY_PREPROCESSED_IMG + id + ".npy"): np.save(config.DIRECTORY_PREPROCESSED_IMG + id + ".npy", img)

        if self.calculate:
            mean = sum / length
            print("     Mean = {}".format(mean))
        if self.calculate:
            pbar = tqdm(dataset.id)
            for id in pbar:
                img = dataset.get_load_image_by_id(id).astype(np.uint8)
                if self.expected_img_size != img.size: raise ValueError("Expected image size:{} is not equal to actual image size:{}".format(self.expected_img_size, img.size))
                img_mean = np.stack((img.astype(np.float32).mean(0).mean(0)) / 255.)
                img_variance = (img_mean - mean) ** 2
                sum_variance = sum_variance + img_variance

                pbar.set_description("{} Var:[{:.2f},{:.2f},{:.2f},{:.2f}]".format(id, img_variance[0], img_variance[1], img_variance[2], img_variance[3]))
            std = (sum_variance / length) ** 0.5
            std1 = (sum_variance / (length - 1)) ** 0.5
            print("     STD  = {}".format(std))
            print("     STD1 = {}".format(std1))
            return mean, std, std1
        return 0, 0, 0
