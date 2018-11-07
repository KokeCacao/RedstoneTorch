import itertools
import os
import sys

import matplotlib as mpl
import numpy as np
import torch
import pandas as pd
from sklearn import metrics
from torch.nn import BCELoss
from torch.utils import data
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image

import config
import tensorboardwriter
from dataset.hpa_dataset import HPAData, train_collate, val_collate, transform
from gpu import gpu_profile
from loss.f1 import f1_macro, Differenciable_F1
from loss.focal import Focal_Loss_from_git
from net.proteinet.proteinet_model import se_resnext101_32x4d_modified
from utils import encode
from utils.load import save_checkpoint_fold, load_checkpoint_all_fold, cuda, load_checkpoint_all_fold_without_optimizers, save_onnx

if os.environ.get('DISPLAY', '') == '':
    print('WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.')
    mpl.use('Agg')
from matplotlib import pyplot as plt


class HPAProject:
    """"."""

    """URGENT"""
    # TODO: (https://zhuanlan.zhihu.com/p/22252270)Yes, using SGD with cosine annealing schedule. Also used Adadelta to start training, Padam for mid training, and SGD at the end. Then I freeze parts of the model and train the other layers. My current leading model is 2.3M params. Performs great locally, but public LB is 45% lower. (https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/69462#412909)
    # TODO: FIx tensorboardwriter.write_eval_loss(self.writer, {"EvalFocalMean": evaluation.mean(), "EvalFocalSTD": evaluation.std()}, config.epoch)
    # TODO: fix image display or augmentation
    # TODO: adjust lr  set base lr to 1/3 or 1/4 of max lr.
    # TODO: LB probing for threshold adjust
    # TODO: My top model on 512x512x3 is similar to gap net. It is a convnet encoder + one gap at the end + dense layers + sigmoid. I've trained it for hundreds of epochs at least. (https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/69955)
    # TODO: Compare BCE and focal
    # TODO: focal loss with gamma=2
    # TODO: put image in SSD: https://cloud.google.com/compute/docs/disks/add-persistent-disk#create_disk

    """"TODO"""
    # TODO: Data pre processing - try normalize data mean and std (https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/18) Save as ".npy" with dtype = "uint8". Before augmentation, convert back to float32 and normalize them with dataset mean/std.
    # TODO: Ask your biology teacher about yellow channel
    # TODO: cosine (https://github.com/SeuTao/Kaggle_TGS2018_4th_solution/blob/master/loss/cyclic_lr.py) change to AdamW
    # TODO: try set f1 to 0 when 0/0; (7 missing classes in LB) / (28 total classes) = 0.25, and if the organizer is interpreting 0/0 as 0
    # TODO: try to process RBY first, and then concat Green layer
    # TODO: Zero padding in augmentation
    # TODO: Better augmentation
    # TODO: Adjust weight init (in or out) and init dense layers
    # TODO: I tried focal loss + soft F1 and focal loss - log(soft F1). Initially the convergence is faster, but later on I ended up with about the same result. Though, I didn't train the model for a long time, just ~6 hours.
    # TODO: load faster, try not to transform test, try adding CPU
    # TODO: mixup https://arxiv.org/pdf/1710.09412.pdf
    # TODO: attention Residual Attention Network for Image Classification - Fei Wang, cvpr 2017 https://arxiv.org/abs/1704.06904 https://www.youtube.com/watch?v=Deq1BGTHIPA, https://blog.csdn.net/wspba/article/details/73727469
    # TODO: train on 1028*1028 image
    # TODO: visualization: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/70173
    # TODO: try a better GPI
    # TODO: train using predicted label
    # TODO: Distribute Train and Validation by label

    """"TESTINGS"""
    # TODO: fix display image (color and tag)
    # TODO: fix memory leak and memory display

    """"READINGS"""
    # TODO: https://www.proteinatlas.org/learn/dictionary/cell/microtubule+organizing+center+3; https://www.proteinatlas.org/learn/dictionary/cell
    # TODO: For model34 , a signle fold with 7 cycle may cost 6~7h (about 66s/epoch on 1 1080ti).
    # TODO: read wechat Alexander Liao's comment
    # TODO: your upvote list

    """GIVE UP"""
    # TODO: test visualize your network
    # TODO: freeze loaded layer, check if the layers loaded correctly (ie. I want to load as much as I can)

    """LB PROBING
    you can actually verify the correct metric by probing the LB.
    submit "prediction= a single class A" for all images
    submit "prediction= a single class B" for all images
    submit "prediction= 2 classes A and B" for all images
    if the F1 is averaged over the class, you can derived 3. from 1.,2.
    e.g.
    say for class-A: public LB = 0.01, then F1 for class A = 28*0.01
    and for class-B: public LB = 0.02, then F1 for class B = 28*0.02,
    then for class-A and B , i should public LB = ( (280.01) + (280.02) + 0 + 0 )/28
    """

    """Note"""
    # Increase Batch size tend to overfit
    # SGD is great, no overfit, but slow
    # increase batch size by a, -> increase lr by a
    # adjust dropout not overfit (first layer)

    """FINISHED"""

    def __init__(self, writer):
        self.writer = writer

        self.optimizers = []
        self.nets = []
        for fold in range(config.MODEL_FOLD):
            if fold not in config.MODEL_TRAIN_FOLD:
                print("     Junping Fold: #{}".format(fold))
            else:
                print("     Creating Fold: #{}".format(fold))
                net = se_resnext101_32x4d_modified(num_classes=config.TRAIN_NUMCLASS, pretrained='imagenet')
                if config.TRAIN_GPU_ARG: net = torch.nn.DataParallel(net, device_ids=config.TRAIN_GPU_LIST)

                # self.optimizers.append(torch.optim.Adam(params=net.parameters(), lr=config.MODEL_INIT_LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.MODEL_WEIGHT_DEFAY))
                self.optimizers.append(torch.optim.Adadelta(params=net.parameters(), lr=config.MODEL_INIT_LEARNING_RATE, rho=0.9, eps=1e-6, weight_decay=config.MODEL_WEIGHT_DEFAY))
                net = cuda(net)
                self.nets.append(net)
                # for name, param in net.named_parameters():
                #     if param.requires_grad:
                #         print (name)
        load_checkpoint_all_fold(self.nets, self.optimizers, config.DIRECTORY_LOAD)
        if config.DISPLAY_SAVE_ONNX and config.DIRECTORY_LOAD: save_onnx(self.nets[0], (config.MODEL_BATCH_SIZE, 4, config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE), config.DIRECTORY_LOAD + ".onnx")

        self.dataset = HPAData(config.DIRECTORY_CSV, load_img_dir=config.DIRECTORY_IMG, img_suffix = config.DIRECTORY_PREPROCESSED_SUFFIX_IMG, load_preprocessed_dir=config.DIRECTORY_PREPROCESSED_IMG)
        self.folded_samplers = self.dataset.get_fold_sampler(fold=config.MODEL_FOLD)

        self.run()

    def run(self):
        try:
            for epoch in range(config.MODEL_EPOCHS):
                self.step_epoch(nets=self.nets,
                                optimizers=self.optimizers,
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
                   batch_size
                   ):
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

        evaluation = HPAEvaluation(self.writer, self.dataset.multilabel_binarizer)
        for fold, (net, optimizer) in enumerate(zip(nets, optimizers)):
            self.step_fold(fold, net, optimizer, batch_size, evaluation)
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()

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
        f2 = metrics.f1_score((evaluation.epoch_label > 0.5).astype(np.int16), (evaluation.epoch_pred > 0.5).astype(np.int16), average='macro')  # sklearn does not automatically import matrics.
        print("F1 by sklearn = ".format(f2))
        tensorboardwriter.write_epoch_loss(self.writer, {"EvalF1": f1}, config.epoch)
        tensorboardwriter.write_pred_distribution(self.writer, evaluation.epoch_pred.flatten(), config.epoch)

        """THRESHOLD"""
        if config.EVAL_IF_THRESHOLD_TEST:
            best_threshold = 0.0
            best_val = 0.0
            pbar = tqdm(config.EVAL_TRY_THRESHOLD)
            for threshold in pbar:
                score = f1_macro(evaluation.epoch_pred, evaluation.epoch_label, thresh=threshold).mean()
                tensorboardwriter.write_threshold(self.writer, {"Fold/{}".format(config.fold): score}, threshold*1000.0)
                if score > best_val:
                    best_threshold = threshold
                    best_val = score
                pbar.set_description("Threshold: {}; F1: {}".format(threshold, score))
            print("BestThreshold: {}, F1: {}".format(best_threshold, best_val))
            tensorboardwriter.write_best_threshold(self.writer, best_threshold, best_val, config.epoch)

        """HISTOGRAM"""
        if config.DISPLAY_HISTOGRAM:
            tensorboardwriter.write_eval_loss(self.writer, {"EvalFocalMean": evaluation.mean(), "EvalFocalSTD": evaluation.std()}, config.epoch)
            tensorboardwriter.write_loss_distribution(self.writer, np.array(list(itertools.chain.from_iterable(evaluation.epoch_losses))).flatten(), config.epoch)

        """CLEAN UP"""
        del evaluation

    def step_fold(self, fold, net, optimizer, batch_size, evaluation):
        config.fold = fold

        epoch_loss = 0
        epoch_f1 = 0

        train_loader = data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.folded_samplers[config.fold]["train"], shuffle=False, num_workers=config.TRAIN_NUM_WORKER, collate_fn=train_collate)
        pbar = tqdm(train_loader)
        train_len = len(train_loader)

        for batch_index, (ids, image, labels_0, image_for_display) in enumerate(pbar):
            """UPDATE LR"""
            optimizer.state['lr'] = config.TRAIN_TRY_LR_FORMULA(config.global_steps[fold]) if config.TRAIN_TRY_LR else config.TRAIN_COSINE(config.global_steps[fold])

            """TRAIN NET"""
            config.global_steps[fold] = config.global_steps[fold] + 1
            if config.TRAIN_GPU_ARG:
                image = image.cuda()
                labels_0 = labels_0.cuda()
            predict = net(image)

            """LOSS"""
            focal = Focal_Loss_from_git(alpha=0.25, gamma=2, eps=1e-7)(labels_0, predict)
            f1 = Differenciable_F1()(labels_0, predict)
            weighted_bce = BCELoss(weight=torch.Tensor([1801.5/12885, 1801.5/1254, 1801.5/3621, 1801.5/1561, 1801.5/1858, 1801.5/2513, 1801.5/1008, 1801.5/2822, 1801.5/53, 1801.5/45, 1801.5/28, 1801.5/1093, 1801.5/688, 1801.5/537, 1801.5/1066, 1801.5/21, 1801.5/530, 1801.5/210, 1801.5/902, 1801.5/1482, 1801.5/172, 1801.5/3777, 1801.5/802, 1801.5/2965, 1801.5/322, 1801.5/8228, 1801.5/328, 1801.5/11]).cuda())(predict, labels_0)
            loss = focal.sum() + f1 + weighted_bce.sum()
            """BACKPROP"""
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """DETATCH"""
            focal = focal.detach().cpu().numpy().mean()
            f1 = f1.detach().cpu().numpy().mean()
            weighted_bce = weighted_bce.detach().cpu().numpy().mean()
            loss = loss.detach().cpu().numpy().mean()
            labels_0 = labels_0.cpu().numpy()

            """SUM"""
            epoch_loss = epoch_loss + loss.mean()
            epoch_f1 = epoch_f1 + f1.mean()
            # f1 = f1_macro(predict, labels_0).mean()

            """DISPLAY"""
            tensorboardwriter.write_memory(self.writer, "train")
            pbar.set_description("(E{}-F{}) Step:{} Focal:{:.4f} F1:{:.4f} lr:{:.4E} BCE:{:.2f}".format(config.epoch, config.fold, int(config.global_steps[fold]), focal, f1, optimizer.state['lr'], weighted_bce))
            tensorboardwriter.write_loss(self.writer, {'Epoch/{}'.format(config.fold): config.epoch, 'Loss/{}'.format(config.fold): loss, 'F1/{}'.format(config.fold): f1, 'Focal/{}'.format(config.fold): focal}, config.global_steps[fold])

            """CLEAN UP"""
            del ids, image, labels_0, image_for_display
            del predict, loss, focal, f1
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()  # release gpu memory
        del train_loader, pbar

        val_loss, val_f1 = evaluation.eval_fold(net, data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.folded_samplers[config.fold]["val"], shuffle=False, num_workers=config.TRAIN_NUM_WORKER, collate_fn=val_collate))
        train_loss = epoch_loss / train_len
        print("""
            Epoch: {}, Fold: {}
            TrainLoss: {}, TrainF1: 
            ValidLoss: {}, ValidF1: {}
        """.format(config.epoch, config.fold, train_loss, val_loss, val_f1))

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
        fold_loss_dict = dict()
        predict_total = None
        label_total = None

        self.best_id = np.array([])
        self.worst_id = np.array([])
        self.best_loss = np.array([])
        self.worst_loss = np.array([])

        pbar = tqdm(validation_loader)
        for batch_index, (ids, image, labels_0, image_for_display) in enumerate(pbar):
            """CALCULATE LOSS"""
            if config.TRAIN_GPU_ARG:
                image = image.cuda()
                labels_0 = labels_0.cuda()
            predict = net(image)

            """LOSS"""
            focal = Focal_Loss_from_git(alpha=0.25, gamma=2, eps=1e-7)(labels_0, predict)
            f1 = Differenciable_F1()(labels_0, predict)

            """DETATCH"""
            focal = focal.detach().cpu().numpy()
            f1 = f1.detach().cpu().numpy()
            labels_0 = labels_0.cpu().numpy()

            """SUM"""
            # np.append(self.f1_losses, f1_macro(predict, labels_0).mean())
            np.append(self.f1_losses, f1.mean())

            """PRINT"""
            predict = F.softmax(predict, dim=1)
            pbar.set_description("Focal:{} F1:{}".format(focal.mean(), f1.mean()))
            if config.DISPLAY_HISTOGRAM: self.epoch_losses.append(focal.flatten())
            for id, loss_item in zip(ids, focal.flatten()): fold_loss_dict[id] = loss_item
            predict_total = np.concatenate((predict_total, predict.detach().cpu().numpy()), axis=0) if predict_total is not None else predict.detach().cpu().numpy()
            label_total = np.concatenate((label_total, labels_0), axis=0) if label_total is not None else labels_0

            """EVALUATE LOSS"""
            min_loss = min(fold_loss_dict.values())
            min_key = min(fold_loss_dict, key=fold_loss_dict.get)
            np.append(self.best_loss, min_loss)
            np.append(self.best_id, min_key)
            max_loss = max(fold_loss_dict.values())
            max_key = max(fold_loss_dict, key=fold_loss_dict.get)
            np.append(self.worst_loss, max_loss)
            np.append(self.worst_id, max_key)

            """DISPLAY"""
            tensorboardwriter.write_memory(self.writer, "train")
            if config.DISPLAY_VISUALIZATION and batch_index < 5: self.display(config.fold, ids, image, image_for_display, labels_0, predict, focal)

            """CLEAN UP"""
            del ids, image, labels_0, image_for_display
            del predict, focal
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
            if config.DEBUG_TRAISE_GPU: gpu_profile(frame=sys._getframe(), event='line', arg=None)
        del pbar
        """LOSS"""
        f1 = f1_macro(predict_total, label_total).mean()
        tensorboardwriter.write_eval_loss(self.writer, {"FoldLoss/{}".format(config.fold): np.array(fold_loss_dict.values()).mean(), "FoldF1/{}".format(config.fold): f1}, config.epoch)
        tensorboardwriter.write_pr_curve(self.writer, label_total, predict_total, config.epoch, config.fold)
        self.epoch_pred = np.concatenate((self.epoch_pred, predict_total), axis=0) if self.epoch_pred is not None else predict_total
        self.epoch_label = np.concatenate((self.epoch_label, label_total), axis=0) if self.epoch_label is not None else label_total
        del predict_total, label_total

        # self.epoch_dict = np.concatenate((self.epoch_dict, [fold_loss_dict]), axis=0)

        if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
        mean_loss = np.array(fold_loss_dict.values()).mean()
        del fold_loss_dict
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
            label = self.binarlizer.inverse_transform(label)

            F = plt.figure()

            plt.subplot(321)
            plt.imshow(encode.tensor_to_np_three_channel_without_green(untransfered))
            plt.title("Image_Real")
            plt.grid(False)

            plt.subplot(322)
            plt.imshow(encode.tensor_to_np_three_channel_without_green(transfered))
            plt.title("Image_Trans")
            plt.grid(False)

            plt.subplot(323)
            plt.imshow(encode.tensor_to_np_three_channel_with_green(untransfered), norm=mpl.colors.NoNorm(vmin=0, vmax=1, clip=True))
            plt.title("Mask_Real; label:{}".format(label))
            plt.grid(False)

            plt.subplot(324)
            plt.imshow(encode.tensor_to_np_three_channel_with_green(transfered))
            plt.title("Mask_Trans; loss:{}".format(loss))
            plt.grid(False)
            tensorboardwriter.write_image(self.writer, "{}-{}".format(fold, id), F, config.epoch)


class HPAPrediction:
    def __init__(self, writer):
        self.threshold = config.EVAL_CHOSEN_THRESHOLD
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
            # for index, net in enumerate(self.nets):
            #     save_onnx(net, (config.MODEL_BATCH_SIZE, 4, config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE), config.DIRECTORY_LOAD + "-" + str(index) + ".onnx")

        self.dataset = HPAData(config.DIRECTORY_CSV, config.DIRECTORY_IMG, img_suffix=".png", test=True, load_preprocessed_dir=None)

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
                    predict = net(input)
                    predict = F.softmax(predict, dim=1)
                    predict = (predict.detach().cpu().numpy() > self.threshold).astype(np.int16)
                    encoded = self.dataset.multilabel_binarizer.inverse_transform(predict)
                    encoded = list(encoded[0])

                    f.write('{},{}\n'.format(id, " ".join(str(x) for x in encoded)))
                    pbar.set_description("Fold: {}; Id: {}; Out: {}".format(fold, id, encoded))
                    del id, input, predict, encoded
                    if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()

            """ORGANIZE"""
            f1 = pd.read_csv(config.DIRECTORY_SAMPLE_CSV)
            f1.drop('Predicted', axis=1, inplace=True)
            f2 = pd.read_csv(save_path)
            f1 = f1.merge(f2, left_on='Id', right_on='Id', how='outer')
            os.remove(save_path)
            f1.to_csv(save_path, index=False)


class HPAPreprocess:
    def __init__(self, calculate=False):
        self.calculate = calculate # 6item/s when turn off calculation, 6item/s when turn on, 85item/s when loaded in memory (80 save + 85to_np = 6 load)
        if not os.path.exists(config.DIRECTORY_PREPROCESSED_IMG):
            os.makedirs(config.DIRECTORY_PREPROCESSED_IMG)
        mean, std, std1 = self.run(HPAData(config.DIRECTORY_CSV, load_img_dir=config.DIRECTORY_IMG, img_suffix=".png", test=False, load_preprocessed_dir=None))
        print("""
        Train Data:
            Mean = {}
            STD  = {}
            STD1 = {}
        """.format(mean, std, std1))
        """
        Train Data:
            Mean = [0.0804419  0.05262986 0.05474701 0.08270896]
            STD  = [0.0025557  0.0023054  0.0012995  0.00293925]
            STD1 = [0.00255578 0.00230547 0.00129955 0.00293934]
        """
        mean, std, std1 = self.run(HPAData(config.DIRECTORY_CSV, load_img_dir=config.DIRECTORY_IMG, img_suffix=".png", test=True, load_preprocessed_dir=None))
        print("""
        Test Data:
            Mean = {}
            STD  = {}
            STD1 = {}
        """.format(mean, std, std1))
        """
        Test Data:
            Mean = [0.05908022 0.04532852 0.04065233 0.05923426]
            STD  = [0.00235361 0.0020402  0.00137821 0.00246495]
            STD1 = [0.00235381 0.00204037 0.00137833 0.00246516]
        """


        """ https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/69462
        Hi lafoss, 
        just out of interest: How did you calculate these values? I am asking because I did the same a couple of days ago, on the original 512x512 images and got slightly different results, i.e.:
        Means for train image data (originals)

        Red average: 0.080441904331346
        Green average: 0.05262986230955176
        Blue average: 0.05474700710311806
        Yellow average: 0.08270895676048498

        Means for test image data (originals)

        Red average: 0.05908022413399168
        Green average: 0.04532851916280794
        Blue average: 0.040652325092460015
        Yellow average: 0.05923425759572161

        Did you resize the images before checking the means? 
        As I say, just out of interest, 
        cheers and thanks, 
        Wolfgang
        """

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
            # print(img.shape) # (512, 512, 4)
            if self.calculate:
                img_mean = np.stack((img.astype(np.float32).mean(0).mean(0))/255.)
                sum = sum + img_mean
                pbar.set_description("{} Sum:[{:.2f},{:.2f},{:.2f},{:.2f}]".format(id, img_mean[0], img_mean[1], img_mean[2], img_mean[3]))

            if not os.path.exists(config.DIRECTORY_PREPROCESSED_IMG + id + ".npy"): np.save(config.DIRECTORY_PREPROCESSED_IMG + id + ".npy", img)
        if self.calculate:
            mean = sum/length
            print("     Mean = {}".format(mean))
        if self.calculate:
            pbar = tqdm(dataset.id)
            for id in pbar:
                img = dataset.get_load_image_by_id(id).astype(np.uint8)
                img_mean = np.stack((img.astype(np.float32).mean(0).mean(0))/255.)
                img_variance = (img_mean - mean)**2
                sum_variance = sum_variance + img_variance

                pbar.set_description("{} Var:[{:.2f},{:.2f},{:.2f},{:.2f}]".format(id, img_variance[0], img_variance[1], img_variance[2], img_variance[3]))
            std = sum_variance/length
            std1 = sum_variance/(length-1)
            print("     STD  = {}".format(std))
            print("     STD1 = {}".format(std1))
            return mean, std, std1
        return 0, 0, 0




