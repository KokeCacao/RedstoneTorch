import itertools
import operator
import os
import sys

import matplotlib as mpl
import numpy as np
import torch
from sklearn import metrics
from torch.nn import BCELoss
from torch.utils import data
from tqdm import tqdm

import config
import tensorboardwriter
from dataset.hpa_dataset import HPAData, train_collate, val_collate
from gpu import gpu_profile
from loss.f1 import f1_macro, Differenciable_F1
from loss.focal import FocalLoss_Sigmoid
from project.hpa_project.hpa_model import se_resnext101_32x4d_modified
from utils import encode, load
from utils.load import save_checkpoint_fold, load_checkpoint_all_fold, save_onnx

if os.environ.get('DISPLAY', '') == '':
    print('WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.')
    mpl.use('Agg')
from matplotlib import pyplot as plt

"""URGENT"""
# TODO: compare batch norm and without batch norm in prediction. Try and compare using threshold 0.01 and 0.1 in submitted CP, try and compare tuning threshold by hand picking, compare result when CP52
# TODO: more aug so that your model can distinguish dark image
# TODO: give minority class more appearance so that it trains faster
# TODO: emsemble augmentation?
# TODO: go 8 times augmentation to generate image and then train
# TODO: search threshold on all fold
# TODO: PyCQA/flake8 to check code
# TODO: change to ResNet50, Xception, Inception ResNet v2 x 5, SEResNext too lag?
# TODO: understand F1-macro and so that you know how to adjust your post processing
# TODO: ensemble with majority voting on stage 1: 0.505 + 0.501 + 0.511 LB: 0.516
# TODO: https://arxiv.org/pdf/1802.10171.pdf supervision
# TODO: data augmentation by sliding, get more data
# TODO: 1,983,191 labels associated. Then I dropped labels with frequency less than 50 decreasing the number of labels to around ~ 1.6M. This way the unique number of labels in my dataset decreased to 3862.
# TODO: download data: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/69984
# TODO: Brian: I do a random dropout on the high labels, removing 60% of the values 0 and 25. https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/71147#419480
# TODO: I tried focal loss + soft F1 and focal loss - log(soft F1). Initially the convergence is faster, but later on I ended up with about the same result. Though, I didn't train the model for a long time, just ~6 hours., I get the same result by focal loss + soft F1. Accelerates convergence from 130 epochs to 30., You can check this https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
# TODO: bigger img method: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/71179#419005
# TODO: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/71179#419005
# TODO: (https://zhuanlan.zhihu.com/p/22252270)Yes, using SGD with cosine annealing schedule. Also used Adadelta to start training, Padam for mid training, and SGD at the end. Then I freeze parts of the model and train the other layers. My current leading model is 2.3M params. Performs great locally, but public LB is 45% lower. (https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/69462#412909)
# TODO: adjust lr  set base lr to 1/3 or 1/4 of max lr.
# TODO: LB probing for threshold adjust
# TODO: My top model on 512x512x3 is similar to gap net. It is a convnet encoder + one gap at the end + dense layers + sigmoid. I've trained it for hundreds of epochs at least. (https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/69955)
# TODO: remove 6f062840-bba9-11e8-b2ba-ac1f6b6435d0 from dataset since channel shifted

"""SCHEDULE"""
# TODO: train without tta before submit
# TODO: train with small constant lr before submit
# TODO: LB probing threshold is better or fold best threshold is better? test it in CV and !

""""TODO"""
# TODO: cosine (https://github.com/SeuTao/Kaggle_TGS2018_4th_solution/blob/master/loss/cyclic_lr.py) change to AdamW
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
# TODO: train using predicted label
# TODO: reproducability https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#reproducibility
# NON DETERMINISTIC ISSUE: https://www.zhihu.com/question/67209417/answer/534749835
# class MyUpsample2(nn.Module):
#     def forward(self, x):
#         return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2)

"""ASSUMPTIONS
1. How well can you know a cell's structure by looking at 1 or 2 of 3 images
2. Largest class have huge surface area covered?
3. The green filter is only the location of ONE protein
4. There are 27 cell types, would it be reasonable to assume that the location of a specific protein in [cell-type]-[cell1] is the same as [cell-type]-[cell2]

"""

""""TESTINGS"""
# TODO: fix display image (color and tag)
# TODO: fix memory leak and memory display

""""READINGS"""
# TODO: https://www.proteinatlas.org/learn/dictionary/cell/microtubule+organizing+center+3; https://www.proteinatlas.org/learn/dictionary/cell
# TODO: For model34 , a signle fold with 7 cycle may cost 6~7h (about 66s/epoch on 1 1080ti).
# TODO: read wechat Alexander Liao's comment
# TODO: your upvote list

"""GIVE UP"""
# TODO: try albumnentation: https://albumentations.readthedocs.io/en/latest/writing_tests.html
# TODO: put image in SSD: https://cloud.google.com/compute/docs/disks/add-persistent-disk#create_disk
# TODO: test visualize your network
# TODO: freeze loaded layer, check if the layers loaded correctly (ie. I want to load as much as I can)
# TODO: strange data: da9c2918-bbbf-11e8-b2bb-ac1f6b6435d0, 209fa196-bbad-11e8-b2ba-ac1f6b6435d0

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


# TODO: add false example
# TODO: try pytorch's lr_schedular
# TODO: make sure all lose input are correct
# TODO: visualize prediction and train of each class
# TODO: stratisfy and weight batch
# TODO: Distribute Train and Validation by label
# TODO: try set f1 to 0 when 0/0; (7 missing classes in LB) / (28 total classes) = 0.25, and if the organizer is interpreting 0/0 as 0
# TODO: Ask your biology teacher about yellow channel
# TODO: try a better GPU
# TODO: fix image display or augmentation
# TODO: normalize and stratisfy training data into folds
# TODO: Data pre processing - try normalize data mean and std (https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/18) Save as ".npy" with dtype = "uint8". Before augmentation, convert back to float32 and normalize them with dataset mean/std.
# TODO: Compare BCE and focal
# TODO: focal loss with gamma=2
# TODO: normalize data using all data
class HPATrain:
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
                optimizer = torch.optim.Adadelta(params=net.parameters(), lr=config.MODEL_INIT_LEARNING_RATE, rho=0.9, eps=1e-6, weight_decay=config.MODEL_WEIGHT_DEFAY)
                self.optimizers.append(optimizer)
                self.nets.append(net)
                # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4*int(27964.8/config.MODEL_BATCH_SIZE), verbose=False, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=False, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
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
        for fold, (net, optimizer, lr_scheduler) in enumerate(zip(nets, optimizers, lr_schedulers)):
            # import pdb; pdb.set_trace() #1357Mb -> 1215Mb
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
            optimizer = load.move_optimizer_to_cpu(optimizer) #3299Mb
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache() #1215Mb

        """DISPLAY"""
        best_id, best_loss = evaluation.best()
        worst_id, worst_loss = evaluation.worst()
        for fold, (best_id, best_loss, worst_id, worst_loss) in enumerate(zip(best_id, best_loss, worst_id, worst_loss)):
            best_img = self.dataset.get_load_image_by_id(best_id)
            best_label = self.dataset.multilabel_binarizer.inverse_transform(np.expand_dims(self.dataset.get_load_label_by_id(best_id), axis=0))[0]
            worst_img = self.dataset.get_load_image_by_id(worst_id)
            worst_label = self.dataset.multilabel_binarizer.inverse_transform(np.expand_dims(self.dataset.get_load_label_by_id(worst_id), axis=0))[0]
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

        print("Set Model Trainning mode to trainning=[{}]".format(net.train().training))
        pbar = tqdm(train_loader)
        train_len = len(train_loader) + 1e-10
        for batch_index, (ids, image, labels_0, image_for_display) in enumerate(pbar):
            #1215MB -> 4997MB = 3782

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
            positive_bce = BCELoss(weight=labels_0*20+1)(sigmoid_predict, labels_0)
            # [1801.5 / 12885, 1801.5 / 1254, 1801.5 / 3621, 1801.5 / 1561, 1801.5 / 1858, 1801.5 / 2513, 1801.5 / 1008, 1801.5 / 2822, 1801.5 / 53, 1801.5 / 45, 1801.5 / 28, 1801.5 / 1093, 1801.5 / 688, 1801.5 / 537, 1801.5 / 1066, 1801.5 / 21, 1801.5 / 530, 1801.5 / 210, 1801.5 / 902, 1801.5 / 1482, 1801.5 / 172, 1801.5 / 3777, 1801.5 / 802, 1801.5 / 2965, 1801.5 / 322, 1801.5 / 8228, 1801.5 / 328, 1801.5 / 11] / (1801.5 / 11)
            weighted_bce = BCELoss(weight=(torch.Tensor(
                [8.53705860e-04, 8.77192982e-03, 3.03783485e-03, 7.04676489e-03,
                 5.92034446e-03, 4.37723836e-03, 1.09126984e-02, 3.89794472e-03,
                 2.07547170e-01, 2.44444444e-01, 3.92857143e-01, 1.00640439e-02,
                 1.59883721e-02, 2.04841713e-02, 1.03189493e-02, 5.23809524e-01,
                 2.07547170e-02, 5.23809524e-02, 1.21951220e-02, 7.42240216e-03,
                 6.39534884e-02, 2.91236431e-03, 1.37157107e-02, 3.70994941e-03,
                 3.41614907e-02, 1.33689840e-03, 3.35365854e-02, 1.00000000e+00]
            )*5.00000000e01).cuda())(sigmoid_predict, labels_0)
            if config.epoch < 10:
                loss = bce
            else:
                loss = f1 + weighted_bce
            if config.epoch == 10 and batch_index == 0: tensorboardwriter.write_text(self.writer, "Switch to f1 at epoch={}".format(config.epoch), config.global_steps[fold])
            """BACKPROP"""
            # lr_scheduler.step(f1.detach().cpu().numpy().mean(), epoch=config.global_steps[fold])
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
        lr_scheduler.step(epoch_f1, epoch=config.epoch)

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

        print("Set Model Trainning mode to trainning=[{}]".format(net.eval().training))
        for eval_index in range(4): # TODO: set to range(8)
            config.eval_index = eval_index
            pbar = tqdm(validation_loader)
            for batch_index, (ids, image, labels_0, image_for_display) in enumerate(pbar):
                """CALCULATE LOSS"""
                if config.TRAIN_GPU_ARG:
                    image = image.cuda()
                    labels_0 = labels_0.cuda()
                logits_predict = net(image)
                if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
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
                focal_min = focal.min().item()
                focal_min_id = (focal == focal_min).nonzero()
                focal_min_id = focal_min_id.view(focal_min_id.size(), -1)[0]
                focal_min_id = ids[focal_min_id.cpu().numpy()[0]]
                focal_max = focal.max().item()
                focal_max_id = (focal == focal_max).nonzero()
                focal_max_id = focal_max_id.view(focal_max_id.size(), -1)[0]
                focal_max_id = ids[focal_max_id.cpu().numpy()[0]]
                self.best_loss = np.append(self.best_loss, focal_min)
                self.worst_loss = np.append(self.worst_loss, focal_max)
                self.best_id = np.append(self.best_id, focal_min_id)
                self.worst_id = np.append(self.worst_id, focal_max_id)
                del focal_min, focal_min_id, focal_max, focal_max_id
                if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()

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
                self.f1_losses = np.append(self.f1_losses, f1.mean())
                focal_losses = np.append(focal_losses, focal_mean)

                """PRINT"""
                # label = np.array(self.dataset.multilabel_binarizer.inverse_transform(labels_0)[0])
                # pred = np.array(self.dataset.multilabel_binarizer.inverse_transform(sigmoid_predict>0.5)[0])
                # pbar.set_description_str("(E{}-F{}) Stp:{} Label:{} Pred:{} Left:{}".format(int(config.global_steps[fold]), label, pred, left))
                pbar.set_description("(E{}F{}I{}) Focal:{} F1:{}".format(config.epoch, config.fold, config.eval_index, focal_mean, f1.mean()))
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
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
        """LOSS"""
        f1 = f1_macro(predict_total, label_total).mean()
        tensorboardwriter.write_eval_loss(self.writer, {"FoldFocal/{}".format(config.fold): focal_losses.mean(), "FoldF1/{}".format(config.fold): f1}, config.epoch)
        tensorboardwriter.write_pr_curve(self.writer, label_total, predict_total, config.epoch, config.fold)
        self.epoch_pred = np.concatenate((self.epoch_pred, predict_total), axis=0) if self.epoch_pred is not None else predict_total
        self.epoch_label = np.concatenate((self.epoch_label, label_total), axis=0) if self.epoch_label is not None else label_total

        if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
        mean_loss = focal_losses.mean()
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


