import itertools
import operator
import os
import sys

import matplotlib as mpl
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import classification_report
from torch.utils import data
from tqdm import tqdm

import config
import tensorboardwriter
from dataset.HisCancer_dataset import HisCancerDataset
from dataset.HisCancer_dataset import train_collate, val_collate
from gpu import gpu_profile
from loss.f import f1_macro, differenciable_f_softmax
from loss.focal import focalloss_softmax
from lr_scheduler.PlateauCyclicRestart import PlateauCyclicRestart
from project.HisCancer_project import HisCancer_net
from utils import load
from utils.load import save_checkpoint_fold, load_checkpoint_all_fold, save_onnx, remove_checkpoint_fold, set_milestone
from utils.lr_finder import LRFinder

if os.environ.get('DISPLAY', '') == '':
    print('WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.')
    mpl.use('Agg')


class HisCancerValidation:
    def __init__(self, writer):
        self.writer = writer

        self.optimizers = []
        self.nets = []
        self.lr_schedulers = []
        self.train_loader = []
        self.validation_loader = []

        self.dataset = HisCancerDataset(config.DIRECTORY_CSV, config.DIRECTORY_SAMPLE_CSV, load_strategy="train", writer=self.writer, column='Target')
        self.folded_samplers = self.dataset.get_wsl_samples(fold=config.MODEL_FOLD)

        for fold in range(config.MODEL_FOLD):
            if fold not in config.train_fold:
                print("     Skipping dataset = HisCancerDataset(config.DIRECTORY_CSV, fold: #{})".format(fold))
                self.optimizers.append(None)
                self.nets.append(None)
                self.lr_schedulers.append(None)
            else:
                print("     Creating Fold: #{}".format(fold))
                net = HisCancer_net.se_resnext50_32x4d(config.TRAIN_NUM_CLASS, pretrained="imagenet", dropout_p=0.9)

                """FREEZING LAYER"""
                for i, c in enumerate(net.children()):
                    if len(config.MODEL_NO_GRAD) > i:
                        l = config.MODEL_NO_GRAD[i]
                        for child_counter, child in enumerate(list(c.children())):
                            if child_counter in l or l == [-1]:
                                print("Disable Gradient for child_counter: {}".format(child_counter))
                                for paras in child.parameters():
                                    paras.requires_grad = False
                            else:
                                print("Enable Gradient for child_counter: {}".format(child_counter))
                    else:
                        print("Enable Gradient for layer: {} (default)".format(i))

                if config.TRAIN_GPU_ARG: net = torch.nn.DataParallel(net, device_ids=config.TRAIN_GPU_LIST)
                optimizer = torch.optim.SGD(params=net.parameters(), lr=config.MODEL_INIT_LEARNING_RATE, momentum=config.MODEL_MOMENTUM, weight_decay=config.MODEL_WEIGHT_DECAY)
                self.optimizers.append(optimizer)
                self.nets.append(net)
                self.lr_schedulers.append(PlateauCyclicRestart(optimizer, eval_mode='max', factor=0.2, patience=1, verbose=False, threshold=1e-4, threshold_mode='abs', cooldown=0, eps=1e-8, base_lr=0.0023, max_lr=0.0069, step_size=2705, mode='plateau_cyclic', gamma=1., scale_mode='cycle', last_batch_iteration=-1, reduce_restart=3))

            self.train_loader.append(data.DataLoader(self.dataset,
                                           batch_size=config.MODEL_BATCH_SIZE,
                                           shuffle=False,
                                           sampler=self.folded_samplers[fold]["train"],
                                           batch_sampler=None,
                                           num_workers=config.TRAIN_NUM_WORKER,
                                           collate_fn=train_collate,
                                           pin_memory=True,
                                           drop_last=False,
                                           timeout=0,
                                           worker_init_fn=None,
                                           ))
            self.validation_loader.append(data.DataLoader(self.dataset,
                                                             batch_size=config.MODEL_BATCH_SIZE,
                                                             shuffle=False,
                                                             sampler=self.folded_samplers[fold]["val"],
                                                             batch_sampler=None,
                                                             num_workers=config.TRAIN_NUM_WORKER,
                                                             collate_fn=val_collate,
                                                             pin_memory=False,
                                                             drop_last=False,
                                                             timeout=0,
                                                             worker_init_fn=None,
                                                             ))
        if config.TRAIN_LOAD_OPTIMIZER: load_checkpoint_all_fold(self.nets, self.optimizers, self.lr_schedulers, config.DIRECTORY_LOAD)
        set_milestone(config.DIRECTORY_LOAD)

        """RESET LR"""
        if config.resetlr != 0:
            print("Reset Learning Rate to {}".format(config.resetlr))
            for optim in self.optimizers:
                if optim == None:
                    continue
                for g in optim.param_groups:
                    g['lr'] = config.resetlr

        if config.DISPLAY_SAVE_ONNX and config.DIRECTORY_LOAD: save_onnx(self.nets[config.train_fold[0]], (config.MODEL_BATCH_SIZE, 4, config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE), config.DIRECTORY_LOAD + ".onnx")

        if config.DEBUG_LR_FINDER:
            val_loader = data.DataLoader(self.dataset,
                                         batch_size=config.MODEL_BATCH_SIZE,
                                         shuffle=False,
                                         sampler=self.folded_samplers[config.train_fold[0]]["val"],
                                         batch_sampler=None,
                                         num_workers=config.TRAIN_NUM_WORKER,
                                         collate_fn=val_collate,
                                         pin_memory=False,
                                         drop_last=False,
                                         timeout=0,
                                         worker_init_fn=None,
                                         ) if config.FIND_LR_ON_VALIDATION else None
            lr_finder = LRFinder(self.nets[config.train_fold[0]], self.optimizers[config.train_fold[0]], torch.nn.BCEWithLogitsLoss(), writer=self.writer, device="cuda")
            lr_finder.range_test(data.DataLoader(self.dataset,
                                                 batch_size=config.MODEL_BATCH_SIZE,
                                                 shuffle=False,
                                                 sampler=self.folded_samplers[config.train_fold[0]]["train"],
                                                 batch_sampler=None,
                                                 num_workers=config.TRAIN_NUM_WORKER,
                                                 collate_fn=train_collate,
                                                 pin_memory=True,
                                                 drop_last=False,
                                                 timeout=0,
                                                 worker_init_fn=None,
                                                 ), val_loader=val_loader, end_lr=0.1, num_iter=config.FIND_LR_RATIO, step_mode="exp")
            tensorboardwriter.write_plot(self.writer, lr_finder.plot(skip_start=0, skip_end=0, log_lr=False), "lr_finder-Linear")
            tensorboardwriter.write_plot(self.writer, lr_finder.plot(skip_start=0, skip_end=0, log_lr=True), "lr_finder-Log")
            lr_finder.reset()
            return

        """FREEZE DETECT"""
        for c in self.nets[config.train_fold[0]].children():
            for child_counter, child in enumerate(c.children()):
                req_grad = True
                for p in child.parameters():
                    if not p.requires_grad:
                        req_grad = False
                print("=======================Start Child Number #{} Grad: {}=======================".format(child_counter, req_grad))
                print("{}".format(child))
                print("=======================End Child Number #{} Grad: {}=======================".format(child_counter, req_grad))

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

            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()

        except KeyboardInterrupt as e:
            print(e)
            self.writer.close()
            print("To Resume: python train.py --versiontag 'test' --projecttag " + config.PROJECT_TAG + " --loadfile " + config.lastsave)
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

        evaluation = HisCancerEvaluation(self.writer, self.dataset.multilabel_binarizer)
        for fold, (net, optimizer, lr_scheduler) in enumerate(zip(nets, optimizers, lr_schedulers)):
            if net == None or optimizer == None or lr_scheduler == None:
                continue

            """UNFREEZE"""
            if config.epoch > config.MODEL_FREEZE_EPOCH:
                updated_children = []
                for i, c in enumerate(net.children()):
                    if len(config.MODEL_NO_GRAD) > i:
                        l = config.MODEL_NO_GRAD[i]
                        for child_counter, child in enumerate(list(c.children())):
                            if child_counter in l or l == [-1]:
                                for paras in child.parameters():
                                    if not paras.requires_grad:
                                        updated_children.append(child_counter)
                                        paras.requires_grad = True
                if len(updated_children) !=0:
                    print("Enable Gradient for child_counter: {}".format(updated_children))
                    tensorboardwriter.write_text(self.writer, "Unfreeze {} layers at epoch: {}".format(updated_children, config.epoch), config.global_steps[fold])
                # if config.MODEL_LEARNING_RATE_AFTER_UNFREEZE != 0:
                #     print("Reset Learning Rate to {}".format(config.MODEL_LEARNING_RATE_AFTER_UNFREEZE))
                #     for g in optimizer.param_groups:
                #         g['lr'] = config.MODEL_LEARNING_RATE_AFTER_UNFREEZE

            # import pdb; pdb.set_trace() #1357Mb -> 1215Mb
            """Switch Optimizers"""
            net = net.cuda()
            optimizer = load.move_optimizer_to_cuda(optimizer)
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
            val_loss, val_f1 = evaluation.eval_fold(net, self.validation_loader[config.fold])
            print("""
            ValidLoss: {}, ValidF1: {}
            """.format(val_loss, val_f1))
            net = net.cpu()
            optimizer = load.move_optimizer_to_cpu(optimizer) #3299Mb
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache() #1215Mb
        """LOSS"""
        f1 = f1_macro(evaluation.epoch_pred, evaluation.epoch_label).mean()
        f1_2 = metrics.f1_score((evaluation.epoch_label > config.EVAL_THRESHOLD).astype(np.byte), (evaluation.epoch_pred > config.EVAL_THRESHOLD).astype(np.byte), average='macro')  # sklearn does not automatically import matrics.
        f1_dict = dict(("Class-{}".format(i), x) for i, x in enumerate(metrics.f1_score((evaluation.epoch_label > config.EVAL_THRESHOLD).astype(np.byte), (evaluation.epoch_pred > config.EVAL_THRESHOLD).astype(np.byte), average=None)))
        f1_dict.update({"EvalF1": f1, "Sklearn": f1_2})

        # IT WILL MESS UP THE RANDOM SEED (CAREFUL)
        shakeup = dict()
        for i in range(100):
            public_lb = set(np.random.choice(range(int(len(evaluation.epoch_pred)*0.5)), int(len(evaluation.epoch_pred)*0.5), replace=False))
            private_lb = set(range(len(evaluation.epoch_pred)))-public_lb
            public_lb = list(public_lb)
            public_lb = metrics.roc_auc_score(evaluation.epoch_label[public_lb], evaluation.epoch_pred[public_lb])
            private_lb = list(private_lb)
            private_lb = metrics.roc_auc_score(evaluation.epoch_label[private_lb], evaluation.epoch_pred[private_lb])
            score_diff = private_lb-public_lb
            shakeup[score_diff] = (public_lb, private_lb)
        shakeup_keys = sorted(shakeup)
        shakeup_mean, shakeup_std = np.mean(shakeup_keys), np.std(shakeup_keys)
        tensorboardwriter.write_shakeup(self.writer, shakeup, shakeup_keys, config.epoch)

        soft_auc_macro = metrics.roc_auc_score(evaluation.epoch_label, evaluation.epoch_pred)
        hard_auc_macro = metrics.roc_auc_score((evaluation.epoch_label > config.EVAL_THRESHOLD).astype(np.byte), (evaluation.epoch_pred>config.EVAL_THRESHOLD).astype(np.byte))
        soft_auc_micro = metrics.roc_auc_score(evaluation.epoch_label, evaluation.epoch_pred, average='micro')
        hard_auc_micro = metrics.roc_auc_score((evaluation.epoch_label > config.EVAL_THRESHOLD).astype(np.byte), (evaluation.epoch_pred>config.EVAL_THRESHOLD).astype(np.byte), average='micro')

        report = classification_report(np.argmax(evaluation.epoch_label, axis=1), np.argmax(evaluation.epoch_pred, axis=1), target_names=["Negative", "Positive"])
        max_names = max(f1_dict.items(), key=operator.itemgetter(1))
        min_names = min(f1_dict.items(), key=operator.itemgetter(1))
        report = report + """
        Shakeup Mean: {}
        Shakeup STD: {}""".format(shakeup_mean, shakeup_std) + """
        Soft AUC Macro: {}
        Hard AUC Macro: {}
        Soft AUC Micro: {}
        Hard AUC Micro: {}
        """.format(soft_auc_macro, hard_auc_macro, soft_auc_micro, hard_auc_micro) + """
        F1 by sklearn = {}
        Max = {}, socre = {}
        Min = {}, score = {}
        """.format(f1_2, max_names[0], max_names[1], min_names[0], min_names[1])
        print(report)
        for lr_scheduler in lr_schedulers:
            if lr_scheduler == None:
                continue
            lr_scheduler.step(metrics=soft_auc_macro, epoch=config.epoch)
        tensorboardwriter.write_text(self.writer, report, config.epoch)

        tensorboardwriter.write_epoch_loss(self.writer, f1_dict, config.epoch)
        tensorboardwriter.write_pred_distribution(self.writer, evaluation.epoch_pred.flatten(), config.epoch)

        """THRESHOLD"""
        if config.EVAL_IF_THRESHOLD_TEST:
            best_threshold = 0.0
            best_val = 0.0

            best_threshold_dict = np.zeros(config.TRAIN_NUM_CLASS)
            best_val_dict = np.zeros(config.TRAIN_NUM_CLASS)

            pbar = tqdm(config.EVAL_TRY_THRESHOLD)
            for threshold in pbar:
                score = f1_macro(evaluation.epoch_pred, evaluation.epoch_label, thresh=threshold).mean()
                tensorboardwriter.write_threshold(self.writer, -1, score, threshold * 1000.0, config.fold)
                if score > best_val:
                    best_threshold = threshold
                    best_val = score
                pbar.set_description("Threshold: {}; F1: {}".format(threshold, score))

                for c in range(config.TRAIN_NUM_CLASS):
                    score = metrics.f1_score(evaluation.epoch_label[:][c], (evaluation.epoch_pred[:][c] > threshold))
                    tensorboardwriter.write_threshold(self.writer, c, score, threshold * 1000.0, config.fold)
                    if score > best_val_dict[c]:
                        best_threshold_dict[c] = threshold
                        best_val_dict[c] = score

            tensorboardwriter.write_best_threshold(self.writer, -1, best_val, best_threshold, config.epoch, config.fold)
            for c in range(config.TRAIN_NUM_CLASS): tensorboardwriter.write_best_threshold(self.writer, c, best_val_dict[c], best_threshold_dict[c], config.epoch, config.fold)

        """HISTOGRAM"""
        if config.DISPLAY_HISTOGRAM:
            tensorboardwriter.write_eval_loss(self.writer, {"EvalFocalMean": evaluation.mean(), "EvalFocalSTD": evaluation.std()}, config.epoch)
            tensorboardwriter.write_loss_distribution(self.writer, np.array(list(itertools.chain.from_iterable(evaluation.epoch_losses))).flatten(), config.epoch)

        """CLEAN UP"""
        del evaluation

class HisCancerEvaluation:
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

        # self.best_id = None
        # self.worst_id = None
        # self.best_loss = None
        # self.worst_loss = None

        self.epoch_pred = None
        self.epoch_label = None

    def eval_fold(self, net, validation_loader):
        focal_losses = np.array([])
        predict_total = None
        label_total = None

        # self.best_id = []
        # self.worst_id = []
        # self.best_loss = []
        # self.worst_loss = []

        print("Set Model Trainning mode to trainning=[{}]".format(net.eval().training))
        for eval_index in tqdm(range(config.EVAL_RATIO)):
            config.eval_index = eval_index
            pbar = tqdm(validation_loader)
            total_confidence = 0

            for batch_index, (ids, image, labels_0, image_for_display) in enumerate(pbar):

                """CALCULATE LOSS"""
                if config.TRAIN_GPU_ARG:
                    image = image.cuda()
                    labels_0 = labels_0.cuda()

                logits_predict = net(image)
                if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
                prob_predict = torch.nn.Softmax()(logits_predict)

                """LOSS"""
                focal = focalloss_softmax(alpha=0.25, gamma=5, eps=1e-7)(labels_0, logits_predict)
                f1, precise, recall = differenciable_f_softmax(beta=1)(labels_0, logits_predict)

                """EVALUATE LOSS"""
                focal = focal.detach()
                # focal_min = focal.min().item()
                # focal_min_id = (focal == focal_min).nonzero()
                # focal_min_id = focal_min_id.view(focal_min_id.size(), -1)[0]
                # focal_min_id = ids[focal_min_id.cpu().numpy()[0]]
                # focal_max = focal.max().item()
                # focal_max_id = (focal == focal_max).nonzero()
                # focal_max_id = focal_max_id.view(focal_max_id.size(), -1)[0]
                # focal_max_id = ids[focal_max_id.cpu().numpy()[0]]
                # self.best_loss = np.append(self.best_loss, focal_min)
                # self.worst_loss = np.append(self.worst_loss, focal_max)
                # self.best_id = np.append(self.best_id, focal_min_id)
                # self.worst_id = np.append(self.worst_id, focal_max_id)
                # del focal_min, focal_min_id, focal_max, focal_max_id
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
                prob_predict = prob_predict.detach().cpu().numpy()

                """SUM"""
                # np.append(self.f1_losses, f1_macro(prob_predict, labels_0).mean())
                self.f1_losses = np.append(self.f1_losses, f1.mean())
                focal_losses = np.append(focal_losses, focal_mean)

                confidence = np.absolute(prob_predict - 0.5).mean() + 0.5
                total_confidence = total_confidence + confidence

                """PRINT"""
                # label = np.array(self.dataset.multilabel_binarizer.inverse_transform(labels_0)[0])
                # pred = np.array(self.dataset.multilabel_binarizer.inverse_transform(prob_predict>0.5)[0])
                # pbar.set_description_str("(E{}-F{}) Stp:{} Label:{} Pred:{} Left:{}".format(int(config.global_steps[fold]), label, pred, left))
                pbar.set_description("(E{}F{}I{}) Focal:{} F1:{} Conf:{}".format(config.epoch, config.fold, config.eval_index, focal_mean, f1.mean(), total_confidence/(batch_index+1)))
                # if config.DISPLAY_HISTOGRAM: self.epoch_losses.append(focal.flatten())
                predict_total = np.concatenate((predict_total, prob_predict), axis=0) if predict_total is not None else prob_predict
                label_total = np.concatenate((label_total, labels_0), axis=0) if label_total is not None else labels_0

                """CLEAN UP"""
                del ids, image, image_for_display
                del focal, f1, precise, recall, labels_0, logits_predict, prob_predict
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