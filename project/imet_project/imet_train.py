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
from dataset.imet_dataset import IMetDataset
from dataset.imet_dataset import train_collate, val_collate
from gpu import gpu_profile
from loss.f import differenciable_f_sigmoid, fbeta_score
from loss.focal import focalloss_sigmoid
from lr_scheduler.PlateauCyclicRestart import PlateauCyclicRestart
from project.imet_project import imet_net
from utils import load
from utils.load import save_checkpoint_fold, load_checkpoint_all_fold, save_onnx, remove_checkpoint_fold, set_milestone
from utils.lr_finder import LRFinder

if os.environ.get('DISPLAY', '') == '':
    print('WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.')
    mpl.use('Agg')
from matplotlib import pyplot as plt


class IMetTrain:
    def __init__(self, writer):
        self.writer = writer

        self.optimizers = []
        self.nets = []
        self.lr_schedulers = []
        self.train_loader = []
        self.validation_loader = []

        # TODO
        self.dataset = IMetDataset(config.DIRECTORY_CSV, config.DIRECTORY_SAMPLE_CSV, load_strategy="train", writer=self.writer, id_col=config.DIRECTORY_CSV_ID, target_col=config.DIRECTORY_CSV_TARGET)
        self.folded_samplers = self.dataset.get_stratified_samplers(fold=config.MODEL_FOLD)

        for fold in range(config.MODEL_FOLD):
            if fold not in config.train_fold:
                print("     Skipping dataset = IMetDataset(config.DIRECTORY_CSV, fold: #{})".format(fold))
                self.optimizers.append(None)
                self.nets.append(None)
                self.lr_schedulers.append(None)
            else:
                print("     Creating Fold: #{}".format(fold))
                net = imet_net.se_resnext50_32x4d(config.TRAIN_NUM_CLASS, pretrained="imagenet")

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
                self.lr_schedulers.append(PlateauCyclicRestart(optimizer, eval_mode='max', factor=config.MODEL_LR_SCHEDULER_REDUCE_FACTOR, patience=config.MODEL_LR_SCHEDULER_PATIENT, verbose=False, threshold=1e-4, threshold_mode='abs', cooldown=0, eps=1e-8, base_lr=config.MODEL_LR_SCHEDULER_BASELR, max_lr=config.MODEL_LR_SCHEDULER_MAXLR, step_size=config.MODEL_LR_SCHEDULER_STEP, mode='plateau_cyclic', gamma=1., scale_mode='cycle', last_batch_iteration=-1, reduce_restart=config.MODEL_LR_SCHEDULER_REDUCE_RESTART))

            self.train_loader.append(data.DataLoader(self.dataset,
                                                     batch_size=config.MODEL_BATCH_SIZE,
                                                     shuffle=False,
                                                     sampler=self.folded_samplers[fold]["train"],
                                                     batch_sampler=None,
                                                     num_workers=config.TRAIN_NUM_WORKER,
                                                     collate_fn=train_collate,
                                                     pin_memory=True,
                                                     drop_last=True,  # Last batch will mess up with batch norm https://github.com/pytorch/pytorch/issues/4534
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
                if optim is None:
                    continue
                for g in optim.param_groups:
                    g['lr'] = config.resetlr
                    g['initial_lr'] = config.resetlr

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
                                                 ), val_loader=val_loader, end_lr=0.5, num_iter=config.FIND_LR_RATIO, step_mode="exp")
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
            while config.epoch < config.MODEL_EPOCHS:
                # """CAM"""
                # if np.array(config.MODEL_NO_GRAD).flatten() == []:
                #     pbar = tqdm(data.DataLoader(self.dataset,
                #                                  batch_size=1,
                #                                  shuffle=False,
                #                                  sampler=self.folded_samplers[0]["val"],
                #                                  batch_sampler=None,
                #                                  num_workers=config.TRAIN_NUM_WORKER,
                #                                  collate_fn=val_collate,
                #                                  pin_memory=False,
                #                                  drop_last=False,
                #                                  timeout=0,
                #                                  worker_init_fn=None,
                #                                  ))
                #     if config.TRAIN_GPU_ARG: self.nets[0].cuda()
                #
                #     print("Set Model Trainning mode to trainning=[{}]".format(self.nets[0].eval().training))
                #     for batch_index, (ids, image, labels_0, image_for_display) in enumerate(pbar):
                #         if config.TRAIN_GPU_ARG:
                #             image = image.cuda()
                #             labels_0 = labels_0.cuda()
                #
                #         cam_img = cam(self.nets[0], image, labels_0)
                #         logits_predict = self.nets[0](image)
                #         prob_predict = torch.nn.Softmax()(logits_predict).detach().cpu().numpy()
                #         pbar.set_description_str("Cam...")
                #
                #         tensorboardwriter.write_focus(self.writer, ids[0].split("/")[-1], cam_img, image_for_display[0].numpy().transpose((1, 2, 0)), np.argmax(labels_0.cpu().numpy(), axis=1), np.argmax(prob_predict, axis=1), batch_index, config.fold)
                #         del image, labels_0
                #         if batch_index > 50: break
                #     self.nets[0].cpu()
                #     if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()

                self.step_epoch(nets=self.nets,
                                optimizers=self.optimizers,
                                lr_schedulers=self.lr_schedulers,
                                batch_size=config.MODEL_BATCH_SIZE
                                )
                if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
                """SAVE AND DELETE"""
                save_checkpoint_fold([x.state_dict() if x is not None else None for x in self.nets], [x.state_dict() if x is not None else None for x in self.optimizers if x is not None], [x.state_dict() if x is not None else None for x in self.lr_schedulers if x is not None])
                remove_checkpoint_fold()

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

        evaluation = IMetEvaluation(self.writer, self.dataset.multilabel_binarizer)
        for fold, (net, optimizer, lr_scheduler) in enumerate(zip(nets, optimizers, lr_schedulers)):
            if net is None or optimizer is None or lr_scheduler is None:
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
                if len(updated_children) != 0:
                    print("Enable Gradient for child_counter: {}".format(updated_children))
                    tensorboardwriter.write_text(self.writer, "Unfreeze {} layers at epoch: {}".format(updated_children, config.epoch), config.global_steps[fold])
                # if config.MODEL_LEARNING_RATE_AFTER_UNFREEZE != 0:
                #     print("Reset Learning Rate to {}".format(config.MODEL_LEARNING_RATE_AFTER_UNFREEZE))
                #     for g in optimizer.param_groups:
                #         g['lr'] = config.MODEL_LEARNING_RATE_AFTER_UNFREEZE

            # import pdb; pdb.set_trace() #1357Mb -> 1215Mb
            """Switch Optimizers"""
            # if config.epoch == 50:
            #     optimizer = torch.optim.SGD(net.parameters(), lr=config.MODEL_INIT_LEARNING_RATE, momentum=config.MODEL_MOMENTUM, dampening=0, weight_decay=config.MODEL_WEIGHT_DEFAY, nesterov=False)
            #     tensorboardwriter.write_text(self.writer, "Switch to torch.optim.SGD, weight_decay={}, momentum={}".format(config.MODEL_WEIGHT_DEFAY, config.MODEL_MOMENTUM), config.global_steps[fold])
            net = net.cuda()
            optimizer = load.move_optimizer_to_cuda(optimizer)
            if config.TRAIN: self.step_fold(fold, net, optimizer, lr_scheduler, batch_size)
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
            val_loss, val_f = evaluation.eval_fold(net, self.validation_loader[config.fold])
            print("""
            ValidLoss: {}, ValidF: {}
            """.format(val_loss, val_f))
            net = net.cpu()
            optimizer = load.move_optimizer_to_cpu(optimizer)
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()

        # """DISPLAY"""
        # best_id, best_loss = evaluation.best()
        # worst_id, worst_loss = evaluation.worst()
        # for fold, (best_id, best_loss, worst_id, worst_loss) in enumerate(zip(best_id, best_loss, worst_id, worst_loss)):
        #     best_img = self.dataset.get_load_image_by_id(best_id)
        #     best_label = self.dataset.multilabel_binarizer.inverse_transform(np.expand_dims(self.dataset.get_load_label_by_id(best_id), axis=0))[0]
        #     worst_img = self.dataset.get_load_image_by_id(worst_id)
        #     worst_label = self.dataset.multilabel_binarizer.inverse_transform(np.expand_dims(self.dataset.get_load_label_by_id(worst_id), axis=0))[0]
        #     # print("best_img.shape = {}".format(best_img.shape))
        #     tensorboardwriter.write_best_img(self.writer, img=best_img, label=best_label, id=best_id, loss=best_loss, fold=fold)
        #     tensorboardwriter.write_worst_img(self.writer, img=worst_img, label=worst_label, id=worst_id, loss=worst_loss, fold=fold)

        """LOSS"""
        # f = f1_macro(evaluation.epoch_pred, evaluation.epoch_label).mean()
        f = fbeta_score(evaluation.epoch_label, evaluation.epoch_pred, beta=2, threshold=0.5)
        f_sklearn = metrics.fbeta_score((evaluation.epoch_label > config.EVAL_THRESHOLD).astype(np.byte), (evaluation.epoch_pred > config.EVAL_THRESHOLD).astype(np.byte), beta=2, average='macro')  # sklearn does not automatically import matrics.
        f_dict = dict(("Class-{}".format(i), x) for i, x in enumerate(metrics.fbeta_score((evaluation.epoch_label > config.EVAL_THRESHOLD).astype(np.byte), (evaluation.epoch_pred > config.EVAL_THRESHOLD).astype(np.byte), beta=2, average=None)))
        tensorboardwriter.write_classwise_loss_distribution(self.writer, np.array(f_dict.values()), config.epoch)

        # IT WILL MESS UP THE RANDOM SEED (CAREFUL)
        shakeup = dict()
        pbar = tqdm(range(config.EVAL_SHAKEUP_RATIO))
        for i in pbar:
            public_lb = set(np.random.choice(range(len(evaluation.epoch_pred)), int(len(evaluation.epoch_pred) * 0.5), replace=False))
            private_lb = set(range(len(evaluation.epoch_pred))) - public_lb
            public_lb = np.array(list(public_lb)).astype(dtype=np.int)
            # public_lb = metrics.roc_auc_score(evaluation.epoch_label[public_lb], evaluation.epoch_pred[public_lb])
            public_lb = fbeta_score(evaluation.epoch_label[public_lb], evaluation.epoch_pred[public_lb], beta=2, threshold=0.5)
            private_lb = np.array(list(private_lb)).astype(dtype=np.int)
            # private_lb = metrics.roc_auc_score(evaluation.epoch_label[private_lb], evaluation.epoch_pred[private_lb])
            private_lb = fbeta_score(evaluation.epoch_label[private_lb], evaluation.epoch_pred[private_lb], beta=2, threshold=0.5)
            score_diff = private_lb - public_lb
            shakeup[score_diff] = (public_lb, private_lb)
            pbar.set_description_str("Public LB: {}, Private LB: {}, Difference: {}".format(public_lb, private_lb, score_diff))
        shakeup_keys = sorted(shakeup)
        shakeup_mean, shakeup_std = np.mean(shakeup_keys), np.std(shakeup_keys)
        tensorboardwriter.write_shakeup(self.writer, shakeup, shakeup_keys, config.epoch)

        # soft_auc_macro = metrics.roc_auc_score(evaluation.epoch_label, evaluation.epoch_pred)
        # hard_auc_macro = metrics.roc_auc_score((evaluation.epoch_label > config.EVAL_THRESHOLD).astype(np.byte), (evaluation.epoch_pred>config.EVAL_THRESHOLD).astype(np.byte))
        # soft_auc_micro = metrics.roc_auc_score(evaluation.epoch_label, evaluation.epoch_pred, average='micro')
        # hard_auc_micro = metrics.roc_auc_score((evaluation.epoch_label > config.EVAL_THRESHOLD).astype(np.byte), (evaluation.epoch_pred>config.EVAL_THRESHOLD).astype(np.byte), average='micro')

        # report = classification_report(np.argmax(evaluation.epoch_label, axis=1), np.argmax(evaluation.epoch_pred, axis=1), target_names=["Negative", "Positive"])
        report = ""
        max_names = max(f_dict.items(), key=operator.itemgetter(1))
        min_names = min(f_dict.items(), key=operator.itemgetter(1))
        report = report + """
        Shakeup Mean of Sample Mean: {}
        Shakeup STD of Sample Mean: {}
        Predicted Shakeup STD: {}
        n={} --> n={} (<1.750261596x)""".format(shakeup_mean, shakeup_std, shakeup_std * np.sqrt(len(evaluation.epoch_label)) / np.sqrt(57459 / 2), len(evaluation.epoch_label), int(57459 / 2)) \
                 + """
        F2 sklearn = {}
        Max = {}, socre = {}
        Min = {}, score = {}
        """.format(f_sklearn, max_names[0], max_names[1], min_names[0], min_names[1])
        #          + """
        # Soft AUC Macro: {}
        # Hard AUC Macro: {}
        # Soft AUC Micro: {}
        # Hard AUC Micro: {}
        # """.format(soft_auc_macro, hard_auc_macro, soft_auc_micro, hard_auc_micro)
        print(report)
        for lr_scheduler in lr_schedulers:
            if lr_scheduler is None:
                continue
            lr_scheduler.step(metrics=f_sklearn, epoch=config.epoch)
        tensorboardwriter.write_text(self.writer, report, config.epoch)

        tensorboardwriter.write_epoch_loss(self.writer, {"EvalF": f, "F2Sklearn": f_sklearn}, config.epoch)
        if config.EVAL_IF_PRED_DISTRIBUTION: tensorboardwriter.write_pred_distribution(self.writer, evaluation.epoch_pred.flatten(), config.epoch)

        """THRESHOLD"""
        if config.EVAL_IF_THRESHOLD_TEST:
            best_threshold = 0.0
            best_val = 0.0

            best_threshold_dict = np.zeros(config.TRAIN_NUM_CLASS)
            best_val_dict = np.zeros(config.TRAIN_NUM_CLASS)

            pbar = tqdm(config.EVAL_TRY_THRESHOLD)
            for threshold in pbar:
                score = fbeta_score(evaluation.epoch_label, evaluation.epoch_pred, beta=2, threshold=threshold)
                tensorboardwriter.write_threshold(self.writer, -1, score, threshold * 1000.0, config.fold)
                if score > best_val:
                    best_threshold = threshold
                    best_val = score
                pbar.set_description("Threshold: {}; F: {}".format(threshold, score))

                for c in range(config.TRAIN_NUM_CLASS):
                    score = metrics.fbeta_score(evaluation.epoch_label[:][c], (evaluation.epoch_pred[:][c] > threshold), beta=2)
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

    def step_fold(self, fold, net, optimizer, lr_scheduler, batch_size):
        config.fold = fold

        epoch_loss = 0
        epoch_f = 0
        train_len = 1e-10
        total_confidence = 0

        # pin_memory: https://blog.csdn.net/tsq292978891/article/details/80454568
        train_loader = self.train_loader[config.fold]

        print("Set Model Trainning mode to trainning=[{}]".format(net.train().training))

        ratio = int(config.TRAIN_RATIO) if config.TRAIN_RATIO >= 1 else 1
        for train_index in tqdm(range(ratio)):
            pbar = tqdm(train_loader)
            train_len = train_len + len(train_loader)
            for batch_index, (ids, image, labels_0, image_for_display) in enumerate(pbar):
                if train_len < 1 and config.epoch % (1 / config.TRAIN_RATIO) != batch_index % (1 / config.TRAIN_RATIO):
                    continue

                # 1215MB -> 4997MB = 3782

                # """UPDATE LR"""
                # if config.global_steps[fold] == 2 * 46808 / 32 - 1: print("Perfect Place to Stop")
                # optimizer.state['lr'] = config.TRAIN_TRY_LR_FORMULA(config.global_steps[fold]) if config.TRAIN_TRY_LR else config.TRAIN_COSINE(config.global_steps[fold])

                lr_scheduler.step(0, config.epoch, config.global_steps[fold], float(train_len))
                """TRAIN NET"""
                config.global_steps[fold] = config.global_steps[fold] + 1
                if config.TRAIN_GPU_ARG:
                    image = image.cuda()
                    labels_0 = labels_0.cuda()
                logits_predict = net(image)
                prob_predict = torch.nn.Sigmoid()(logits_predict)

                """LOSS"""
                focal = focalloss_sigmoid(alpha=0.25, gamma=2, eps=1e-7)(labels_0, logits_predict)
                f, precise, recall = differenciable_f_sigmoid(beta=2)(labels_0, logits_predict)
                bce = BCELoss()(prob_predict, labels_0)
                positive_bce = BCELoss(weight=labels_0 * 20 + 1)(prob_predict, labels_0)
                loss = focal.mean()
                """BACKPROP"""
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                """DETATCH"""
                focal = focal.detach().cpu().numpy().mean()
                f = f.detach().cpu().numpy().mean()
                precise = precise.detach().cpu().numpy().mean()
                recall = recall.detach().cpu().numpy().mean()
                bce = bce.detach().cpu().numpy().mean()
                positive_bce = positive_bce.detach().cpu().numpy().mean()
                # weighted_bce = weighted_bce.detach().cpu().numpy().mean()
                loss = loss.detach().cpu().numpy().mean()
                labels_0 = labels_0.cpu().numpy()
                logits_predict = logits_predict.detach().cpu().numpy()
                prob_predict = prob_predict.detach().cpu().numpy()
                # print(image)

                """SUM"""
                epoch_loss = epoch_loss + loss.mean()
                epoch_f = epoch_f + f.mean()
                # f = f1_macro(logits_predict, labels_0).mean()
                confidence = np.absolute(prob_predict - 0.5).mean() + 0.5
                total_confidence = total_confidence + confidence

                """DISPLAY"""
                tensorboardwriter.write_memory(self.writer, "train")

                # soft_auc_macro = metrics.roc_auc_score(y_true=labels_0, y_score=prob_predict, average='macro')
                # soft_auc_micro = metrics.roc_auc_score(y_true=labels_0, y_score=prob_predict, average='micro')
                # left = self.dataset.multilabel_binarizer.inverse_transform((np.expand_dims((np.array(labels_0).sum(0) < 1).astype(np.byte), axis=0)))[0]
                # label = np.array(self.dataset.multilabel_binarizer.inverse_transform(labels_0)[0])
                label = 0
                # pred = np.array(self.dataset.multilabel_binarizer.inverse_transform(prob_predict > config.EVAL_THRESHOLD)[0])
                pred = 0
                pbar.set_description_str("(E{}-F{}) Stp:{} Label:{} Pred:{} Conf:{:.4f} lr:{}".format(config.epoch, config.fold, int(config.global_steps[fold]), label, pred, total_confidence / (batch_index + 1), optimizer.param_groups[0]['lr']))
                # pbar.set_description_str("(E{}-F{}) Stp:{} Label:{} Pred:{} Left:{}".format(config.epoch, config.fold, int(config.global_steps[fold]), label, pred, left))
                # pbar.set_description_str("(E{}-F{}) Stp:{} Focal:{:.4f} F:{:.4f} lr:{:.4E} BCE:{:.2f}|{:.2f}".format(config.epoch, config.fold, int(config.global_steps[fold]), focal, f, optimizer.param_groups[0]['lr'], weighted_bce, bce))
                # pbar.set_description_str("(E{}-F{}) Stp:{} Y:{}, y:{}".format(config.epoch, config.fold, int(config.global_steps[fold]), labels_0, logits_predict))

                out_dict = {'Epoch/{}'.format(config.fold): config.epoch,
                            'LearningRate{}/{}'.format(optimizer.__class__.__name__, config.fold): optimizer.param_groups[0]['lr'],
                            'Loss/{}'.format(config.fold): loss,
                            'F/{}'.format(config.fold): f,
                            'Focal/{}'.format(config.fold): focal,
                            'PositiveBCE/{}'.format(config.fold): positive_bce,
                            # 'WeightedBCE/{}'.format(config.fold): weighted_bce,
                            'BCE/{}'.format(config.fold): bce,
                            'Precision/{}'.format(config.fold): precise,
                            'Recall/{}'.format(config.fold): recall,
                            'LogitsProbability/{}'.format(config.fold): logits_predict.mean(),
                            'PredictProbability/{}'.format(config.fold): prob_predict.mean(),
                            'LabelProbability/{}'.format(config.fold): labels_0.mean(),
                            # 'AUCMacro/{}'.format(config.fold): soft_auc_macro,
                            # 'AUCMicro/{}'.format(config.fold): soft_auc_micro,
                            }
                # for c in range(config.TRAIN_NUM_CLASS):
                #     out_dict['PredictProbability-Class-{}/{}'.format(c, config.fold)] = prob_predict[:][c].mean()
                # the code above is wrong - IndexError: index 64 is out of bounds for axis 0 with size 64

                tensorboardwriter.write_loss(self.writer, out_dict, config.global_steps[fold])

                """CLEAN UP"""
                del ids, image, image_for_display
                del focal, f, precise, recall, bce, positive_bce, loss, labels_0, logits_predict, prob_predict
                if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()  # release gpu memory
            del pbar

        train_loss = epoch_loss / train_len
        epoch_f = epoch_f / train_len
        print("""
            Epoch: {}, Fold: {}
            TrainLoss: {}, FLoss: {}
        """.format(config.epoch, config.fold, train_loss, epoch_f))
        # lr_scheduler.step(epoch_f, epoch=config.epoch)

        del train_loss

        # if config.DISPLAY_HISTOGRAM:
        #     for i, (name, param) in enumerate(net.named_parameters()):
        #         print("Calculating Histogram #{}".format(i))
        #         writer.add_histogram(name, param.clone().cpu().data.numpy(), config.epoch)
        if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()  # release gpu memory


class IMetEvaluation:
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
        self.f_losses = np.array([])

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
                prob_predict = torch.nn.Sigmoid()(logits_predict)

                """LOSS"""
                focal = focalloss_sigmoid(alpha=0.25, gamma=5, eps=1e-7)(labels_0, logits_predict)
                f, precise, recall = differenciable_f_sigmoid(beta=2)(labels_0, logits_predict)

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
                f = f.detach().cpu().numpy()
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
                # np.append(self.f_losses, f1_macro(prob_predict, labels_0).mean())
                self.f_losses = np.append(self.f_losses, f.mean())
                focal_losses = np.append(focal_losses, focal_mean)

                confidence = np.absolute(prob_predict - 0.5).mean() + 0.5
                total_confidence = total_confidence + confidence

                """PRINT"""
                # label = np.array(self.dataset.multilabel_binarizer.inverse_transform(labels_0)[0])
                # pred = np.array(self.dataset.multilabel_binarizer.inverse_transform(prob_predict>0.5)[0])
                # pbar.set_description_str("(E{}-F{}) Stp:{} Label:{} Pred:{} Left:{}".format(int(config.global_steps[fold]), label, pred, left))
                pbar.set_description("(E{}F{}I{}) Focal:{} F:{} Conf:{}".format(config.epoch, config.fold, config.eval_index, focal_mean, f.mean(), total_confidence / (batch_index + 1)))
                # if config.DISPLAY_HISTOGRAM: self.epoch_losses.append(focal.flatten())
                predict_total = np.concatenate((predict_total, prob_predict), axis=0) if predict_total is not None else prob_predict
                label_total = np.concatenate((label_total, labels_0), axis=0) if label_total is not None else labels_0

                """DISPLAY"""
                tensorboardwriter.write_memory(self.writer, "train")
                if config.DISPLAY_VISUALIZATION and batch_index < max(1, config.MODEL_BATCH_SIZE / 32): self.display(config.fold, ids, image, image_for_display, labels_0, prob_predict, focal)

                """CLEAN UP"""
                del ids, image, image_for_display
                del focal, f, precise, recall, labels_0, logits_predict, prob_predict
                if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
                if config.DEBUG_TRAISE_GPU: gpu_profile(frame=sys._getframe(), event='line', arg=None)
            del pbar
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
        """LOSS"""
        f = fbeta_score(label_total, predict_total, beta=2, threshold=0.5)
        tensorboardwriter.write_eval_loss(self.writer, {"FoldFocal/{}".format(config.fold): focal_losses.mean(), "FoldF/{}".format(config.fold): f}, config.epoch)
        tensorboardwriter.write_pr_curve(self.writer, label_total, predict_total, config.epoch, config.fold)
        self.epoch_pred = np.concatenate((self.epoch_pred, predict_total), axis=0) if self.epoch_pred is not None else predict_total
        self.epoch_label = np.concatenate((self.epoch_label, label_total), axis=0) if self.epoch_label is not None else label_total

        if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
        mean_loss = focal_losses.mean()
        self.mean_losses.append(mean_loss)
        if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
        return mean_loss, f

    def __int__(self):
        return self.mean()

    def mean(self, axis=None):
        # if axis == None: return np.array(list(itertools.chain.from_iterable(self.epoch_losses))).mean()
        # print("WARNING: self.epoch_losse may have different shape according to different shape of loss caused by different batch. Numpy cannot take the mean of it is baches shapes are different.")
        # return np.array(self.epoch_losses).mean(axis)
        return np.array(self.mean_losses).mean()

    def std(self, axis=None):
        if axis is None: return np.array(list(itertools.chain.from_iterable(self.epoch_losses))).std()
        print("WARNING: self.epoch_losse may have different shape according to different shape of loss caused by different batch. Numpy cannot take the mean of it is baches shapes are different.")
        return np.array(self.epoch_losses).std(axis)

    def f_mean(self):
        return self.f_losses.mean()

    # def best(self):
    #     return (self.best_id, self.best_loss)

    # def worst(self):
    #     return (self.worst_id, self.worst_loss)

    def display(self, fold, ids, transfereds, untransfereds, labels, predicteds, losses):
        # tensorboardwriter.write_pr_curve(self.writer, labels, predicteds, config.global_steps[fold], fold)

        for index, (img_id, transfered, untransfered, label, predicted, loss) in enumerate(zip(ids, transfereds, untransfereds, labels, predicteds, losses)):
            if index != 0: continue

            label = self.binarlizer.inverse_transform(np.expand_dims(np.array(label).astype(np.byte), axis=0))[0]
            predict = self.binarlizer.inverse_transform(np.expand_dims((predicted > config.EVAL_THRESHOLD).astype(np.byte), axis=0))[0]

            F = plt.figure()

            plt.subplot(321)
            plt.imshow(untransfered.transpose((1, 2, 0)), vmin=0, vmax=1)
            plt.title("Real; pred:{}".format(predict))
            plt.grid(False)

            plt.subplot(322)
            plt.imshow(transfered.transpose((1, 2, 0)), vmin=0, vmax=1)
            plt.title("Trans")
            plt.grid(False)

            plt.subplot(323)
            plt.imshow(untransfered.transpose((1, 2, 0)), vmin=0, vmax=1)
            plt.title("Real; label:{}".format(label))
            plt.grid(False)

            plt.subplot(324)
            plt.imshow(transfered.transpose((1, 2, 0)), vmin=0, vmax=1)
            plt.title("Trans; f:{}".format(loss))
            plt.grid(False)
            tensorboardwriter.write_image(self.writer, "{}-{}".format(fold, img_id), F, config.epoch)
