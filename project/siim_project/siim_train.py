import itertools
import operator
import os
import sys

import matplotlib as mpl
import numpy as np
import torch
from sklearn import metrics
from torch.nn import BCELoss, CrossEntropyLoss
from torch.utils import data
from tqdm import tqdm

import config
import tensorboardwriter
from dataset.siim_dataset import SIIMDataset
from dataset.siim_dataset import train_collate, val_collate
from gpu import gpu_profile
from loss.dice import denoised_siim_dice, siim_dice_overall, cmp_instance_dice
from loss.f import differenciable_f_sigmoid, fbeta_score_numpy
from loss.focal import focalloss_sigmoid_refined
from loss.hinge import lovasz_hinge
from lr_scheduler.Constant import Constant
from lr_scheduler.PlateauCyclicRestart import PlateauCyclicRestart
from optimizer import adamw
from project.siim_project import siim_net
from project.siim_project.siim_net import model50A_DeepSupervion
from utils import load
from utils.load import save_checkpoint_fold, load_checkpoint_all_fold, save_onnx, remove_checkpoint_fold, set_milestone
from utils.lr_finder import LRFinder
from utils.other import calculate_shakeup, calculate_threshold

if os.environ.get('DISPLAY', '') == '':
    print('WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.')
    mpl.use('Agg')
from matplotlib import pyplot as plt


class SIIMTrain:
    def __init__(self, writer):
        self.writer = writer

        self.optimizers = []
        self.nets = []
        self.lr_schedulers = []
        self.train_loader = []
        self.validation_loader = []

        # TODO
        self.dataset = SIIMDataset(config.DIRECTORY_CSV, config.DIRECTORY_SAMPLE_CSV, load_strategy="train", writer=self.writer, id_col=config.DIRECTORY_CSV_ID, target_col=config.DIRECTORY_CSV_TARGET)
        self.folded_samplers = self.dataset.get_stratified_samplers(fold=config.MODEL_FOLD)

        for fold in range(config.MODEL_FOLD):
            if fold not in config.train_fold:
                print("     Skipping dataset = SIIMDataset(config.DIRECTORY_CSV, fold: #{})".format(fold))
                self.optimizers.append(None)
                self.nets.append(None)
                self.lr_schedulers.append(None)
            else:
                print("     Creating Fold: #{}".format(fold))
                # net = siim_net.resunet(encoder_depth=50, num_classes=config.TRAIN_NUM_CLASS, num_filters=32, dropout_2d=0.2, pretrained=False, is_deconv=False)
                net = model50A_DeepSupervion(num_classes=config.TRAIN_NUM_CLASS)

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

                # optimizer = torch.optim.SGD(params=net.parameters(), lr=config.MODEL_INIT_LEARNING_RATE, momentum=config.MODEL_MOMENTUM, weight_decay=config.MODEL_WEIGHT_DECAY)
                # optimizer = torch.optim.Adam(params=net.parameters(), lr=config.MODEL_INIT_LEARNING_RATE, weight_decay=config.MODEL_WEIGHT_DECAY)
                optimizer = adamw.AdamW(params=net.parameters(), lr=config.MODEL_INIT_LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=config.MODEL_WEIGHT_DECAY, amsgrad=False)
                self.optimizers.append(optimizer)
                self.nets.append(net)
                # self.lr_schedulers.append(PlateauCyclicRestart(optimizer,
                #                                                eval_mode='max',
                #                                                factor=config.MODEL_LR_SCHEDULER_REDUCE_FACTOR,
                #                                                patience=config.MODEL_LR_SCHEDULER_PATIENT,
                #                                                verbose=False,
                #                                                threshold=config.MODEL_LR_SCHEDULER_THRESHOLD,
                #                                                threshold_mode='abs',
                #                                                cooldown=0,
                #                                                eps=1e-8,
                #                                                base_lr=config.MODEL_LR_SCHEDULER_BASELR,
                #                                                max_lr=config.MODEL_LR_SCHEDULER_MAXLR,
                #                                                step_size=config.MODEL_LR_SCHEDULER_STEP,
                #                                                mode='plateau_cyclic',
                #                                                gamma=1.,
                #                                                scale_mode='cycle',
                #                                                last_batch_iteration=-1,
                #                                                reduce_restart=config.MODEL_LR_SCHEDULER_REDUCE_RESTART,
                #                                                restart_coef=config.MODEL_LR_SCHEDULER_RESTART_COEF))
                self.lr_schedulers.append(Constant(optimizer, eval_mode="max", threshold=config.MODEL_LR_SCHEDULER_THRESHOLD, threshold_mode="abs", last_batch_iteration=-1))

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
                                                          pin_memory=True,
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

        """LR FINDER"""
        if config.DEBUG_LR_FINDER:
            val_loader = data.DataLoader(self.dataset,
                                         batch_size=config.MODEL_BATCH_SIZE,
                                         shuffle=False,
                                         sampler=self.folded_samplers[config.train_fold[0]]["val"],
                                         batch_sampler=None,
                                         num_workers=config.TRAIN_NUM_WORKER,
                                         collate_fn=val_collate,
                                         pin_memory=True,
                                         drop_last=False,
                                         timeout=0,
                                         worker_init_fn=None,
                                         ) if config.FIND_LR_ON_VALIDATION else None
            lr_finder = LRFinder(self.nets[config.train_fold[0]], self.optimizers[config.train_fold[0]], denoised_siim_dice(threshold=config.EVAL_THRESHOLD, iou=False, denoised=False), writer=self.writer, device="cuda")
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

                """Step Epoch"""
                if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
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

            """KeyboardInterrupt"""
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

            """Switch Optimizers"""
            # if config.epoch == 50:
            #     optimizer = torch.optim.SGD(net.parameters(), lr=config.MODEL_INIT_LEARNING_RATE, momentum=config.MODEL_MOMENTUM, dampening=0, weight_decay=config.MODEL_WEIGHT_DEFAY, nesterov=False)
            #     tensorboardwriter.write_text(self.writer, "Switch to torch.optim.SGD, weight_decay={}, momentum={}".format(config.MODEL_WEIGHT_DEFAY, config.MODEL_MOMENTUM), config.global_steps[fold])

            """Train and Eval"""
            net = net.cuda()
            optimizer = load.move_optimizer_to_cuda(optimizer)
            if config.TRAIN: self.step_fold(fold, net, optimizer, lr_scheduler, batch_size)
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
            score = eval_fold(net, self.writer, self.validation_loader[config.fold])

            """Update Learning Rate Scheduler"""
            if lr_scheduler is not None:
                _ = lr_scheduler.step(metrics=score, epoch=config.epoch)
                print(_)

            net = net.cpu()
            optimizer = load.move_optimizer_to_cpu(optimizer)
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()

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
            for batch_index, (ids, image, labels, image_0, labels_0, empty) in enumerate(pbar):
                labels = image # for testing

                # drop last batch that has irregular shape
                if train_len < 1 and config.epoch % (1 / config.TRAIN_RATIO) != batch_index % (1 / config.TRAIN_RATIO):
                    continue

                """UPDATE LR"""
                # if config.global_steps[fold] == 2 * 46808 / 32 - 1: print("Perfect Place to Stop")
                # optimizer.state['lr'] = config.TRAIN_TRY_LR_FORMULA(config.global_steps[fold]) if config.TRAIN_TRY_LR else config.TRAIN_COSINE(config.global_steps[fold])
                lr_scheduler.step(0, config.epoch, config.global_steps[fold], float(train_len))

                """TRAIN NET"""
                config.global_steps[fold] = config.global_steps[fold] + 1
                if config.TRAIN_GPU_ARG: image = image.cuda()
                empty_logits, _idkwhatthisis_, logits_predict = net(image)
                prob_predict = torch.nn.Sigmoid()(logits_predict)
                prob_empty = torch.nn.Sigmoid()(empty_logits)

                """LOSS"""
                if config.TRAIN_GPU_ARG:
                    labels = labels.cuda()
                    empty = empty.cuda().float()  # I don't know why I need to specify float() -> otherwise it will be long
                dice = denoised_siim_dice(threshold=config.EVAL_THRESHOLD, iou=False, denoised=False)(labels, logits_predict)
                iou = denoised_siim_dice(threshold=config.EVAL_THRESHOLD, iou=True, denoised=False)(labels, logits_predict)
                hinge = lovasz_hinge(labels.squeeze(1), logits_predict.squeeze(1))
                bce = BCELoss()(prob_empty, empty)
                ce = CrossEntropyLoss()(logits_predict.squeeze(1), labels.squeeze(1).long())
                # loss = 0.5 * dice.mean() + 0.5 * bce.mean()
                loss = ce.mean()

                """BACKPROP"""
                loss.backward()
                if config.epoch > config.TRAIN_GRADIENT_ACCUMULATION:
                    if (batch_index + 1) % config.TRAIN_GRADIENT_ACCUMULATION == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    elif batch_index + 1 == len(train_loader):  # drop last
                        optimizer.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()

                """DETATCH"""
                dice = dice.detach().cpu().numpy().mean()
                iou = iou.detach().cpu().numpy().mean()
                hinge = hinge.detach().cpu().numpy().mean()
                bce = bce.detach().cpu().numpy().mean()
                ce = bce.detach().cpu().numpy().mean()
                loss = loss.detach().cpu().numpy().mean()

                image = image.cpu().numpy()
                labels = labels.cpu().numpy()
                empty = empty.cpu().numpy()
                logits_predict = logits_predict.detach().cpu().numpy()
                prob_predict = prob_predict.detach().cpu().numpy()
                prob_empty = prob_empty.detach().cpu().numpy()

                """SUM"""
                epoch_loss = epoch_loss + loss.mean()
                # f = f1_macro(logits_predict, labels).mean()
                confidence = np.absolute(prob_predict - 0.5).mean() + 0.5
                total_confidence = total_confidence + confidence

                """DISPLAY"""
                tensorboardwriter.write_memory(self.writer, "train")

                pbar.set_description_str("(E{}-F{}) Stp:{} Dice:{} BCE:{} Conf:{:.4f} lr:{}".format(config.epoch, config.fold, int(config.global_steps[fold]), dice, bce, total_confidence / (batch_index + 1), optimizer.param_groups[0]['lr']))
                out_dict = {'Epoch/{}'.format(config.fold): config.epoch,
                            'LearningRate{}/{}'.format(optimizer.__class__.__name__, config.fold): optimizer.param_groups[0]['lr'],
                            'Loss/{}'.format(config.fold): loss,
                            'Dice/{}'.format(config.fold): dice,
                            'IOU/{}'.format(config.fold): iou,
                            'Hinge/{}'.format(config.fold): hinge,
                            'BCE/{}'.format(config.fold): bce,
                            'LogitsProbability/{}'.format(config.fold): logits_predict.mean(),
                            'PredictProbability/{}'.format(config.fold): prob_predict.mean(),
                            'EmptyProbability/{}'.format(config.fold): prob_empty.mean(),
                            'LabelProbability/{}'.format(config.fold): labels.mean(),
                            'EmptyGroundProbability'.format(config.fold): empty.mean(),
                            }
                # for c in range(config.TRAIN_NUM_CLASS):
                #     out_dict['PredictProbability-Class-{}/{}'.format(c, config.fold)] = prob_predict[:][c].mean()
                # the code above is wrong - IndexError: index 64 is out of bounds for axis 0 with size 64

                tensorboardwriter.write_loss(self.writer, out_dict, config.global_steps[fold])

                """CLEAN UP"""
                del ids, image_0, labels_0  # things threw away
                del dice, iou, hinge, bce, ce, loss, image, labels, empty, logits_predict, prob_predict, prob_empty  # detach
                if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()  # release gpu memory
            del pbar

        train_loss = epoch_loss / train_len
        print("""
        Epoch: {}, Fold: {}
        TrainLoss: {}
        """.format(config.epoch, config.fold, train_loss))
        tensorboardwriter.write_text(self.writer, """
        Epoch: {}, Fold: {}
        TrainLoss: {}
        """.format(config.epoch, config.fold, train_loss), config.global_steps[config.fold] - 1)
        # lr_scheduler.step(epoch_f, epoch=config.epoch)

        del train_loss

        # if config.DISPLAY_HISTOGRAM:
        #     for i, (name, param) in enumerate(net.named_parameters()):
        #         print("Calculating Histogram #{}".format(i))
        #         writer.add_histogram(name, param.clone().cpu().data.numpy(), config.epoch)
        if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()  # release gpu memory


def eval_fold(net, writer, validation_loader):
    dice_losses = np.array([])
    iou_losses = np.array([])
    hinge_losses = np.array([])
    loss_losses = np.array([])

    predict_total = None
    label_total = None
    prob_empty_total = None
    empty_total = None

    # self.best_id = []
    # self.worst_id = []
    # self.best_loss = []
    # self.worst_loss = []

    print("Set Model Trainning mode to trainning=[{}]".format(net.eval().training))
    pbar = tqdm(range(config.EVAL_RATIO)) if config.epoch >= config.MODEL_FREEZE_EPOCH + 2 else tqdm(range(1))
    for eval_index in pbar:
        config.eval_index = eval_index
        pbar = tqdm(validation_loader)
        total_confidence = 0

        for batch_index, (ids, image, labels, image_0, labels_0, empty) in enumerate(pbar):
            """TRAIN NET"""
            if config.TRAIN_GPU_ARG: image = image.cuda()
            empty_logits, _idkwhatthisis_, logits_predict = net(image)
            prob_predict = torch.nn.Sigmoid()(logits_predict)
            prob_empty = torch.nn.Sigmoid()(empty_logits)

            """LOSS"""
            if config.TRAIN_GPU_ARG:
                labels = labels.cuda()
                empty = empty.cuda().float()  # I don't know why I need to specify float() -> otherwise it will be long
            dice = denoised_siim_dice(threshold=config.EVAL_THRESHOLD, iou=False, denoised=False)(labels, logits_predict)
            iou = denoised_siim_dice(threshold=config.EVAL_THRESHOLD, iou=True, denoised=False)(labels, logits_predict)
            hinge = lovasz_hinge(labels.squeeze(1), logits_predict.squeeze(1))
            bce = BCELoss(reduction='none')(prob_empty, empty)
            loss = 0.5 * dice.mean() + 0.5 * bce.mean()

            """DETATCH WITHOUT MEAN"""
            dice = dice.detach().cpu().numpy()
            iou = iou.detach().cpu().numpy()
            hinge = hinge.detach().cpu().numpy()
            bce = bce.detach().cpu().numpy()
            loss = loss.detach().cpu().numpy()

            image = image.cpu().numpy()
            labels = labels.cpu().numpy()
            empty = empty.cpu().numpy()
            logits_predict = logits_predict.detach().cpu().numpy()
            prob_predict = prob_predict.detach().cpu().numpy()
            prob_empty = prob_empty.detach().cpu().numpy()

            """SUM"""
            dice_losses = np.append(dice_losses, dice)
            iou_losses = np.append(iou_losses, iou)
            hinge_losses = np.append(hinge_losses, hinge)
            loss_losses = np.append(loss_losses, loss)

            predict_total = np.concatenate((predict_total, prob_predict), axis=0) if predict_total is not None else prob_predict
            label_total = np.concatenate((label_total, labels), axis=0) if label_total is not None else labels
            prob_empty_total = np.concatenate((prob_empty_total, prob_empty), axis=0) if prob_empty_total is not None else prob_empty
            empty_total = np.concatenate((empty_total, empty), axis=0) if empty_total is not None else empty

            confidence = np.absolute(prob_predict - 0.5).mean() + 0.5
            total_confidence = total_confidence + confidence

            """PRINT"""
            # label = np.array(self.dataset.multilabel_binarizer.inverse_transform(labels_0)[0])
            # pred = np.array(self.dataset.multilabel_binarizer.inverse_transform(prob_predict>0.5)[0])
            # pbar.set_description_str("(E{}-F{}) Stp:{} Label:{} Pred:{} Left:{}".format(int(config.global_steps[fold]), label, pred, left))
            pbar.set_description("(E{}F{}I{}) Dice:{} IOU:{} Conf:{}".format(config.epoch, config.fold, config.eval_index, dice.mean(), iou.mean(), total_confidence / (batch_index + 1)))
            # if config.DISPLAY_HISTOGRAM: self.epoch_losses.append(focal.flatten())

            """DISPLAY"""
            tensorboardwriter.write_memory(writer, "train")
            # TODO
            if config.DISPLAY_VISUALIZATION and batch_index < max(1, config.MODEL_BATCH_SIZE / 32):
                for i, (image_, label_, prob_predict_, empty_, prob_empty_, dice_, bce_) in enumerate(zip(image, labels, prob_predict, empty, prob_empty, dice, bce)):
                    F = draw_image(image_, label_, prob_predict_, empty_, prob_empty_, dice_, bce_)
                    tensorboardwriter.write_image(writer, "{}-{}".format(config.fold, i), F, config.epoch)

            """CLEAN UP"""
            del ids, image_0, labels_0  # things threw away
            del dice, iou, hinge, bce, loss, image, labels, empty, logits_predict, prob_predict, prob_empty  # detach
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
            if config.DEBUG_TRAISE_GPU: gpu_profile(frame=sys._getframe(), event='line', arg=None)
        del pbar
        if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()

    """LOSS"""
    tensorboardwriter.write_eval_loss(writer, {"Dice/{}".format(config.fold): dice_losses.mean(),
                                               "IOU/{}".format(config.fold): iou_losses.mean(),
                                               "Hinge/{}".format(config.fold): hinge_losses.mean(),
                                               "Loss/{}".format(config.fold): loss_losses.mean(),
                                               }, config.epoch)
    if config.EVAL_IF_PR_CURVE: tensorboardwriter.write_pr_curve(writer, empty_total, prob_empty_total, config.epoch, config.fold)

    """Result Summary"""
    # epoch_pred = None
    # epoch_pred = np.concatenate((epoch_pred, predict_total), axis=0) if epoch_pred is not None else predict_total
    # epoch_label = None
    # epoch_label = np.concatenate((epoch_label, label_total), axis=0) if epoch_label is not None else label_total

    pred_soft = predict_total
    pred_hard = (predict_total > config.EVAL_THRESHOLD).astype(np.byte)
    label = label_total

    if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
    if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()

    """LOSS"""
    score = cmp_instance_dice(label, pred_hard, mean=True)

    """Shakeup"""
    # IT WILL MESS UP THE RANDOM SEED (CAREFUL)
    shakeup, shakeup_keys, shakeup_mean, shakeup_std = calculate_shakeup(label, pred_hard, cmp_instance_dice, config.EVAL_SHAKEUP_RATIO, mean=True)
    tensorboardwriter.write_shakeup(writer, shakeup, shakeup_keys, shakeup_std, config.epoch)

    """Print"""
    report = """
    Shakeup Mean of Sample Mean: {}
    Shakeup STD of Sample Mean: {}""".format(shakeup_mean, shakeup_std) + """
    Score = {}""".format(score)
    tensorboardwriter.write_epoch_loss(writer, {"Score": score}, config.epoch)

    """THRESHOLD"""
    if config.EVAL_IF_THRESHOLD_TEST:
        best_threshold, best_val, total_score, total_tried = calculate_threshold(label, pred_soft, cmp_instance_dice, config.EVAL_TRY_THRESHOLD, writer, config.fold, n_class=1, mean=True)
        report = report + """Best Threshold is: {}, Score: {}, AreaUnder: {}""".format(best_threshold, best_val, total_score / total_tried)
        tensorboardwriter.write_best_threshold(writer, -1, best_val, best_threshold, total_score / total_tried, config.epoch, config.fold)

    print(report)
    tensorboardwriter.write_text(writer, report, config.global_steps[config.fold])

    return score


def draw_image(image, ground, pred, empty, prob_empty, dice, bce):
    F = plt.figure()

    plt.subplot(321)
    plt.imshow(np.squeeze(image))
    plt.title("E:{} P:{}".format(empty, prob_empty))
    plt.grid(False)

    plt.subplot(322)
    plt.imshow(np.squeeze(ground))
    plt.title("D:{}".format(dice))
    plt.grid(False)

    plt.subplot(323)
    plt.imshow(np.squeeze(pred))
    plt.title("B:{}".format(bce))
    plt.grid(False)

    return F
