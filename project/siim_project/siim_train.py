import os
import sys
import time
from timeit import default_timer

import matplotlib as mpl
import numpy as np
import torch
from torch.nn import BCELoss
from torch.utils import data
from torchsummary import summary
from tqdm import tqdm

import config
import tensorboardwriter
from dataset.siim_dataset import SIIMDataset
from dataset.siim_dataset import train_collate, val_collate
from gpu import gpu_profile
from loss.cross_entropy import segmentation_weighted_binary_cross_entropy
from loss.dice import binary_dice_pytorch_loss, binary_dice_numpy_gain
# from loss.hinge import lovasz_hinge
from loss.iou import mIoULoss
from lr_scheduler.Constant import Constant
from optimizer import adamw
from project.siim_project import siim_net
from project.siim_project.siim_net import model34_DeepSupervion, model50A_DeepSupervion, model34_DeepSupervion_GroupNorm
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
                self.optimizers.append(None)
                self.nets.append(None)
                self.lr_schedulers.append(None)
            else:
                print("     Creating Fold: #{}".format(fold))

                if config.net == "resunet50":
                    net = siim_net.resunet(encoder_depth=50, num_classes=config.TRAIN_NUM_CLASS, num_filters=32, dropout_2d=0.2, pretrained=False, is_deconv=False)
                elif config.net == "resunet50-ds":
                    net = model50A_DeepSupervion(num_classes=config.TRAIN_NUM_CLASS)
                elif config.net == "resunet34-ds":
                    net = model34_DeepSupervion(num_classes=config.TRAIN_NUM_CLASS)
                elif config.net == "resunet34-ds-gn":
                    net = model34_DeepSupervion_GroupNorm(num_classes=config.TRAIN_NUM_CLASS)
                ## leaky relu?
                else:
                    raise ValueError("The Network {} you specified is not in one of the network you can use".format(config.net))

                if config.freeze:
                    for name, param in net.named_parameters():
                        if config.freeze in name:
                            param.requires_grad = False
                            print("Set {} require_grad = False because it contains '{}'".format(name, config.freeze))

                """FREEZING LAYER"""
                if config.manual_freeze:
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

                print("Let's use", torch.cuda.device_count(), "GPUs!")
                net = torch.nn.DataParallel(net, device_ids=[i for i in range(torch.cuda.device_count())])

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
        load_checkpoint_all_fold(self.nets, self.optimizers, self.lr_schedulers, config.DIRECTORY_LOAD)
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
        if config.debug_lr_finder:
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
            lr_finder = LRFinder(self.nets[config.train_fold[0]], self.optimizers[config.train_fold[0]], binary_dice_pytorch_loss, writer=self.writer, device="cuda")
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
                if config.display_architecture:
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
                #     self.nets[0].cuda()
                #
                #     print("Set Model Trainning mode to trainning=[{}]".format(self.nets[0].eval().training))
                #     for batch_index, (ids, image, labels_0, image_for_display) in enumerate(pbar):
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
                #     torch.cuda.empty_cache()

                """Step Epoch"""
                torch.cuda.empty_cache()
                self.step_epoch(nets=self.nets,
                                optimizers=self.optimizers,
                                lr_schedulers=self.lr_schedulers,
                                batch_size=config.MODEL_BATCH_SIZE
                                )
                torch.cuda.empty_cache()

                """SAVE AND DELETE"""
                save_checkpoint_fold([x.state_dict() if x is not None else None for x in self.nets], [x.state_dict() if x is not None else None for x in self.optimizers if x is not None], [x.state_dict() if x is not None else None for x in self.lr_schedulers if x is not None])
                remove_checkpoint_fold()
            torch.cuda.empty_cache()

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
                    if config.display_architecture: print("Enable Gradient for child_counter: {}".format(updated_children))
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

            if config.display_architecture:
                summary(net, (1, 1024, 1024))
                config.display_architecture = False

            optimizer = load.move_optimizer_to_cuda(optimizer)
            if config.train: self.step_fold(fold, net, optimizer, lr_scheduler, batch_size)
            torch.cuda.empty_cache()
            with torch.no_grad():
                score = eval_fold(net, self.writer, self.validation_loader[config.fold])

            """Update Learning Rate Scheduler"""
            if lr_scheduler is not None:
                _ = lr_scheduler.step_epoch(metrics=score, epoch=config.epoch)
                print(_)

            net = net.cpu()
            optimizer = load.move_optimizer_to_cpu(optimizer)
            torch.cuda.empty_cache()

    def step_fold(self, fold, net, optimizer, lr_scheduler, batch_size):
        config.fold = fold

        epoch_loss = 0
        epoch_f = 0
        train_len = 1e-10
        total_confidence = 0

        train_loss = np.zeros(20, np.float32)
        batch_loss = np.zeros(20, np.float32)
        sum_loss = np.zeros(20, np.float32)
        sum_number = np.zeros(20, np.float32) + 1e-8

        # pin_memory: https://blog.csdn.net/tsq292978891/article/details/80454568
        train_loader = self.train_loader[config.fold]

        print("Set Model Trainning mode to trainning=[{}]".format(net.train().training))

        ratio = int(config.TRAIN_RATIO) if config.TRAIN_RATIO >= 1 else 1
        out_dict = None
        for train_index in tqdm(range(ratio)):
            pbar = tqdm(train_loader)
            train_len = train_len + len(train_loader)
            for batch_index, (ids, image, labels, image_0, labels_0, empty) in enumerate(pbar):
                # drop last batch that has irregular shape
                if train_len < 1 and config.epoch % (1 / config.TRAIN_RATIO) != batch_index % (1 / config.TRAIN_RATIO):
                    continue

                """UPDATE LR"""
                # if config.global_steps[fold] == 2 * 46808 / 32 - 1: print("Perfect Place to Stop")
                # optimizer.state['lr'] = config.TRAIN_TRY_LR_FORMULA(config.global_steps[fold]) if config.TRAIN_TRY_LR else config.TRAIN_COSINE(config.global_steps[fold])
                lr_scheduler.step(0, config.epoch, config.global_steps[fold], float(train_len))

                """TRAIN NET"""
                config.global_steps[fold] = config.global_steps[fold] + 1
                image = image.cuda()
                empty_logits, _idkwhatthisis_, logits_predict = net(image)
                prob_predict = torch.nn.Sigmoid()(logits_predict)
                prob_empty = torch.nn.Sigmoid()(empty_logits)
                # prob_predict = prob_empty.unsqueeze(-1).unsqueeze(-1) * prob_predict -> THIS IS REALLY BAD BEHAVIOR

                """LOSS"""
                labels = labels.cuda().float()
                empty = empty.cuda().float()  # I don't know why I need to specify float() -> otherwise it will be long

                # dice = denoised_siim_dice(threshold=config.EVAL_THRESHOLD, iou=False, denoised=False)(labels, prob_predict)
                dice = binary_dice_pytorch_loss(labels, prob_predict, smooth=1e-5)
                # iou = denoised_siim_dice(threshold=config.EVAL_THRESHOLD, iou=True, denoised=False)(labels, prob_predict)
                iou = mIoULoss(mean=False, eps=1e-5)(labels, prob_predict)
                # hinge = lovasz_hinge(labels.squeeze(1), logits_predict.squeeze(1))
                bce = BCELoss(reduction='none')(prob_empty.squeeze(-1), empty)
                # ce = BCELoss(reduction='none')(prob_predict.squeeze(1).view(prob_predict.shape[0], -1), labels.squeeze(1).view(labels.shape[0], -1))
                ce = segmentation_weighted_binary_cross_entropy(prob_predict.squeeze(1), labels.squeeze(1), pos_prob=0.25, neg_prob=0.75)

                """Heng CherKeng"""
                dice_cherkeng, dice_neg, dice_pos, num_neg, num_pos = metric(labels, logits_predict)

                if config.epoch < 21:
                    loss = 0.9 * ce.mean() + 0.1 * bce.mean()
                elif config.epoch < 61:
                    loss = 0.4 * dice.mean() + 0.4 * ce.mean() + 0.1 * bce.mean()
                elif config.epoch < 62:
                    loss = 0.4 * dice.mean() + 0.4 * ce.mean() + 0.1 * bce.mean()
                elif config.epoch < 81:
                    loss = 0.4 * dice.mean() + 0.4 * ce.mean() + 0.1 * bce.mean()
                elif config.epoch < 121:
                    loss = 0.3 * dice.mean() + 0.4 * ce.mean() + 0.3 * bce.mean()
                elif config.epoch < 161:
                    loss = 0.2 * dice.mean() + 0.5 * ce.mean() + 0.3 * bce.mean()
                elif config.epoch < 195:
                    loss = 0.1 * dice.mean() + 0.6 * ce.mean() + 0.3 * bce.mean()
                elif config.epoch < 221:
                    loss = 0.2 * dice.mean() + 0.5 * 50 * ce.mean() + 0.3 * 50 * bce.mean()
                else:
                    raise ValueError("Please Specify the Loss at Epoch = {}".format(config.epoch))

                """BACKPROP"""
                loss.backward()
                if (batch_index + 1) % config.TRAIN_GRADIENT_ACCUMULATION == 0:  # backward
                    optimizer.step()
                    optimizer.zero_grad()
                elif batch_index + 1 == len(train_loader):  # drop last batch if it can't backprop
                    optimizer.zero_grad()
                else:
                    pass  # accumulation

                """Heng CherKeng"""
                batch_loss[:4] = [loss.item(), dice_cherkeng, dice_neg, dice_pos]
                sum_loss[:4] += [loss.item() * batch_size, dice_cherkeng * batch_size, dice_neg * num_neg, dice_pos * num_pos]
                sum_number[:4] += [batch_size, batch_size, num_neg, num_pos]
                if (batch_index + 1) % config.TRAIN_GRADIENT_ACCUMULATION == 0:
                    train_loss = sum_loss / sum_number
                    sum_loss[...] = 0
                    sum_number[...] = 1e-8

                """DETATCH"""
                dice = dice.detach().cpu().numpy().mean()
                iou = iou.detach().cpu().numpy().mean()
                # hinge = hinge.detach().cpu().numpy().mean()
                bce = bce.detach().cpu().numpy().mean()
                ce = ce.detach().cpu().numpy().mean()
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
                # tensorboardwriter.write_memory(self.writer, "train")
                self.writer.add_scalars('stats/Memory', {"GPU-Tensor": float(torch.cuda.memory_allocated()),
                                                         "GPU-Cache": float(torch.cuda.memory_cached()),
                                                         "GPU-Tensor-Max": float(torch.cuda.max_memory_allocated()),
                                                         "GPU-Cache-Max": float(torch.cuda.max_memory_cached()),
                                                         }, global_step=int(time.time() - config.start_time))
                """Heng CherKeng"""
                pbar.set_description_str('%0.5f  %5.1f%s %5.1f |  %5.3f   %5.3f  %4.2f  %4.2f  |  %5.3f   %5.3f  %4.2f  %4.2f  | %s' % (optimizer.param_groups[0]['lr'], iter / 1000, " ", config.epoch, *train_loss[:4], *train_loss[:4], config.time_to_str((default_timer() - config.start_time), 'min'))
                    )
                # pbar.set_description_str("(E{}-F{}) Stp:{} Dice:{} BCE:{} Conf:{:.4f} lr:{}".format(config.epoch, config.fold, int(config.global_steps[fold]), dice, bce, total_confidence / (batch_index + 1), optimizer.param_groups[0]['lr']))

                if out_dict is None:
                    out_dict = {'LearningRate{}/{}'.format(optimizer.__class__.__name__, config.fold): optimizer.param_groups[0]['lr'] / config.TRAIN_GRADIENT_ACCUMULATION,
                                'Loss/{}'.format(config.fold): loss / config.TRAIN_GRADIENT_ACCUMULATION,
                                'Dice/{}'.format(config.fold): dice / config.TRAIN_GRADIENT_ACCUMULATION,
                                'IOU/{}'.format(config.fold): iou / config.TRAIN_GRADIENT_ACCUMULATION,
                                # 'Hinge/{}'.format(config.fold): hinge/config.TRAIN_GRADIENT_ACCUMULATION,
                                'BCE/{}'.format(config.fold): bce / config.TRAIN_GRADIENT_ACCUMULATION,
                                'CE/{}'.format(config.fold): ce / config.TRAIN_GRADIENT_ACCUMULATION,
                                'LogitsProbability/{}'.format(config.fold): logits_predict.mean() / config.TRAIN_GRADIENT_ACCUMULATION,
                                'PredictProbability/{}'.format(config.fold): prob_predict.mean() / config.TRAIN_GRADIENT_ACCUMULATION,
                                'EmptyProbability/{}'.format(config.fold): prob_empty.mean() / config.TRAIN_GRADIENT_ACCUMULATION,
                                'LabelProbability/{}'.format(config.fold): labels.mean() / config.TRAIN_GRADIENT_ACCUMULATION,
                                'EmptyGroundProbability'.format(config.fold): empty.mean() / config.TRAIN_GRADIENT_ACCUMULATION,
                                }
                else:
                    out_dict['LearningRate{}/{}'.format(optimizer.__class__.__name__, config.fold)] += optimizer.param_groups[0]['lr'] / config.TRAIN_GRADIENT_ACCUMULATION
                    out_dict['Loss/{}'.format(config.fold)] += loss / config.TRAIN_GRADIENT_ACCUMULATION
                    out_dict['Dice/{}'.format(config.fold)] += dice / config.TRAIN_GRADIENT_ACCUMULATION
                    out_dict['IOU/{}'.format(config.fold)] += iou / config.TRAIN_GRADIENT_ACCUMULATION
                    out_dict['BCE/{}'.format(config.fold)] += bce / config.TRAIN_GRADIENT_ACCUMULATION
                    out_dict['CE/{}'.format(config.fold)] += ce / config.TRAIN_GRADIENT_ACCUMULATION
                    out_dict['LogitsProbability/{}'.format(config.fold)] += logits_predict.mean() / config.TRAIN_GRADIENT_ACCUMULATION
                    out_dict['PredictProbability/{}'.format(config.fold)] += prob_predict.mean() / config.TRAIN_GRADIENT_ACCUMULATION
                    out_dict['EmptyProbability/{}'.format(config.fold)] += prob_empty.mean() / config.TRAIN_GRADIENT_ACCUMULATION
                    out_dict['LabelProbability/{}'.format(config.fold)] += labels.mean() / config.TRAIN_GRADIENT_ACCUMULATION
                    out_dict['EmptyGroundProbability'.format(config.fold)] += empty.mean() / config.TRAIN_GRADIENT_ACCUMULATION
                if (batch_index + 1) % config.TRAIN_GRADIENT_ACCUMULATION == 0:  # backward
                    tensorboardwriter.write_loss(self.writer, out_dict, config.global_steps[fold])
                    out_dict = None
                # for c in range(config.TRAIN_NUM_CLASS):
                #     out_dict['PredictProbability-Class-{}/{}'.format(c, config.fold)] = prob_predict[:][c].mean()
                # the code above is wrong - IndexError: index 64 is out of bounds for axis 0 with size 64

                """CLEAN UP"""
                del ids, image_0, labels_0  # things threw away
                del dice, iou, bce, ce, loss, image, labels, empty, logits_predict, prob_predict, prob_empty  # detach
                torch.cuda.empty_cache()  # release gpu memory
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
        # lr_scheduler.step_epoch(epoch_f, epoch=config.epoch)

        del train_loss

        # if config.DISPLAY_HISTOGRAM:
        #     for i, (name, param) in enumerate(net.named_parameters()):
        #         print("Calculating Histogram #{}".format(i))
        #         writer.add_histogram(name, param.clone().cpu().data.numpy(), config.epoch)
        torch.cuda.empty_cache()  # release gpu memory


def eval_fold(net, writer, validation_loader):
    dice_losses = np.array([])
    iou_losses = np.array([])
    bce_losses = np.array([])
    ce_losses = np.array([])
    # hinge_losses = np.array([])
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
        displayed_img = 0
        displayed_empty = 0
        display_max = 0
        display_min = 0

        for batch_index, (ids, image, labels, image_0, labels_0, empty) in enumerate(pbar):
            """TRAIN NET"""
            image = image.cuda()
            empty_logits, _idkwhatthisis_, logits_predict = net(image)
            prob_predict = torch.nn.Sigmoid()(logits_predict)
            prob_empty = torch.nn.Sigmoid()(empty_logits)
            # prob_predict = prob_empty.unsqueeze(-1).unsqueeze(-1) * prob_predict -> THIS IS REALLY BAD BEHAVIOR

            """LOSS"""
            labels = labels.cuda().float()
            empty = empty.cuda().float()  # I don't know why I need to specify float() -> otherwise it will be long

            # dice = denoised_siim_dice(threshold=config.EVAL_THRESHOLD, iou=False, denoised=False)(labels, prob_predict)
            dice = binary_dice_pytorch_loss(labels, prob_predict, smooth=1e-5)
            # iou = denoised_siim_dice(threshold=config.EVAL_THRESHOLD, iou=True, denoised=False)(labels, prob_predict)
            iou = mIoULoss(mean=False, eps=1e-5)(labels, prob_predict)
            # hinge = lovasz_hinge(labels.squeeze(1), logits_predict.squeeze(1))
            bce = BCELoss(reduction='none')(prob_empty.squeeze(-1), empty)
            # ce = BCELoss(reduction='none')(prob_predict.squeeze(1).view(prob_predict.shape[0], -1), labels.squeeze(1).view(labels.shape[0], -1))
            ce = segmentation_weighted_binary_cross_entropy(prob_predict.squeeze(1), labels.squeeze(1), pos_prob=0.25, neg_prob=0.75)

            if config.epoch < 21:
                loss = 0.9 * ce.mean() + 0.1 * bce.mean()
            elif config.epoch < 61:
                loss = 0.4 * dice.mean() + 0.4 * ce.mean() + 0.1 * bce.mean()
            elif config.epoch < 62:
                loss = 0.4 * dice.mean() + 0.4 * ce.mean() + 0.1 * bce.mean()
            elif config.epoch < 80:
                loss = 0.4 * dice.mean() + 0.4 * ce.mean() + 0.1 * bce.mean()
            elif config.epoch < 121:
                loss = 0.3 * dice.mean() + 0.4 * ce.mean() + 0.3 * bce.mean()
            elif config.epoch < 161:
                loss = 0.2 * dice.mean() + 0.5 * ce.mean() + 0.3 * bce.mean()
            elif config.epoch < 195:
                loss = 0.1 * dice.mean() + 0.6 * ce.mean() + 0.3 * bce.mean()
            elif config.epoch < 221:
                loss = 0.2 * dice.mean() + 0.5 * 50 * ce.mean() + 0.3 * 50 * bce.mean()
            else:
                raise ValueError("Please Specify the Loss at Epoch = {}".format(config.epoch))

            """DETATCH WITHOUT MEAN"""
            dice = dice.detach().cpu().numpy()
            iou = iou.detach().cpu().numpy()
            # hinge = hinge.detach().cpu().numpy().mean()
            bce = bce.detach().cpu().numpy()
            ce = ce.detach().cpu().numpy()
            loss = loss.detach().cpu().numpy()

            image = image.cpu().numpy()
            labels = labels.cpu().numpy()
            empty = empty.cpu().numpy()
            logits_predict = logits_predict.detach().cpu().numpy()
            prob_predict = prob_predict.detach().cpu().numpy()
            prob_empty = prob_empty.detach().cpu().numpy()

            """Draw Image"""
            if displayed_img < 8 or displayed_empty < 8:
                for image_, label_, prob_predict_, empty_, prob_empty_, dice_, bce_, ce_ in zip(image, labels, prob_predict, empty, prob_empty, dice, bce, ce):
                    if empty_.sum() is not 0 and displayed_img < 8:
                        F = draw_image(image_, label_, prob_predict_, empty_, prob_empty_, dice_, bce_, ce_)
                        tensorboardwriter.write_image(writer, "{}-{}".format(config.fold, displayed_img), F, config.epoch, category="non-empty")
                        displayed_img = displayed_img + 1
                    if empty_.sum() is 0 and displayed_empty < 8:
                        F = draw_image(image_, label_, prob_predict_, empty_, prob_empty_, dice_, bce_, ce_)
                        tensorboardwriter.write_image(writer, "{}-{}".format(config.fold, displayed_empty), F, config.epoch, category="empty")
                        displayed_empty = displayed_empty + 1
            if display_max < 0:
                arg = np.argmax(dice)
                F = draw_image(image[arg], labels[arg], prob_predict[arg], empty[arg], prob_empty[arg], dice[arg], bce[arg], ce[arg])
                tensorboardwriter.write_image(writer, "{}-{}".format(config.fold, display_max), F, config.epoch, category="batch_max")
                display_max = display_max + 1
            if display_min < 8:
                arg = np.argmin(dice)
                F = draw_image(image[arg], labels[arg], prob_predict[arg], empty[arg], prob_empty[arg], dice[arg], bce[arg], ce[arg])
                tensorboardwriter.write_image(writer, "{}-{}".format(config.fold, display_min), F, config.epoch, category="batch_min")
                display_min = display_min + 1

            """SUM"""
            dice = dice.mean()
            iou = iou.mean()
            bce = bce.mean()
            ce = ce.mean()
            loss = loss.mean()

            dice_losses = np.append(dice_losses, dice)
            iou_losses = np.append(iou_losses, iou)
            bce_losses = np.append(bce_losses, bce)
            ce_losses = np.append(ce_losses, ce)
            # hinge_losses = np.append(hinge_losses, hinge)
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
            # tensorboardwriter.write_memory(writer, "train")
            writer.add_scalars('stats/GPU-Memory', {"GPU-Tensor": float(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())}, global_step=int(time.time() - config.start_time))
            writer.add_scalars('stats/GPU-Memory', {"GPU-Cache": float(torch.cuda.memory_cached() / torch.cuda.max_memory_cached())}, global_step=int(time.time() - config.start_time))

            """CLEAN UP"""
            del ids, image_0, labels_0  # things threw away
            del dice, iou, bce, ce, loss, image, labels, empty, logits_predict, prob_predict, prob_empty  # detach
            torch.cuda.empty_cache()
            if config.DEBUG_TRAISE_GPU: gpu_profile(frame=sys._getframe(), event='line', arg=None)
        del pbar
        torch.cuda.empty_cache()

    """LOSS"""
    tensorboardwriter.write_eval_loss(writer, {"Dice/{}".format(config.fold): dice_losses.mean(),
                                               "IOU/{}".format(config.fold): iou_losses.mean(),
                                               "BCE/{}".format(config.fold): bce_losses.mean(),
                                               "CE/{}".format(config.fold): ce_losses.mean(),
                                               # "Hinge/{}".format(config.fold): hinge_losses.mean(),
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

    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    """LOSS"""
    score = binary_dice_numpy_gain(label, pred_hard, mean=True)

    """Shakeup"""
    # IT WILL MESS UP THE RANDOM SEED (CAREFUL)
    shakeup, shakeup_keys, shakeup_mean, shakeup_std = calculate_shakeup(label, pred_hard, binary_dice_numpy_gain, config.EVAL_SHAKEUP_RATIO, mean=True)
    tensorboardwriter.write_shakeup(writer, shakeup, shakeup_keys, shakeup_std, config.epoch)

    """Print"""
    report = """
    Shakeup Mean of Sample Mean: {}
    Shakeup STD of Sample Mean: {}""".format(shakeup_mean, shakeup_std) + """
    Score = {} """.format(score)
    tensorboardwriter.write_epoch_loss(writer, {"Score": score}, config.epoch)

    """THRESHOLD"""
    if config.EVAL_IF_THRESHOLD_TEST:
        best_threshold, best_val, total_score, total_tried = calculate_threshold(label, pred_soft, binary_dice_numpy_gain, config.EVAL_TRY_THRESHOLD, writer, config.fold, n_class=1, mean=True)
        report = report + """Best Threshold is: {}, Score: {}, AreaUnder: {}""".format(best_threshold, best_val, total_score / total_tried)
        tensorboardwriter.write_best_threshold(writer, -1, best_val, best_threshold, total_score / total_tried, config.epoch, config.fold)

    print(report)
    tensorboardwriter.write_text(writer, report, config.global_steps[config.fold])

    return score


def draw_image(image, ground, pred, empty, prob_empty, dice, bce, ce):
    F = plt.figure()

    plt.subplot(321)
    plt.imshow(np.squeeze(image), cmap='plasma', vmin=0, vmax=1)
    plt.title("P:{:.4f}".format(prob_empty[0]))
    plt.grid(False)

    plt.subplot(322)
    plt.imshow(np.squeeze(ground), cmap='plasma', vmin=0, vmax=1)
    plt.title("D:{:.4f} Empty:{}".format(dice, empty != 0.))
    plt.grid(False)

    plt.subplot(323)
    plt.imshow(np.squeeze(pred), cmap='plasma', vmin=pred.min(), vmax=pred.max())
    plt.title("B:{:.4f} C:{:.4f}".format(bce, ce.mean()))
    plt.grid(False)

    plt.subplot(324)
    thresholded = (pred > config.EVAL_THRESHOLD).astype(np.byte)
    plt.imshow(np.squeeze(thresholded), cmap='plasma', vmin=0, vmax=1)
    plt.title("Empty:{}".format(thresholded.sum() == 0))
    plt.grid(False)

    return F


def metric(truth, logit, threshold=0.5, reduction='none'):
    batch_size = len(truth)

    with torch.no_grad():
        logit = logit.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert (logit.shape == truth.shape)

        probability = torch.sigmoid(logit)
        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)
        # print(len(neg_index), len(pos_index))

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos
