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
from dataset.siim_dataset import SIIMDataset
from dataset.siim_dataset import train_collate, val_collate
from gpu import gpu_profile
from loss.dice import denoised_siim_dice
from loss.f import differenciable_f_sigmoid, fbeta_score_numpy
from loss.focal import focalloss_sigmoid_refined
from lr_scheduler.Constant import Constant
from lr_scheduler.PlateauCyclicRestart import PlateauCyclicRestart
from optimizer import adamw
from project.siim_project import siim_net
from project.siim_project.siim_net import model50A_DeepSupervion
from utils import load
from utils.load import save_checkpoint_fold, load_checkpoint_all_fold, save_onnx, remove_checkpoint_fold, set_milestone
from utils.lr_finder import LRFinder

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
            lr_finder = LRFinder(self.nets[config.train_fold[0]], self.optimizers[config.train_fold[0]], focalloss_sigmoid_refined(alpha=0.25, gamma=2, eps=1e-7), writer=self.writer, device="cuda")
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

        evaluation = SIIMEvaluation(self.writer, self.dataset.multilabel_binarizer)
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
        f = fbeta_score_numpy(evaluation.epoch_label, evaluation.epoch_pred, beta=2, threshold=config.EVAL_THRESHOLD)
        f_sklearn = metrics.fbeta_score((evaluation.epoch_label > config.EVAL_THRESHOLD).astype(np.byte), (evaluation.epoch_pred > config.EVAL_THRESHOLD).astype(np.byte), beta=2, average='samples')  # sklearn does not automatically import matrics.
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
            public_lb = fbeta_score_numpy(evaluation.epoch_label[public_lb], evaluation.epoch_pred[public_lb], beta=2, threshold=config.EVAL_THRESHOLD)
            private_lb = np.array(list(private_lb)).astype(dtype=np.int)
            # private_lb = metrics.roc_auc_score(evaluation.epoch_label[private_lb], evaluation.epoch_pred[private_lb])
            private_lb = fbeta_score_numpy(evaluation.epoch_label[private_lb], evaluation.epoch_pred[private_lb], beta=2, threshold=config.EVAL_THRESHOLD)
            score_diff = private_lb - public_lb
            shakeup[score_diff] = (public_lb, private_lb)
            pbar.set_description_str("""Public LB: {}, Private LB: {}, Difference: {}""".format(public_lb, private_lb, score_diff))
        shakeup_keys = sorted(shakeup)
        shakeup_mean, shakeup_std = np.mean(shakeup_keys), np.std(shakeup_keys)
        tensorboardwriter.write_shakeup(self.writer, shakeup, shakeup_keys, shakeup_std, config.epoch)

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
        Min = {}, score = {}""".format(f_sklearn, max_names[0], max_names[1], min_names[0], min_names[1])
        #          + """
        # Soft AUC Macro: {}
        # Hard AUC Macro: {}
        # Soft AUC Micro: {}
        # Hard AUC Micro: {}
        # """.format(soft_auc_macro, hard_auc_macro, soft_auc_micro, hard_auc_micro)
        for lr_scheduler in lr_schedulers:
            if lr_scheduler is None:
                continue
            report = report + lr_scheduler.step(metrics=f_sklearn, epoch=config.epoch)

        tensorboardwriter.write_epoch_loss(self.writer, {"EvalF": f, "F2Sklearn": f_sklearn}, config.epoch)
        if config.EVAL_IF_PRED_DISTRIBUTION: tensorboardwriter.write_pred_distribution(self.writer, evaluation.epoch_pred.flatten(), config.epoch)

        """THRESHOLD"""
        if config.EVAL_IF_THRESHOLD_TEST:
            best_threshold = 0.0
            best_val = 0.0
            bad_value = 0
            total_score = 0
            total_tried = 0

            best_threshold_dict = np.zeros(config.TRAIN_NUM_CLASS)
            best_val_dict = np.zeros(config.TRAIN_NUM_CLASS)

            pbar = tqdm(config.EVAL_TRY_THRESHOLD)
            for threshold in pbar:
                total_tried = total_tried+1
                score = fbeta_score_numpy(evaluation.epoch_label, evaluation.epoch_pred, beta=2, threshold=threshold)
                total_score = total_score + score
                tensorboardwriter.write_threshold(self.writer, -1, score, threshold * 1000.0, config.fold)
                if score > best_val:
                    best_threshold = threshold
                    best_val = score
                    bad_value = 0
                else: bad_value = bad_value + 1
                pbar.set_description("Threshold: {}; F: {}; AreaUnder: {}".format(threshold, score, total_score/total_tried))
                if bad_value > 100: break

                for c in range(config.TRAIN_NUM_CLASS):
                    score = metrics.fbeta_score(evaluation.epoch_label[:][c], (evaluation.epoch_pred[:][c] > threshold), beta=2)
                    # tensorboardwriter.write_threshold(self.writer, c, score, threshold * 1000.0, config.fold)
                    if score > best_val_dict[c]:
                        best_threshold_dict[c] = threshold
                        best_val_dict[c] = score

            tensorboardwriter.write_threshold_class(self.writer, best_threshold_dict, best_val_dict,
                                                    class_list = [366, 81, 11, 293, 221, 112, 146, 199, 805, 281, 104, 230, 328, 262, 396, 389, 142, 904, 987, 240, 271, 201, 108, 312, 187, 71, 94, 160, 123, 190, 3, 376, 372, 329, 250, 203, 140, 39, 1017, 7, 100, 53, 727, 36, 198, 367, 103, 787, 233, 129, 278, 6, 291, 177, 290, 166, 260, 34, 92, 176, 214, 20, 855, 132, 88, 327, 284, 115, 1060, 30, 207, 297, 873, 298, 365, 211, 268, 21, 31, 80, 431, 126, 286, 391, 82, 843, 167, 883, 130, 186, 314, 363, 137, 527, 388, 311, 118, 219, 264, 181, 197, 242, 19, 241, 235, 128, 73, 159, 340, 119, 892, 544, 917, 635, 355, 845, 370, 224, 333, 101, 346, 37, 364, 68, 215, 561, 394, 254, 460, 381, 652, 84, 168, 812, 174, 599, 395, 288, 798, 183, 305, 47, 200, 193, 476, 643, 523, 303, 452, 854, 296, 143, 919, 277, 38, 302, 752, 164, 310, 56, 10, 220, 87, 54, 343, 209, 719, 225, 243, 46, 66, 474, 897, 206, 1031, 62, 386, 295, 779, 246, 356, 16, 28, 918, 697, 1094, 148, 72, 244, 49, 4, 352, 213, 97, 330, 1048, 1052, 27, 64, 63, 238, 267, 12, 77, 8, 157, 853, 594, 2, 152, 1076, 139, 443, 319, 930, 208, 300, 361, 799, 85, 172, 392, 301, 856, 491, 575, 353, 1087, 357, 317, 644, 982, 344, 5, 228, 790, 117, 336, 261, 525, 1021, 52, 266, 1090, 481, 337, 687, 371, 970, 944, 0, 153, 223, 407, 89, 416, 435, 377, 589, 98, 136, 255, 414, 852, 362, 403, 17, 765, 928, 851, 736, 943, 379, 1028, 781, 380, 456, 1011, 350, 216, 632, 229, 417, 601, 247, 170, 1029, 857, 981, 374, 1067, 694, 222, 22, 93, 245, 880, 841, 341, 909, 785, 743, 59, 402, 398, 67, 150, 171, 325, 814, 1008, 120, 689, 802, 75, 869, 455, 810, 625, 602, 550, 801, 859, 504, 106, 469, 1079, 387, 762, 658, 751, 640, 165, 472, 471, 326, 1016, 109, 933, 151, 440, 169, 547, 320, 839, 609, 717, 837, 877, 9, 957, 499, 1042, 83, 419, 617, 270, 630, 50, 500, 1014, 74, 900, 397, 358, 997, 141, 894, 958, 509, 990, 205, 730, 1100, 265, 531, 299, 90, 539, 1102, 651, 408, 893, 122, 134, 847, 788, 342, 257, 385, 789, 249, 57, 292, 1082, 475, 659, 777, 15, 715, 1070, 442, 1075, 48, 759, 722, 44, 382, 429, 656, 251, 441, 530, 979, 614, 935, 984, 158, 792, 421, 980, 411, 195, 729, 926, 149, 526, 848, 740, 315, 775, 686, 144, 946, 237, 1000, 279, 178, 636, 945, 560, 324, 572, 272, 1074, 773, 484, 818, 124, 309, 360, 107, 470, 24, 969, 427, 451, 938, 401, 902, 1026, 95, 473, 69, 274, 770, 263, 375, 196, 86, 757, 683, 1049, 321, 562, 110, 865, 831, 998, 571, 778, 876, 782, 318, 820, 467, 446, 574, 760, 711, 1086, 951, 256, 797, 1043, 424, 1063, 533, 40, 399, 673, 712, 772, 1057, 868, 569, 1006, 42, 349, 748, 641, 977, 623, 976, 605, 878, 934, 828, 660, 514, 1078, 604, 911, 323, 1004, 173, 145, 646, 163, 806, 423, 1073, 677, 578, 937, 1005, 706, 1054, 162, 913, 549, 269, 817, 276, 590, 807, 577, 793, 723, 667, 1044, 1050, 899, 114, 657, 447, 606, 901, 412, 593, 1047, 891, 1002, 91, 862, 699, 384, 289, 113, 767, 351, 986, 610, 956, 96, 294, 557, 1095, 218, 1088, 564, 827, 611, 248, 696, 338, 439, 210, 154, 345, 769, 1027, 914, 895, 968, 964, 373, 444, 437, 881, 1041, 905, 710, 1053, 906, 1003, 648, 795, 565, 965, 879, 390, 700, 463, 1007, 1013, 618, 735, 842, 191, 155, 816, 521, 528, 61, 232, 275, 613, 410, 102, 65, 486, 32, 1091, 702, 60, 824, 588, 458, 226, 461, 815, 920, 453, 875, 368, 1077, 596, 846, 466, 867, 804, 836, 942, 1036, 478, 512, 105, 885, 874, 457, 936, 307, 1010, 513, 334, 43, 258, 1015, 556, 714, 1101, 567, 1055, 537, 449, 838, 941, 653, 661, 761, 679, 850, 175, 406, 316, 664, 608, 409, 763, 585, 1080, 898, 536, 285, 138, 253, 645, 693, 135, 860, 921, 866, 1051, 1025, 185, 505, 400, 287, 959, 939, 995, 823, 960, 234, 967, 966, 529, 1024, 540, 674, 546, 1, 1066, 502, 662, 771, 755, 685, 1018, 459, 861, 422, 479, 55, 882, 192, 649, 516, 1065, 907, 808, 468, 273, 681, 347, 750, 642, 794, 947, 1071, 454, 1083, 217, 709, 566, 654, 413, 204, 510, 619, 179, 393, 359, 622, 35, 948, 749, 999, 708, 1068, 669, 535, 18, 666, 507, 1037, 600, 629, 732, 1058, 511, 116, 741, 915, 829, 825, 496, 680, 76, 576, 1033, 508, 985, 1096, 887, 448, 888, 41, 425, 1069, 1081, 863, 78, 591, 932, 912, 791, 383, 972, 996, 627, 676, 551, 280, 522, 826, 849, 692, 989, 306, 1097, 973, 638, 1001, 731, 1040, 621, 871, 688, 426, 620, 924, 992, 988, 675, 631, 26, 870, 910, 1038, 555, 963, 811, 927, 548, 628, 495, 983, 1009, 832, 1056, 678, 929, 705, 672, 701, 889, 518, 739, 570, 890, 1012, 252, 721, 354, 864, 563, 971, 756, 695, 445, 332, 1045, 713, 482, 908, 822, 834, 450, 884, 1085, 728, 690, 819, 282, 786, 637, 1064, 840, 58, 607, 524, 534, 23, 1030, 1089, 488, 994, 133, 432, 490, 726, 553, 691, 707, 978, 783, 991, 517, 766, 647, 202, 803, 568, 515, 1032, 236, 503, 745, 587, 916, 33, 331, 506, 595, 665, 493, 497, 633, 1019, 925, 322, 725, 438, 545, 592, 348, 558, 940, 581, 573, 462, 520, 809, 465, 684, 239, 308, 954, 703, 1022, 428, 974, 582, 339, 821, 682, 227, 14, 953, 774, 952, 583, 720, 603, 586, 747, 1023, 753, 704, 313, 931, 598, 543, 1098, 922, 125, 127, 844, 45, 742, 923, 378, 29, 532, 886, 184, 634, 580, 746, 494, 415, 764, 950, 668, 182, 436, 758, 716, 180, 858, 231, 670, 624, 519, 1093, 975, 655, 430, 796, 559, 404, 25, 955, 1061, 579, 639, 498, 650, 1035, 131, 615, 283, 1072, 70, 212, 464, 542, 433, 538, 800, 754, 872, 830, 492, 1084, 698, 1020, 784, 626, 949, 993, 724, 833, 768, 434, 418, 616, 584, 663, 259, 99, 554, 718, 335, 480, 597, 737, 405, 483, 734, 541, 961, 1062, 501, 487, 962, 111, 612, 304, 733, 1039, 489, 161, 776, 485, 552, 1099, 420, 903, 835, 188, 1034, 738, 477, 744, 369, 156, 780, 79, 1046, 896, 121, 1059, 194, 51, 671, 13, 189, 147, 1092, 813],
                                                    dic = {813: 19970, 1092: 14281, 147: 13522, 189: 10375, 13: 9151, 671: 8419, 51: 7615, 194: 7394, 1059: 6564, 121: 6542, 896: 5955, 1046: 5591, 79: 5382, 780: 5259, 156: 5163, 369: 4416, 744: 3890, 477: 3692, 738: 3665, 1034: 3570, 188: 3500, 835: 3005, 903: 2552, 420: 2548, 1099: 2327, 552: 2180, 485: 2097, 776: 2075, 161: 2050, 489: 2045, 1039: 2001, 733: 1895, 304: 1881, 612: 1789, 111: 1762, 962: 1744, 487: 1685, 501: 1667, 1062: 1540, 961: 1526, 541: 1492, 734: 1480, 483: 1472, 405: 1457, 737: 1446, 597: 1428, 480: 1414, 335: 1403, 718: 1397, 554: 1390, 99: 1327, 259: 1302, 663: 1286, 584: 1283, 616: 1278, 418: 1260, 434: 1213, 768: 1200, 833: 1138, 724: 1083, 993: 1073, 949: 1053, 626: 1050, 784: 1016, 1020: 977, 698: 965, 1084: 961, 492: 959, 830: 957, 872: 956, 754: 921, 800: 902, 538: 901, 433: 888, 542: 866, 464: 859, 212: 838, 70: 831, 1072: 816, 283: 810, 615: 810, 131: 787, 1035: 775, 650: 773, 498: 766, 639: 750, 579: 745, 1061: 744, 955: 682, 25: 676, 404: 676, 559: 672, 796: 637, 430: 636, 655: 630, 975: 627, 1093: 622, 519: 622, 624: 613, 670: 600, 231: 595, 858: 593, 180: 593, 716: 583, 758: 582, 436: 582, 182: 580, 668: 570, 950: 568, 764: 566, 415: 565, 494: 564, 746: 560, 580: 555, 634: 551, 184: 545, 886: 543, 532: 540, 29: 530, 378: 523, 923: 518, 742: 515, 45: 515, 844: 504, 127: 499, 125: 494, 922: 493, 1098: 487, 543: 481, 598: 480, 931: 477, 313: 467, 704: 462, 753: 461, 1023: 454, 747: 453, 586: 446, 603: 440, 720: 433, 583: 427, 952: 422, 774: 422, 953: 413, 14: 408, 227: 403, 682: 399, 821: 393, 339: 388, 582: 387, 974: 382, 428: 373, 1022: 371, 703: 369, 954: 369, 308: 368, 239: 362, 684: 358, 465: 357, 809: 355, 520: 354, 462: 338, 573: 332, 581: 332, 940: 329, 558: 329, 348: 328, 592: 327, 545: 327, 438: 323, 725: 322, 322: 317, 925: 312, 1019: 303, 633: 303, 497: 302, 493: 298, 665: 297, 595: 296, 506: 295, 331: 293, 33: 292, 916: 290, 587: 289, 745: 286, 503: 284, 236: 281, 1032: 273, 515: 273, 568: 270, 803: 269, 202: 269, 647: 266, 766: 265, 517: 263, 991: 262, 783: 262, 978: 258, 707: 257, 691: 257, 553: 256, 726: 256, 490: 254, 432: 254, 133: 254, 994: 253, 488: 253, 1089: 251, 1030: 251, 23: 244, 534: 244, 524: 243, 607: 240, 58: 240, 840: 239, 1064: 236, 637: 236, 786: 234, 282: 233, 819: 233, 690: 230, 728: 229, 1085: 228, 884: 227, 450: 226, 834: 225, 822: 224, 908: 223, 482: 221, 713: 221, 1045: 220, 332: 219, 445: 219, 695: 219, 756: 218, 971: 216, 563: 216, 864: 215, 354: 210, 721: 208, 252: 207, 1012: 207, 890: 206, 570: 205, 739: 205, 518: 205, 889: 204, 701: 204, 672: 203, 705: 202, 929: 200, 678: 196, 1056: 195, 832: 194, 1009: 193, 983: 188, 495: 186, 628: 186, 548: 185, 927: 184, 811: 182, 963: 181, 555: 181, 1038: 180, 910: 179, 870: 179, 26: 177, 631: 177, 675: 176, 988: 174, 992: 173, 924: 170, 620: 170, 426: 169, 688: 167, 871: 167, 621: 165, 1040: 162, 731: 160, 1001: 160, 638: 160, 973: 159, 1097: 158, 306: 157, 989: 156, 692: 156, 849: 155, 826: 153, 522: 153, 280: 152, 551: 150, 676: 150, 627: 150, 996: 148, 972: 148, 383: 148, 791: 147, 912: 147, 932: 147, 591: 146, 78: 146, 863: 144, 1081: 143, 1069: 143, 425: 142, 41: 141, 888: 141, 448: 140, 887: 139, 1096: 137, 985: 137, 508: 137, 1033: 137, 576: 135, 76: 133, 680: 133, 496: 131, 825: 130, 829: 130, 915: 129, 741: 128, 116: 127, 511: 127, 1058: 126, 732: 124, 629: 124, 600: 124, 1037: 124, 507: 123, 666: 123, 18: 121, 535: 121, 669: 120, 1068: 120, 708: 120, 999: 120, 749: 119, 948: 119, 35: 119, 622: 119, 359: 118, 393: 118, 179: 118, 619: 118, 510: 117, 204: 116, 413: 115, 654: 115, 566: 114, 709: 114, 217: 114, 1083: 114, 454: 114, 1071: 113, 947: 113, 794: 113, 642: 112, 750: 112, 347: 112, 681: 111, 273: 111, 468: 111, 808: 111, 907: 110, 1065: 110, 516: 110, 649: 108, 192: 108, 882: 107, 55: 106, 479: 106, 422: 106, 861: 106, 459: 104, 1018: 104, 685: 104, 755: 103, 771: 103, 662: 103, 502: 102, 1066: 100, 1: 100, 546: 99, 674: 99, 540: 98, 1024: 98, 529: 97, 966: 97, 967: 97, 234: 97, 960: 97, 823: 97, 995: 96, 939: 96, 959: 96, 287: 96, 400: 95, 505: 95, 185: 95, 1025: 95, 1051: 95, 866: 94, 921: 94, 860: 94, 135: 94, 693: 93, 645: 93, 253: 93, 138: 93, 285: 93, 536: 92, 898: 92, 1080: 92, 585: 92, 763: 91, 409: 91, 608: 90, 664: 90, 316: 90, 406: 89, 175: 88, 850: 88, 679: 88, 761: 88, 661: 87, 653: 87, 941: 87, 838: 86, 449: 86, 537: 86, 1055: 85, 567: 85, 1101: 85, 714: 85, 556: 84, 1015: 83, 258: 82, 43: 82, 334: 81, 513: 81, 1010: 81, 307: 81, 936: 81, 457: 81, 874: 80, 885: 80, 105: 80, 512: 80, 478: 79, 1036: 79, 942: 79, 836: 79, 804: 79, 867: 79, 466: 79, 846: 78, 596: 78, 1077: 78, 368: 77, 875: 77, 453: 77, 920: 77, 815: 76, 461: 76, 226: 76, 458: 75, 588: 75, 824: 74, 60: 74, 702: 73, 1091: 72, 32: 72, 486: 72, 65: 72, 102: 72, 410: 71, 613: 71, 275: 71, 232: 71, 61: 71, 528: 70, 521: 70, 816: 70, 155: 69, 191: 69, 842: 69, 735: 69, 618: 69, 1013: 68, 1007: 68, 463: 68, 700: 68, 390: 68, 879: 67, 965: 67, 565: 67, 795: 67, 648: 67, 1003: 67, 906: 66, 1053: 66, 710: 66, 905: 66, 1041: 66, 881: 66, 437: 66, 444: 66, 373: 66, 964: 65, 968: 65, 895: 64, 914: 64, 1027: 64, 769: 64, 345: 64, 154: 64, 210: 63, 439: 63, 338: 63, 696: 63, 248: 63, 611: 63, 827: 62, 564: 62, 1088: 62, 218: 62, 1095: 62, 557: 61, 294: 61, 96: 61, 956: 61, 610: 61, 986: 60, 351: 60, 767: 60, 113: 60, 289: 60, 384: 59, 699: 59, 862: 59, 91: 59, 1002: 59, 891: 59, 1047: 59, 593: 58, 412: 58, 901: 58, 606: 58, 447: 58, 657: 58, 114: 58, 899: 58, 1050: 57, 1044: 57, 667: 57, 723: 57, 793: 57, 577: 57, 807: 56, 590: 56, 276: 56, 817: 56, 269: 56, 549: 56, 913: 56, 162: 55, 1054: 55, 706: 55, 1005: 55, 937: 54, 578: 54, 677: 53, 1073: 53, 423: 53, 806: 53, 163: 53, 646: 52, 145: 52, 173: 52, 1004: 52, 323: 52, 911: 52, 604: 52, 1078: 52, 514: 52, 660: 51, 828: 51, 934: 51, 878: 51, 605: 51, 976: 51, 623: 51, 977: 51, 641: 50, 748: 50, 349: 50, 42: 50, 1006: 50, 569: 50, 868: 49, 1057: 49, 772: 49, 712: 49, 673: 48, 399: 48, 40: 48, 533: 48, 1063: 48, 424: 48, 1043: 48, 797: 48, 256: 48, 951: 48, 1086: 48, 711: 47, 760: 47, 574: 47, 446: 47, 467: 47, 820: 47, 318: 46, 782: 46, 876: 46, 778: 46, 571: 46, 998: 45, 831: 45, 865: 45, 110: 45, 562: 45, 321: 45, 1049: 45, 683: 44, 757: 44, 86: 44, 196: 44, 375: 44, 263: 43, 770: 43, 274: 43, 69: 43, 473: 43, 95: 43, 1026: 42, 902: 42, 401: 41, 938: 41, 451: 41, 427: 41, 969: 41, 24: 41, 470: 41, 107: 41, 360: 40, 309: 40, 124: 40, 818: 40, 484: 40, 773: 40, 1074: 40, 272: 39, 572: 39, 324: 39, 560: 39, 945: 39, 636: 39, 178: 38, 279: 38, 1000: 38, 237: 38, 946: 38, 144: 38, 686: 38, 775: 38, 315: 37, 740: 37, 848: 37, 526: 37, 149: 37, 926: 37, 729: 36, 195: 36, 411: 36, 980: 36, 421: 36, 792: 36, 158: 36, 984: 36, 935: 36, 614: 36, 979: 36, 530: 35, 441: 35, 251: 35, 656: 35, 429: 35, 382: 35, 44: 35, 722: 35, 759: 34, 48: 34, 1075: 34, 442: 34, 1070: 34, 715: 34, 15: 34, 777: 34, 659: 34, 475: 34, 1082: 33, 292: 33, 57: 33, 249: 33, 789: 32, 385: 32, 257: 32, 342: 32, 788: 32, 847: 32, 134: 32, 122: 32, 893: 31, 408: 31, 651: 31, 1102: 31, 539: 31, 90: 30, 299: 30, 531: 30, 265: 30, 1100: 30, 730: 30, 205: 30, 990: 30, 509: 30, 958: 30, 894: 30, 141: 29, 997: 29, 358: 29, 397: 29, 900: 29, 74: 29, 1014: 29, 500: 29, 50: 29, 630: 28, 270: 28, 617: 28, 419: 28, 83: 28, 1042: 28, 499: 28, 957: 28, 9: 28, 877: 27, 837: 27, 717: 27, 609: 27, 839: 27, 320: 27, 547: 27, 169: 27, 440: 26, 151: 26, 933: 26, 109: 26, 1016: 26, 326: 26, 471: 26, 472: 26, 165: 26, 640: 26, 751: 26, 658: 26, 762: 25, 387: 25, 1079: 25, 469: 25, 106: 25, 504: 25, 859: 25, 801: 24, 550: 24, 602: 24, 625: 24, 810: 24, 455: 24, 869: 24, 75: 24, 802: 24, 689: 23, 120: 23, 1008: 23, 814: 23, 325: 23, 171: 23, 150: 22, 67: 22, 398: 22, 402: 22, 59: 22, 743: 22, 785: 22, 909: 22, 341: 22, 841: 22, 880: 22, 245: 22, 93: 22, 22: 22, 222: 22, 694: 22, 1067: 22, 374: 22, 981: 22, 857: 21, 1029: 21, 170: 21, 247: 21, 601: 21, 417: 21, 229: 21, 632: 21, 216: 21, 350: 21, 1011: 21, 456: 21, 380: 21, 781: 20, 1028: 20, 379: 20, 943: 20, 736: 20, 851: 20, 928: 20, 765: 20, 17: 20, 403: 19, 362: 19, 852: 19, 414: 19, 255: 19, 136: 19, 98: 19, 589: 19, 377: 19, 435: 19, 416: 19, 89: 19, 407: 18, 223: 18, 153: 18, 0: 18, 944: 18, 970: 18, 371: 18, 687: 18, 337: 18, 481: 18, 1090: 17, 266: 17, 52: 17, 1021: 17, 525: 17, 261: 17, 336: 17, 117: 17, 790: 17, 228: 17, 5: 17, 344: 17, 982: 17, 644: 17, 317: 16, 357: 16, 1087: 16, 353: 16, 575: 16, 491: 16, 856: 16, 301: 16, 392: 16, 172: 16, 85: 16, 799: 16, 361: 15, 300: 15, 208: 15, 930: 15, 319: 15, 443: 14, 139: 14, 1076: 14, 152: 14, 2: 14, 594: 14, 853: 14, 157: 14, 8: 14, 77: 14, 12: 14, 267: 14, 238: 14, 63: 13, 64: 13, 27: 13, 1052: 13, 1048: 13, 330: 13, 97: 13, 213: 13, 352: 13, 4: 13, 49: 12, 244: 12, 72: 12, 148: 12, 1094: 12, 697: 12, 918: 12, 28: 12, 16: 12, 356: 12, 246: 12, 779: 12, 295: 11, 386: 11, 62: 11, 1031: 11, 206: 11, 897: 11, 474: 11, 66: 11, 46: 11, 243: 10, 225: 10, 719: 10, 209: 10, 343: 10, 54: 10, 87: 10, 220: 10, 10: 10, 56: 10, 310: 10, 164: 10, 752: 10, 302: 10, 38: 10, 277: 10, 919: 10, 143: 10, 296: 9, 854: 9, 452: 9, 303: 9, 523: 9, 643: 9, 476: 9, 193: 9, 200: 9, 47: 9, 305: 9, 183: 9, 798: 9, 288: 9, 395: 8, 599: 8, 174: 8, 812: 8, 168: 8, 84: 8, 652: 8, 381: 8, 460: 8, 254: 8, 394: 8, 561: 8, 215: 8, 68: 8, 364: 7, 37: 7, 346: 7, 101: 7, 333: 7, 224: 7, 370: 7, 845: 7, 355: 7, 635: 7, 917: 7, 544: 7, 892: 7, 119: 7, 340: 7, 159: 7, 73: 7, 128: 7, 235: 7, 241: 7, 19: 7, 242: 7, 197: 7, 181: 6, 264: 6, 219: 6, 118: 6, 311: 6, 388: 6, 527: 6, 137: 6, 363: 6, 314: 6, 186: 6, 130: 6, 883: 6, 167: 6, 843: 6, 82: 6, 391: 6, 286: 6, 126: 6, 431: 5, 80: 5, 31: 5, 21: 5, 268: 5, 211: 5, 365: 5, 298: 5, 873: 5, 297: 5, 207: 5, 30: 5, 1060: 5, 115: 5, 284: 4, 327: 4, 88: 4, 132: 4, 855: 4, 20: 4, 214: 4, 176: 4, 92: 4, 34: 4, 260: 4, 166: 4, 290: 4, 177: 4, 291: 4, 6: 4, 278: 4, 129: 4, 233: 4, 787: 4, 103: 4, 367: 4, 198: 3, 36: 3, 727: 3, 53: 3, 100: 3, 7: 3, 1017: 3, 39: 3, 140: 3, 203: 3, 250: 3, 329: 3, 372: 3, 376: 3, 3: 3, 190: 3, 123: 3, 160: 3, 94: 3, 71: 2, 187: 2, 312: 2, 108: 2, 201: 2, 271: 2, 240: 2, 987: 2, 904: 2, 142: 2, 389: 2, 396: 1, 262: 1, 328: 1, 230: 1, 104: 1, 281: 1, 805: 1, 199: 1, 146: 1, 112: 1, 221: 1, 293: 1, 11: 1, 81: 1, 366: 1})
            # class -> threshold
            # class -> freq

            report = report + """
        Best Threshold is: {}, Score: {}, AreaUnder: {}""".format(best_threshold, best_val, total_score/total_tried)
            tensorboardwriter.write_best_threshold(self.writer, -1, best_val, best_threshold, total_score/total_tried, config.epoch, config.fold)
            # for c in range(config.TRAIN_NUM_CLASS): tensorboardwriter.write_best_threshold(self.writer, c, best_val_dict[c], best_threshold_dict[c], config.epoch, config.fold)

        print(report)
        tensorboardwriter.write_text(self.writer, report, config.global_steps[config.fold])

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
                if config.TRAIN_GPU_ARG: image = image.cuda()
                empty_logits, _idkwhatthisis_, logits_predict = net(image)
                prob_predict = torch.nn.Sigmoid()(logits_predict)
                prob_empty = torch.nn.Sigmoid()(empty_logits)

                print(labels)
                print(logits_predict)
                """LOSS"""
                if config.TRAIN_GPU_ARG:
                    labels = labels.cuda()
                    empty = empty.cuda().float()
                dice = denoised_siim_dice(threshold=config.EVAL_THRESHOLD, iou=False, denoised=False)(labels, logits_predict)
                bce = BCELoss()(prob_empty, empty)
                loss = dice.mean() + bce.mean()

                """BACKPROP"""
                loss.backward()
                if config.epoch > config.TRAIN_GRADIENT_ACCUMULATION:
                    if (batch_index + 1) % config.TRAIN_GRADIENT_ACCUMULATION == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    elif batch_index + 1 == len(train_loader): # drop last
                        optimizer.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()


                """DETATCH"""
                dice = dice.detach().cpu().numpy().mean()
                loss = loss.detach().cpu().numpy().mean()
                labels = labels.cpu().numpy()
                empty = empty.cpu().numpy()
                logits_predict = logits_predict.detach().cpu().numpy()
                prob_predict = prob_predict.detach().cpu().numpy()
                prob_empty = prob_empty.detach().cpu().numpy()
                # print(image)

                """SUM"""
                epoch_loss = epoch_loss + loss.mean()
                # f = f1_macro(logits_predict, labels).mean()
                confidence = np.absolute(prob_predict - 0.5).mean() + 0.5
                total_confidence = total_confidence + confidence

                """DISPLAY"""
                tensorboardwriter.write_memory(self.writer, "train")

                pbar.set_description_str("(E{}-F{}) Stp:{} Label:{} Pred:{} Conf:{:.4f} lr:{}".format(config.epoch, config.fold, int(config.global_steps[fold]), label, pred, total_confidence / (batch_index + 1), optimizer.param_groups[0]['lr']))
                out_dict = {'Epoch/{}'.format(config.fold): config.epoch,
                            'LearningRate{}/{}'.format(optimizer.__class__.__name__, config.fold): optimizer.param_groups[0]['lr'],
                            'Loss/{}'.format(config.fold): loss,
                            'Dice/{}'.format(config.fold): dice,
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
                del ids, image, image_0
                del dice, loss, labels, logits_predict, prob_predict, prob_empty
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
        """.format(config.epoch, config.fold, train_loss), config.global_steps[config.fold]-1)
        # lr_scheduler.step(epoch_f, epoch=config.epoch)

        del train_loss

        # if config.DISPLAY_HISTOGRAM:
        #     for i, (name, param) in enumerate(net.named_parameters()):
        #         print("Calculating Histogram #{}".format(i))
        #         writer.add_histogram(name, param.clone().cpu().data.numpy(), config.epoch)
        if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()  # release gpu memory


class SIIMEvaluation:
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
        pbar = tqdm(range(config.EVAL_RATIO)) if config.epoch >= config.MODEL_FREEZE_EPOCH +2 else tqdm(range(1))
        for eval_index in pbar:
            config.eval_index = eval_index
            pbar = tqdm(validation_loader)
            total_confidence = 0

            for batch_index, (ids, image, labels, image_0, labels_0, empty) in enumerate(pbar):

                """CALCULATE LOSS"""
                if config.TRAIN_GPU_ARG:
                    image = image.cuda()
                    labels_0 = labels_0.cuda()

                logits_predict = net(image)
                if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
                prob_predict = torch.nn.Sigmoid()(logits_predict)

                """LOSS"""
                focal = focalloss_sigmoid_refined(alpha=0.25, gamma=5, eps=1e-7)(labels_0, logits_predict)
                f, precise, recall = differenciable_f_sigmoid(beta=2)(labels_0, logits_predict)
                bce = BCELoss()(prob_predict, labels_0)

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
                bce = bce.detach().cpu().numpy().mean()
                # positive_bce = positive_bce.detach().cpu().numpy().mean()
                # loss = loss.detach().cpu().numpy()
                labels_0 = labels_0.cpu().numpy()
                image = image.cpu().numpy()
                image_0 = image_0.numpy()
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
                if config.DISPLAY_VISUALIZATION and batch_index < max(1, config.MODEL_BATCH_SIZE / 32): self.display(config.fold, ids, image, image_0, labels_0, prob_predict, focal)

                """CLEAN UP"""
                del ids, image, image_0
                del focal, f, bce, precise, recall, labels_0, logits_predict, prob_predict
                if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
                if config.DEBUG_TRAISE_GPU: gpu_profile(frame=sys._getframe(), event='line', arg=None)
            del pbar
            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
        """LOSS"""
        f = fbeta_score_numpy(label_total, predict_total, beta=2, threshold=config.EVAL_THRESHOLD)
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
