import os

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm

import config
from dataset.siim_dataset import test_collate, tta_collate
from dataset.siim_dataset import SIIMDataset
from project.siim_project.siim_net import model50A_DeepSupervion
from utils.encode import mask2rle
from utils.load import save_onnx, load_checkpoint_all_fold


class SIIMPrediction:
    def __init__(self, writer):
        self.writer = writer
        self.nets = []
        self.test_dataset = SIIMDataset(config.DIRECTORY_CSV, config.DIRECTORY_SAMPLE_CSV, load_strategy="predict", writer=self.writer, id_col=config.DIRECTORY_CSV_ID, target_col=config.DIRECTORY_CSV_TARGET)

        for fold in range(config.MODEL_FOLD):
            if fold not in config.train_fold:
                print("     Skipping dataset = SIIMDataset(config.DIRECTORY_CSV, fold: #{})".format(fold))
                self.nets.append(None)
            else:
                print("     Creating Fold: #{}".format(fold))
                # net = siim_net.resunet(encoder_depth=50, num_classes=config.TRAIN_NUM_CLASS, num_filters=32, dropout_2d=0.2, pretrained=False, is_deconv=False)
                net = model50A_DeepSupervion(num_classes=config.TRAIN_NUM_CLASS, test=False)

                """ONNX"""
                if config.DISPLAY_SAVE_ONNX and config.DIRECTORY_LOAD: save_onnx(net, (config.MODEL_BATCH_SIZE, 4, config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE), config.DIRECTORY_LOAD + ".onnx")

                if config.TRAIN_GPU_ARG:
                    print("Let's use", torch.cuda.device_count(), "GPUs!")
                    net = torch.nn.DataParallel(net)
                self.nets.append(net)

        config.load_optimizers = False
        config.load_lr_schedulers = False
        load_checkpoint_all_fold(self.nets, None, None, config.DIRECTORY_LOAD)

        self.run()

    def run(self):
        torch.no_grad()
        """Used for Kaggle submission: predicts and encode all test images"""
        for fold, net in enumerate(self.nets):
            if net == None:
                continue
            net = net.cuda()
            for threshold in config.PREDICTION_CHOSEN_THRESHOLD:

                if config.PREDICTION_TTA == 0:
                    prob_path = "{}-{}-F{}-T{}-Prob.csv".format(config.DIRECTORY_LOAD, config.PREDICTION_TAG, fold, threshold)
                    if os.path.exists(prob_path):
                        os.remove(prob_path)
                        print("WARNING: delete file '{}'".format(prob_path))

                    with open(prob_path, 'a') as prob_file:
                        prob_file.write('{},{}\n'.format(config.DIRECTORY_CSV_ID, config.DIRECTORY_CSV_TARGET))

                        test_loader = data.DataLoader(self.test_dataset,
                                                      batch_size=config.TEST_BATCH_SIZE,
                                                      shuffle=False,
                                                      sampler=SubsetRandomSampler(self.test_dataset.indices),
                                                      batch_sampler=None,
                                                      num_workers=config.TRAIN_NUM_WORKER,
                                                      collate_fn=test_collate,
                                                      pin_memory=True,
                                                      drop_last=False,
                                                      timeout=0,
                                                      worker_init_fn=None,
                                                      )
                        pbar = tqdm(test_loader)
                        total_confidence = 0

                        print("Set Model Trainning mode to trainning=[{}]".format(net.eval().training))

                        for batch_index, (ids, image, labels, image_0, labels_0, empty) in enumerate(pbar):

                            if config.TRAIN_GPU_ARG: image = image.cuda()
                            empty_logits, _idkwhatthisis_, logits_predict = net(image)
                            prob_predict = torch.nn.Sigmoid()(logits_predict)
                            prob_empty = torch.nn.Sigmoid()(empty_logits)

                            image = image.cpu().numpy()
                            labels = labels.cpu().numpy()
                            empty = empty.cpu().numpy()
                            logits_predict = logits_predict.detach().cpu().numpy()
                            prob_predict = prob_predict.detach().cpu().numpy()
                            prob_empty = prob_empty.detach().cpu().numpy()

                            confidence = (np.absolute(prob_predict - 0.5).mean() + 0.5).item()
                            total_confidence = total_confidence + confidence
                            pbar.set_description("Thres:{} Id:{} Confidence:{}/{}".format(threshold, ids[0].split("/")[-1].split(".")[0], confidence, total_confidence / (batch_index + 1)))

                            prob_predict = (prob_predict > threshold).astype(np.byte)

                            for id, predict in zip(ids, prob_predict):
                                predict = predict.squeeze()
                                prob_file.write('{},{}\n'.format(id[0].split("/")[-1].split(".")[0], mask2rle(predict, config.IMG_SIZE, config.IMG_SIZE)))

                            del ids, image, labels, image_0, labels_0, empty
                            del empty_logits, _idkwhatthisis_, logits_predict
                            del prob_predict, prob_empty, confidence, total_confidence
                            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()

                    print("Prob_path: {}".format(prob_path))
                else:
                    """TTA"""
                    tta_path = "{}-{}-F{}-T{}-Prob-TTA.csv".format(config.DIRECTORY_LOAD, config.PREDICTION_TAG, fold, threshold)
                    if os.path.exists(tta_path):
                        os.remove(tta_path)
                        print("WARNING: delete file '{}'".format(tta_path))

                    with open(tta_path, 'a') as prob_file:
                        prob_file.write('{},{}\n'.format(config.DIRECTORY_CSV_ID, config.DIRECTORY_CSV_TARGET))

                        if config.PREDICTION_TTA > 2:
                            test_loader = data.DataLoader(self.test_dataset,
                                                          batch_size=config.TEST_BATCH_SIZE,
                                                          shuffle=False,
                                                          sampler=SubsetRandomSampler(self.test_dataset.indices),
                                                          batch_sampler=None,
                                                          num_workers=config.TRAIN_NUM_WORKER,
                                                          collate_fn=tta_collate,
                                                          pin_memory=True,
                                                          drop_last=False,
                                                          timeout=0,
                                                          worker_init_fn=None,
                                                          )
                        else:
                            test_loader = data.DataLoader(self.test_dataset,
                                                          batch_size=config.TEST_BATCH_SIZE,
                                                          shuffle=False,
                                                          sampler=SubsetRandomSampler(self.test_dataset.indices),
                                                          batch_sampler=None,
                                                          num_workers=config.TRAIN_NUM_WORKER,
                                                          collate_fn=test_collate,
                                                          pin_memory=True,
                                                          drop_last=False,
                                                          timeout=0,
                                                          worker_init_fn=None,
                                                          )

                        print("Set Model Trainning mode to trainning=[{}]".format(net.eval().training))

                        tta_list = []
                        tta_pbar = tqdm(range(config.PREDICTION_TTA))
                        for tta in tta_pbar:
                            tta_dict = dict()
                            config.eval_index = config.eval_index + 1
                            total_confidence = 0
                            pbar = tqdm(test_loader)
                            for batch_index, (ids, image, labels, image_0, labels_0, empty) in enumerate(pbar):

                                image = image.cuda()
                                empty_logits, _idkwhatthisis_, logits_predict = net(image)
                                prob_predict = torch.nn.Sigmoid()(logits_predict)
                                prob_empty = torch.nn.Sigmoid()(empty_logits)

                                image = image.cpu().numpy()
                                labels = labels.cpu().numpy()
                                empty = empty.cpu().numpy()
                                logits_predict = logits_predict.detach().cpu().numpy()
                                prob_predict = prob_predict.detach().cpu().numpy()
                                prob_empty = prob_empty.detach().cpu().numpy()

                                confidence = (np.absolute(prob_predict - 0.5).mean() + 0.5).item()
                                total_confidence = total_confidence + confidence
                                pbar.set_description("Thres:{} Id:{} Confidence:{}/{}".format(threshold, ids[0].split("/")[-1].split(".")[0], confidence, total_confidence / (batch_index + 1)))

                                prob_predict = (prob_predict > threshold).astype(np.byte)
                                for id, predict in zip(ids, prob_predict):
                                    predict = predict.squeeze()
                                    print("DEBUG: {}".format(predict.shape))
                                    tta_dict[id[0].split("/")[-1].split(".")[0]] = mask2rle(predict, config.IMG_SIZE, config.IMG_SIZE)
                                    # prob_file.write('{},{}\n'.format(id[0].split("/")[-1].split(".")[0], mask2rle(predict, config.IMG_SIZE, config.IMG_SIZE)))

                                del ids, image, labels, image_0, labels_0, empty
                                del empty_logits, _idkwhatthisis_, logits_predict
                                del prob_predict, prob_empty, confidence, total_confidence
                                if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
                            tta_list.append(tta_dict)

                        for item in tta_list[0].keys():
                            pred = (tta_list[i][item] for i in range(len(tta_list)))
                            prob_file.write("{},{}\n".format(item, pred))

                        print("TTA_path: {}".format(tta_path))

