import os

import torch
import numpy as np
import pandas as pd
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm

import config
from dataset.HisCancer_dataset import HisCancerDataset, val_collate
from project.HisCancer_project import HisCancer_net
from utils.load import load_checkpoint_all_fold_without_optimizers, save_onnx


class HisCancerPrediction:
    def __init__(self, writer):
        self.thresholds = config.PREDICTION_CHOSEN_THRESHOLD
        self.writer = writer
        self.nets = []
        for fold in range(config.MODEL_FOLD):
            if fold not in config.MODEL_TRAIN_FOLD:
                print("     Junping Fold: #{}".format(fold))
            else:
                print("     Creating Fold: #{}".format(fold))
                net = HisCancer_net.se_resnext50_32x4d(config.TRAIN_NUM_CLASS, pretrained="imagenet")

                """ONNX"""
                if config.DISPLAY_SAVE_ONNX and config.DIRECTORY_LOAD: save_onnx(net, (config.MODEL_BATCH_SIZE, 4, config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE), config.DIRECTORY_LOAD + ".onnx")

                if config.TRAIN_GPU_ARG: net = torch.nn.DataParallel(net, device_ids=config.TRAIN_GPU_LIST)
                self.nets.append(net)
        load_checkpoint_all_fold_without_optimizers(self.nets, config.DIRECTORY_LOAD)

        self.test_dataset = HisCancerDataset(config.DIRECTORY_CSV, config.DIRECTORY_SAMPLE_CSV, load_strategy="predict", writer=self.writer, column='Label')

        self.run()

    def run(self):
        torch.no_grad()
        """Used for Kaggle submission: predicts and encode all test images"""
        for fold, net in enumerate(self.nets):
            for threshold in self.thresholds:
                pred_path = "{}-{}-F{}-T{}.csv".format(config.DIRECTORY_LOAD, config.PREDICTION_TAG, fold, threshold)
                if os.path.exists(pred_path):
                    os.remove(pred_path)
                    print("WARNING: delete file '{}'".format(pred_path))

                with open(pred_path, 'a') as pred_file:
                    pred_file.write('Id,Predicted\n')

                    test_loader = data.DataLoader(self.test_dataset,
                                                  batch_size=config.MODEL_BATCH_SIZE,
                                                  shuffle=False,
                                                  sampler=SubsetRandomSampler(self.test_dataset.indices),
                                                  batch_sampler=None,
                                                  num_workers=config.TRAIN_NUM_WORKER,
                                                  collate_fn=val_collate,
                                                  pin_memory=True,
                                                  drop_last=False,
                                                  timeout=0,
                                                  worker_init_fn=None,
                                                  )
                    pbar = tqdm(test_loader)
                    print("Set Model Trainning mode to trainning=[{}]".format(net.eval().training))
                    for batch_index, (ids, image, labels_0, image_for_display) in enumerate(pbar):

                        if config.TRAIN_GPU_ARG: image = image.cuda()
                        predicts = self.nets[0](image)
                        predicts = torch.nn.Softmax()(predicts).detach().cpu().numpy()
                        encodeds = list(self.test_dataset.multilabel_binarizer.inverse_transform(predicts > threshold))
                        pbar.set_description("Thres:{} Id:{} Certainty:{} Out:{}".format(threshold, ids[0], np.absolute(predicts-0.5).mean()+0.5, encodeds[0]))


                        del ids, image, labels_0, image_for_display, predicts, encodeds
                        if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
                """TURNING THRESHOLD"""


                """ORGANIZE"""
                def sort(dir_sample, dir_save):
                    f1 = pd.read_csv(dir_sample)
                    f1.drop('Predicted', axis=1, inplace=True)
                    f2 = pd.read_csv(dir_save)
                    f1 = f1.merge(f2, left_on='Id', right_on='Id', how='outer')
                    os.remove(dir_save)
                    f1.to_csv(dir_save, index=False)
                sort(config.DIRECTORY_SAMPLE_CSV, pred_path)
                print("Pred_path: {}".format(pred_path))