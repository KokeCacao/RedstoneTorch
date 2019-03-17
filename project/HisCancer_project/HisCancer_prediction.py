import os

import torch
import numpy as np
import pandas as pd
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm

import config
from dataset.HisCancer_dataset import HisCancerDataset, test_collate
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
                prob_path = "{}-{}-F{}-T{}-Prob.csv".format(config.DIRECTORY_LOAD, config.PREDICTION_TAG, fold, threshold)
                if os.path.exists(prob_path):
                    os.remove(prob_path)
                    print("WARNING: delete file '{}'".format(prob_path))

                with open(prob_path, 'a') as prob_file:
                    prob_file.write('Id,Label\n')

                    test_loader = data.DataLoader(self.test_dataset,
                                                  batch_size=config.MODEL_BATCH_SIZE,
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

                    tta_list = []
                    for tta in range(config.PREDICTION_TTA):
                        tta_dict = []
                        config.eval_index = config.eval_index + 1
                        for batch_index, (ids, image, labels_0, image_for_display) in enumerate(pbar):

                            if config.TRAIN_GPU_ARG: image = image.cuda()
                            predicts = self.nets[fold](image)
                            predicts = torch.nn.Softmax()(predicts).detach().cpu().numpy()

                            confidence = np.absolute(predicts-0.5).mean()+0.5
                            total_confidence = total_confidence + confidence
                            pbar.set_description("Thres:{} Id:{} Confidence:{}/{}".format(threshold, ids[0].replace("data/HisCancer_dataset/test/", "").replace(".npy", ""), confidence, total_confidence/(batch_index+1)))

                            for id, predict in zip(ids, predicts):
                                id = id.replace("data/HisCancer_dataset/test/", "").replace(".npy", "")
                                tta_dict.append('{},{}\n'.format(id, str(predict[1])))

                                # prob_file.write('{},{}\n'.format(id, " ".join(str(x) for x in predict)))

                            del ids, image, labels_0, image_for_display, predicts
                            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
                        tta_list.append(tta_dict)

                    for tta in tta_list:
                        for item in tta:
                            prob_file.write(item)