import os

import pandas as pd
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm

import config
from dataset.hpa_dataset import HPAData, train_collate
from project.hpa_project.hpa_model import se_resnext101_32x4d_modified
from utils.load import save_onnx, cuda, load_checkpoint_all_fold_without_optimizers


class HPAPrediction:
    def __init__(self, writer):
        self.thresholds = config.PREDICTION_CHOSEN_THRESHOLD
        self.writer = writer
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

        self.test_dataset = HPAData(config.DIRECTORY_CSV, load_img_dir=config.DIRECTORY_PREPROCESSED_IMG, img_suffix=config.DIRECTORY_PREPROCESSED_SUFFIX_IMG, load_strategy="test", load_preprocessed_dir=True)

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

                prob_path = "{}-{}-F{}-T{}-Prob.csv".format(config.DIRECTORY_LOAD, config.PREDICTION_TAG, fold, threshold)
                if os.path.exists(prob_path):
                    os.remove(prob_path)
                    print("WARNING: delete file '{}'".format(prob_path))

                lb_path = "{}-{}-F{}-T{}-LB.csv".format(config.DIRECTORY_LOAD, config.PREDICTION_TAG, fold, threshold)
                if os.path.exists(lb_path):
                    os.remove(lb_path)
                    print("WARNING: delete file '{}'".format(lb_path))

                with open(pred_path, 'a') as pred_file, open(prob_path, 'a') as prob_file, open(lb_path, 'a') as lb_file:
                    pred_file.write('Id,Predicted\n')
                    prob_file.write('Id,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27\n')
                    lb_file.write('Id,Predicted\n')

                    test_loader = data.DataLoader(self.test_dataset,
                                                  batch_size=config.MODEL_BATCH_SIZE,
                                                  shuffle=False,
                                                  sampler=SubsetRandomSampler(self.test_dataset.indices),
                                                  batch_sampler=None,
                                                  num_workers=config.TRAIN_NUM_WORKER,
                                                  collate_fn=train_collate,
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
                        predicts = torch.sigmoid(predicts).detach().cpu().numpy()
                        encodeds = list(self.test_dataset.multilabel_binarizer.inverse_transform(predicts > 0.5))
                        pbar.set_description("Thres:{} Id:{} Prob:{} Out:{}".format(threshold, ids[0], np.absolute(predicts[0]-0.5).mean()+0.5, encodeds[0]))

                        for id, encoded, predict in zip(ids, encodeds, predicts):
                            pred_file.write('{},{}\n'.format(id, " ".join(str(x) for x in encoded)))
                            prob_file.write('{},{}\n'.format(id, ",".join(str(x) for x in predict)))
                            lb_file.write('{},{}\n'.format(id, " ".join(str(x) for x in encoded if x not in [8, 9, 10, 15, 20, 24, 27])))
                            # figure = plt.figure()
                            #
                            # plt.subplot(121)
                            # plt.imshow(untransfered/255., vmin=0, vmax=1)
                            # plt.title("Image_Real; pred:{}".format(encoded))
                            # plt.grid(False)
                            # plt.subplot(122)
                            # plt.imshow(encode.tensor_to_np_three_channel_with_green(np.array(input[0])), vmin=0, vmax=1)
                            # plt.title("Image_Trans")
                            # plt.grid(False)
                            # tensorboardwriter.write_predict_image(self.writer, "e{}-{}-{}".format(config.epoch, fold, id), figure, config.epoch)

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
                sort(config.DIRECTORY_SAMPLE_CSV, lb_path)
                print("Pred_path: {}".format(pred_path))
                print("Lb_path: {}".format(lb_path))
                print("Prob_path: {}".format(prob_path))

class HPATest:
    def __init__(self, writer):
        self.thresholds = config.PREDICTION_CHOSEN_THRESHOLD
        self.writer = writer
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

        self.test_dataset = HPAData("scripts/augment.csv", load_img_dir="scripts/images/", img_suffix=".jpg", load_strategy="test", load_preprocessed_dir=False, column='Label')

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

                prob_path = "{}-{}-F{}-T{}-Prob.csv".format(config.DIRECTORY_LOAD, config.PREDICTION_TAG, fold, threshold)
                if os.path.exists(prob_path):
                    os.remove(prob_path)
                    print("WARNING: delete file '{}'".format(prob_path))

                lb_path = "{}-{}-F{}-T{}-LB.csv".format(config.DIRECTORY_LOAD, config.PREDICTION_TAG, fold, threshold)
                if os.path.exists(lb_path):
                    os.remove(lb_path)
                    print("WARNING: delete file '{}'".format(lb_path))

                with open(pred_path, 'a') as pred_file, open(prob_path, 'a') as prob_file, open(lb_path, 'a') as lb_file:
                    pred_file.write('Id,Predicted\n')
                    prob_file.write('Id,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27\n')
                    lb_file.write('Id,Predicted\n')

                    test_loader = data.DataLoader(self.test_dataset,
                                                  batch_size=config.MODEL_BATCH_SIZE,
                                                  shuffle=False,
                                                  sampler=SubsetRandomSampler(self.test_dataset.indices),
                                                  batch_sampler=None,
                                                  num_workers=config.TRAIN_NUM_WORKER,
                                                  collate_fn=train_collate,
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
                        predicts = torch.sigmoid(predicts).detach().cpu().numpy()
                        encodeds = list(self.test_dataset.multilabel_binarizer.inverse_transform(predicts > 0.5))
                        pbar.set_description("Thres:{} Id:{} Prob:{} Out:{}".format(threshold, ids[0], np.absolute(predicts[0]-0.5).mean()+0.5, encodeds[0]))

                        for id, encoded, predict in zip(ids, encodeds, predicts):
                            pred_file.write('{},{}\n'.format(id, " ".join(str(x) for x in encoded)))
                            prob_file.write('{},{}\n'.format(id, ",".join(str(x) for x in predict)))
                            lb_file.write('{},{}\n'.format(id, " ".join(str(x) for x in encoded if x not in [8, 9, 10, 15, 20, 24, 27])))

                        del ids, image, labels_0, image_for_display, predicts, encodeds
                        if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
                """TURNING THRESHOLD"""