import os

import torch
import numpy as np
import scipy.stats as st
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm

import config
from dataset.HisCancer_dataset import HisCancerDataset, test_collate, train_collate
from project.HisCancer_project import HisCancer_net
from utils.load import save_onnx, load_checkpoint_all_fold


class HisCancerPrediction:
    def __init__(self, writer):
        self.thresholds = config.PREDICTION_CHOSEN_THRESHOLD
        self.writer = writer
        self.nets = []
        for fold in range(config.MODEL_FOLD):
            if fold not in config.train_fold:
                print("     Junping Fold: #{}".format(fold))
                self.nets.append(None)
            else:
                print("     Creating Fold: #{}".format(fold))
                net = HisCancer_net.se_resnext50_32x4d(config.TRAIN_NUM_CLASS, pretrained="imagenet")
                # net = HisCancer_net.densenet169()

                """ONNX"""
                if config.DISPLAY_SAVE_ONNX and config.DIRECTORY_LOAD: save_onnx(net, (config.MODEL_BATCH_SIZE, 4, config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE), config.DIRECTORY_LOAD + ".onnx")

                if config.TRAIN_GPU_ARG: net = torch.nn.DataParallel(net, device_ids=config.TRAIN_GPU_LIST)
                self.nets.append(net)

        config.load_optimizers = False
        config.load_lr_schedulers = False
        load_checkpoint_all_fold(self.nets, None, None, config.DIRECTORY_LOAD)

        self.test_dataset = HisCancerDataset(config.DIRECTORY_CSV, config.DIRECTORY_SAMPLE_CSV, load_strategy="predict", writer=self.writer, column='Label')
        self.valid_dataset = HisCancerDataset(config.DIRECTORY_CSV, config.DIRECTORY_SAMPLE_CSV, load_strategy="train", writer=self.writer, column='Target')

        self.run()

    def run(self):
        torch.no_grad()
        """Used for Kaggle submission: predicts and encode all test images"""
        for fold, net in enumerate(self.nets):
            if net == None:
                continue
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
                    confidence_list = []

                    print("Set Model Trainning mode to trainning=[{}]".format(net.eval().training))

                    for batch_index, (ids, image, labels_0, image_for_display) in enumerate(pbar):

                        if config.TRAIN_GPU_ARG: image = image.cuda()
                        predicts = net(image)
                        predicts = torch.nn.Softmax()(predicts).detach().cpu().numpy()

                        confidence = (np.absolute(predicts - 0.5).mean() + 0.5).item()
                        total_confidence = total_confidence + confidence
                        confidence_list.append(confidence)
                        pbar.set_description("Thres:{} Id:{} Confidence:{}/{}".format(threshold, ids[0].replace("data/HisCancer_dataset/test/", "").replace(".npy", ""), confidence, total_confidence / (batch_index + 1)))

                        for id, predict in zip(ids, predicts):
                            prob_file.write('{},{}\n'.format(id.replace("data/HisCancer_dataset/test/", "").replace(".npy", ""), str(predict[1])))

                        del ids, image, labels_0, image_for_display, predicts
                        if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()

                # left, right = st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))
                print("""
                Mean Confidence = {}, STD = {}
                """.format(np.mean(confidence_list), np.std(confidence_list)))

                print("Prob_path: {}".format(prob_path))

                """TTA"""
                if config.PREDICTION_TTA != 0:
                    tta_path = "{}-{}-F{}-T{}-Prob-TTA.csv".format(config.DIRECTORY_LOAD, config.PREDICTION_TAG, fold, threshold)
                    if os.path.exists(tta_path):
                        os.remove(tta_path)
                        print("WARNING: delete file '{}'".format(tta_path))

                    with open(tta_path, 'a') as prob_file:
                        prob_file.write('Id,Label\n')

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

                        print("Set Model Trainning mode to trainning=[{}]".format(net.eval().training))

                        tta_list = []
                        tta_pbar = tqdm(range(config.PREDICTION_TTA))
                        for tta in tta_pbar:
                            tta_dict = dict()
                            config.eval_index = config.eval_index + 1
                            total_confidence = 0
                            pbar = tqdm(test_loader)
                            for batch_index, (ids, image, labels_0, image_for_display) in enumerate(pbar):

                                if config.TRAIN_GPU_ARG: image = image.cuda()
                                predicts = self.nets[fold](image)
                                predicts = torch.nn.Softmax()(predicts).detach().cpu().numpy()

                                confidence = np.absolute(predicts-0.5).mean()+0.5
                                total_confidence = total_confidence + confidence
                                pbar.set_description("Thres:{} Id:{} Confidence:{}/{}".format(threshold, ids[0].replace("data/HisCancer_dataset/test/", "").replace(".npy", ""), confidence, total_confidence/(batch_index+1)))

                                for id, predict in zip(ids, predicts):
                                    tta_dict[id.replace("data/HisCancer_dataset/test/", "").replace(".npy", "")]=('{}'.format(str(predict[1])))

                                    # prob_file.write('{},{}\n'.format(id, " ".join(str(x) for x in predict)))

                                del ids, image, labels_0, image_for_display, predicts
                                if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()
                            tta_list.append(tta_dict)


                        for item in tta_list[0].keys():
                            pred = ",".join(tta_list[i][item] for i in range(len(tta_list)))
                            prob_file.write("{},{}\n".format(item, pred))

                        print("TTA_path: {}".format(tta_path))

