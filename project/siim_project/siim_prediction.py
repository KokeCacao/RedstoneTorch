import os
from datetime import datetime

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import SubsetRandomSampler, SequentialSampler
from tqdm import tqdm

import cv2
import config
from dataset.siim_dataset import SIIMDataset
from dataset.siim_dataset import test_collate, tta_collate
from net.seresunet34_scse_hyper import SEResUNetscSEHyper34, ResUNetscSEHyper32
from net.seresunext50_oc_scse_hyper import SeResUNeXtscSEOCHyper50, SeResUNeXtscSEOCHyper34
from project.siim_project import siim_net
from project.siim_project.siim_net import model50A_DeepSupervion, model34_DeepSupervion, model34_DeepSupervion_GroupNorm_OC, model34_DeepSupervion_GroupNorm
from project.siim_project.siim_util import post_process
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
                if config.net == "resunet50":
                    net = siim_net.resunet(encoder_depth=50, num_classes=config.TRAIN_NUM_CLASS, num_filters=32, dropout_2d=0.2, pretrained=False, is_deconv=False)
                elif config.net == "resunet50-ds":
                    net = model50A_DeepSupervion(num_classes=config.TRAIN_NUM_CLASS)
                elif config.net == "resunet34-ds":
                    net = model34_DeepSupervion(num_classes=config.TRAIN_NUM_CLASS)
                elif config.net == "resunet34-ds-gn":
                    net = model34_DeepSupervion_GroupNorm(num_classes=config.TRAIN_NUM_CLASS)
                elif config.net == "resunet34-ds-gn-oc":
                    net = model34_DeepSupervion_GroupNorm_OC(num_classes=config.TRAIN_NUM_CLASS)
                elif config.net == "seresunext50_oc_scse_hyper":
                    net = SeResUNeXtscSEOCHyper50(num_classes=config.TRAIN_NUM_CLASS, dilation=False)
                elif config.net == "seresunext50_oc_scse_hyper_dilate":
                    net = SeResUNeXtscSEOCHyper50(num_classes=config.TRAIN_NUM_CLASS, dilation=True)
                elif config.net == "seresunext34_oc_scse_hyper":
                    net = SeResUNeXtscSEOCHyper34(num_classes=config.TRAIN_NUM_CLASS, dilation=False)
                elif config.net == "seresunext34_oc_scse_hyper_dilate":
                    net = SeResUNeXtscSEOCHyper34(num_classes=config.TRAIN_NUM_CLASS, dilation=True)
                elif config.net == "seresunet34-ds-scse-hyper":
                    net = SEResUNetscSEHyper34(num_classes=config.TRAIN_NUM_CLASS, drop_out=0.1)
                elif config.net == "resunet32-ds-scse-hyper":
                    net = ResUNetscSEHyper32(num_classes=config.TRAIN_NUM_CLASS)
                ## leaky relu?
                else:
                    raise ValueError("The Network {} you specified is not in one of the network you can use".format(config.net))

                """ONNX"""
                if config.DISPLAY_SAVE_ONNX and config.DIRECTORY_LOAD: save_onnx(net, (config.MODEL_BATCH_SIZE, 4, config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE), config.DIRECTORY_LOAD + ".onnx")

                if config.TRAIN_GPU_ARG:
                    print("Let's use", torch.cuda.device_count(), "GPUs!")
                    net = torch.nn.DataParallel(net, device_ids=[i for i in range(torch.cuda.device_count())]) # dim=2 split image into half
                self.nets.append(net)

        config.load_optimizers = False
        config.load_lr_schedulers = False
        load_checkpoint_all_fold(self.nets, None, None, config.DIRECTORY_LOAD)

        with torch.no_grad(): self.run()

    def run(self):
        """Used for Kaggle submission: predicts and encode all test images"""
        for fold, net in enumerate(self.nets):
            if net == None:
                continue
            net = net.cuda()
            for threshold in config.PREDICTION_CHOSEN_THRESHOLD:
                if config.prediction_tta < 2:
                    test_loader = data.DataLoader(self.test_dataset,
                                                  batch_size=config.TEST_BATCH_SIZE,
                                                  shuffle=False,
                                                  sampler=SequentialSampler(self.test_dataset.indices),
                                                  batch_sampler=None,
                                                  num_workers=config.TRAIN_NUM_WORKER,
                                                  collate_fn=test_collate,
                                                  pin_memory=True,
                                                  drop_last=False,
                                                  timeout=0,
                                                  worker_init_fn=None,
                                                  )
                elif config.prediction_tta == 2:
                    test_loader = data.DataLoader(self.test_dataset,
                                                  batch_size=config.TEST_BATCH_SIZE,
                                                  shuffle=False,
                                                  sampler=SequentialSampler(self.test_dataset.indices),
                                                  batch_sampler=None,
                                                  num_workers=config.TRAIN_NUM_WORKER,
                                                  collate_fn=test_collate,
                                                  pin_memory=True,
                                                  drop_last=False,
                                                  timeout=0,
                                                  worker_init_fn=None,
                                                  )
                elif config.prediction_tta > 2:
                    test_loader = data.DataLoader(self.test_dataset,
                                                  batch_size=config.TEST_BATCH_SIZE,
                                                  shuffle=False,
                                                  sampler=SequentialSampler(self.test_dataset.indices),
                                                  batch_sampler=None,
                                                  num_workers=config.TRAIN_NUM_WORKER,
                                                  collate_fn=tta_collate,
                                                  pin_memory=True,
                                                  drop_last=False,
                                                  timeout=0,
                                                  worker_init_fn=None,
                                                  )
                else:
                    raise ValueError("tta cannot be {}".format(config.prediction_tta))

                if not os.path.exists(config.DIRECTORY_CHECKPOINT +  "prediction/"):
                    os.makedirs(config.DIRECTORY_CHECKPOINT +  "prediction/")
                prob_path = config.DIRECTORY_CHECKPOINT +  "prediction/{}-{}-{}-F{}-T{}-Prob.csv".format(str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-"), config.epoch, config.PREDICTION_TAG, fold, threshold)
                print("Creating Path: {}".format(prob_path))

                if not os.path.exists(os.path.dirname(prob_path)):
                    os.makedirs("".join(i for i in prob_path.split("/")[:-1]))

                if os.path.exists(prob_path):
                    os.remove(prob_path)
                    print("WARNING: delete file '{}'".format(prob_path))

                with open(prob_path, 'a') as prob_file:
                    prob_file.write('{},{},{}\n'.format(config.DIRECTORY_CSV_ID, config.DIRECTORY_CSV_TARGET, "Empty"))

                    test_loader = data.DataLoader(self.test_dataset,
                                                  batch_size=config.TEST_BATCH_SIZE,
                                                  shuffle=False,
                                                  sampler=SequentialSampler(self.test_dataset.indices),
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

                    id_total = None
                    predict_total = None
                    prob_empty_total = None
                    for batch_index, (ids, image, labels, image_0, labels_0, empty, flip) in enumerate(pbar):

                        image = image.cuda()
                        flip = flip.cuda().float()
                        empty_logits, _idkwhatthisis_, logits_predict = net(image)
                        prob_predict = torch.nn.Sigmoid()(logits_predict)
                        prob_empty = torch.nn.Sigmoid()(empty_logits)

                        image = image.cpu().numpy()
                        flip = flip.cpu().numpy()
                        labels = labels.cpu().numpy()
                        empty = empty.cpu().numpy()
                        logits_predict = logits_predict.detach().cpu().numpy()
                        prob_predict = prob_predict.detach().cpu().numpy()
                        prob_empty = prob_empty.detach().cpu().numpy()

                        id_total = np.concatenate((id_total, ids), axis=0) if id_total is not None else ids
                        predict_total = np.concatenate((predict_total, prob_predict), axis=0) if predict_total is not None else prob_predict
                        prob_empty_total = np.concatenate((prob_empty_total, prob_empty), axis=0) if prob_empty_total is not None else prob_empty

                        confidence = (np.absolute(prob_predict - 0.5).mean() + 0.5).item()
                        total_confidence = total_confidence + confidence
                        pbar.set_description("Thres:{} Id:{} Confidence:{}/{}".format(threshold, ids[0].split("/")[-1].split(".")[0], confidence, total_confidence / (batch_index + 1)))

                        for id, empty, predict in zip(ids, prob_empty, prob_predict):
                            predict = predict.squeeze()
                            predict = np.transpose(predict)
                            predict = cv2.resize(predict, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
                            predict, num_component = post_process(predict, threshold, config.PREDICTION_CHOSEN_MINPIXEL)
                            # predict, num_component = post_process(predict, threshold, config.PREDICTION_CHOSEN_MINPIXEL, empty=empty, empty_threshold=config.EVAL_EMPTYSHRESHOLD)

                            prob_file.write('{},{},{}\n'.format(id, mask2rle(predict, config.IMG_SIZE, config.IMG_SIZE), empty[0]))

                        del ids, image, labels, image_0, labels_0, empty, flip
                        del empty_logits, _idkwhatthisis_, logits_predict
                        del prob_predict, prob_empty, confidence
                        if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()

                    np.save(config.DIRECTORY_CHECKPOINT + "prediction/{}-CP{}_id_total.npy".format(str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-"), config.epoch), id_total)
                    np.save(config.DIRECTORY_CHECKPOINT + "prediction/{}-CP{}_predict_total.npy".format(str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-"), config.epoch), predict_total)
                    np.save(config.DIRECTORY_CHECKPOINT + "prediction/{}-CP{}_prob_empty_total.npy".format(str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-"), config.epoch), prob_empty_total)

                print("Prob_path: {}".format(prob_path))
