import os
import sys

import torch
import numpy as np
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm

from submission import config
from submission import imet_net
from submission.imet_dataset import IMetDataset, test_collate, tta_collate


class IMetPrediction:
    def __init__(self):
        self.thresholds = config.PREDICTION_CHOSEN_THRESHOLD
        self.nets = []
        for fold in range(config.MODEL_FOLD):
            if fold not in config.train_fold:
                print("     Junping Fold: #{}".format(fold))
                self.nets.append(None)
            else:
                print("     Creating Fold: #{}".format(fold))
                # net = imet_net.fbresnet50(config.TRAIN_NUM_CLASS, pretrained=False)
                # net = imet_net.se_resnext50_32x4d(config.TRAIN_NUM_CLASS, pretrained=None)
                net = imet_net.se_resnext101_32x4d(config.TRAIN_NUM_CLASS, pretrained=None)

                if config.TRAIN_GPU_ARG: net = torch.nn.DataParallel(net, device_ids=config.TRAIN_GPU_LIST)
                self.nets.append(net)

        config.load_optimizers = False
        config.load_lr_schedulers = False
        self.load_checkpoint_all_fold(self.nets, None, None, config.DIRECTORY_LOAD)

        self.test_dataset = IMetDataset(config.DIRECTORY_CSV, config.DIRECTORY_SAMPLE_CSV, writer=None, id_col = 'id', target_col='attribute_ids')

        self.run()

    def load_checkpoint_all_fold(self, nets, optimizers, lr_schedulers, load_path):
        if not load_path or load_path == "False":
            config.epoch = 0
            config.global_steps = np.zeros(len(nets))
            print("=> Nothing loaded because no specify loadfile")
            return
        if not load_path or not os.path.isfile(load_path):
            load_path = os.path.splitext(load_path)[0] + "-MILESTONE" + os.path.splitext(load_path)[1]
        if load_path and os.path.isfile(load_path):
            print("=> Loading checkpoint '{}'".format(load_path))

            def load_file(file):
                if sys.version_info[0] < 3:
                    return torch.load(file)
                else:
                    from functools import partial
                    import pickle
                    pickle.load = partial(pickle.load, encoding="latin1")
                    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
                    return torch.load(file, map_location=lambda storage, loc: storage, pickle_module=pickle)

            checkpoint = load_file(load_path)
            if 'state_dicts' not in checkpoint:
                raise ValueError("=> Checkpoint is broken, nothing loaded")
            config.epoch = checkpoint['epoch']
            config.global_steps = checkpoint['global_steps']

            optimizers = [None] * len(nets) if optimizers is None else optimizers
            lr_schedulers = [None] * len(nets) if lr_schedulers is None else lr_schedulers
            for fold, (net, optimizer, lr_scheduler) in enumerate(zip(nets, optimizers, lr_schedulers)):
                if fold not in config.train_fold:
                    continue

                if config.load_state_dicts:
                    if 'state_dicts' in checkpoint.keys():
                        if fold >= len(checkpoint['state_dicts']):
                            net.load_state_dict(checkpoint['state_dicts'][0])
                            print("[WARNING] No state_dict for the fold found, loading checkpoint['state_dicts'][0]")
                        else:
                            if checkpoint['state_dicts'][fold] is None: print("[ERROR] The fold number of your input is not correct or no fold found")
                            net.load_state_dict(checkpoint['state_dicts'][fold])
                    else:
                        print("[WARNING] No keys [state_dicts] detected from loading")
                else:
                    print("[MESSAGE] No state_dicts loaded because of your settings")

                if config.load_optimizers:
                    if 'optimizers' in checkpoint.keys():
                        if fold >= len(checkpoint['optimizers']):
                            optimizer.load_state_dict(checkpoint['optimizers'][0])  # BAD CODE
                            print("[WARNING] No optimizer for the fold found, loading checkpoint['optimizers'][0]")
                        else:
                            if checkpoint['optimizers'][fold] is None: print("[ERROR] The fold number of your input is not correct or no fold found")
                            optimizer.load_state_dict(checkpoint['optimizers'][fold])  # BAD CODE
                    else:
                        print("[WARNING] No keys [optimizers] detected from loading")
                else:
                    print("[MESSAGE] No optimizers loaded because of your settings")

                if config.load_lr_schedulers:
                    if 'lr_schedulers' in checkpoint.keys():
                        if fold >= len(checkpoint['lr_schedulers']):
                            lr_scheduler.load_state_dict(checkpoint['lr_schedulers'][0])  # BAD CODE
                            print("[WARNING] No lr_schedulers for the fold found, loading checkpoint['lr_schedulers'][0]")
                        else:
                            if checkpoint['lr_schedulers'][fold] is None: print("[ERROR] The fold number of your input is not correct or no fold found")
                            lr_scheduler.load_state_dict(checkpoint['lr_schedulers'][fold])  # BAD CODE
                    else:
                        print("[WARNING] No keys [lr_schedulers] detected from loading")
                else:
                    print("[MESSAGE] No lr_schedulers loaded because of your settings")

                # move_optimizer_to_cuda(optimizer)
                if fold < len(config.global_steps):
                    print("=> Loading checkpoint {} epoch; {} step".format(config.epoch, config.global_steps[fold]))
                else:
                    print("=> Loading checkpoint {} epoch; {} step".format(config.epoch, config.global_steps[0]))
            print("=> Loaded checkpoint {} epoch; {}-{} step".format(config.epoch, config.global_steps[0], config.global_steps[-1]))
        else:
            raise ValueError("=> Nothing loaded because of invalid directory: {}".format(load_path))

    def run(self):
        torch.no_grad()
        """Used for Kaggle submission: predicts and encode all test images"""
        for fold, net in enumerate(self.nets):
            if net == None:
                continue
            for threshold in self.thresholds:

                if config.PREDICTION_TTA == 0:
                    prob_path = "{}{}-F{}-Prob.csv".format("/kaggle/working/", config.PREDICTION_TAG, fold)
                    if os.path.exists(prob_path):
                        os.remove(prob_path)
                        print("WARNING: delete file '{}'".format(prob_path))

                    with open(prob_path, 'a') as prob_file:
                        prob_file.write('Id,'+','.join([str(x) for x in range(config.TRAIN_NUM_CLASS)])+'\n')

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
                            predicts = torch.nn.Sigmoid()(predicts).detach().cpu().numpy()

                            confidence = (np.absolute(predicts - 0.5).mean() + 0.5).item()
                            total_confidence = total_confidence + confidence
                            confidence_list.append(confidence)
                            pbar.set_description("Thres:{} Id:{} Confidence:{}/{}".format(threshold, ids[0].replace("../input/imet-2019-fgvc6/test/", "").replace(".png", ""), confidence, total_confidence / (batch_index + 1)))

                            for id, predict in zip(ids, predicts):
                                prob_file.write('{},{}\n'.format(id.replace("../input/imet-2019-fgvc6/test/", "").replace(".png", ""), ','.join(predict.astype(np.str).tolist())))

                            del ids, image, labels_0, image_for_display, predicts
                            if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()

                    # left, right = st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))
                    print("""
                    Mean Confidence = {}, STD = {}
                    """.format(np.mean(confidence_list), np.std(confidence_list)))
                    print("Prob_path: {}".format(prob_path))
                else:
                    """TTA"""
                    for tta in range(config.PREDICTION_TTA):
                        prob_path = "{}{}-F{}-TTA{}.csv".format("/kaggle/working/", config.PREDICTION_TAG, fold, tta)
                        if os.path.exists(prob_path):
                            os.remove(prob_path)
                            print("WARNING: delete file '{}'".format(prob_path))

                        with open(prob_path, 'a') as prob_file:
                            prob_file.write('Id,'+','.join([str(x) for x in range(config.TRAIN_NUM_CLASS)])+'\n')

                            test_loader = data.DataLoader(self.test_dataset,
                                                          batch_size=config.MODEL_BATCH_SIZE,
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
                            pbar = tqdm(test_loader)
                            total_confidence = 0
                            confidence_list = []

                            print("Set Model Trainning mode to trainning=[{}]".format(net.eval().training))

                            for batch_index, (ids, image, labels_0, image_for_display) in enumerate(pbar):

                                if config.TRAIN_GPU_ARG: image = image.cuda()
                                predicts = net(image)
                                predicts = torch.nn.Sigmoid()(predicts).detach().cpu().numpy()

                                confidence = (np.absolute(predicts - 0.5).mean() + 0.5).item()
                                total_confidence = total_confidence + confidence
                                confidence_list.append(confidence)
                                pbar.set_description("Thres:{} Id:{} Confidence:{}/{}".format(threshold, ids[0].replace("../input/imet-2019-fgvc6/test/", "").replace(".png", ""), confidence, total_confidence / (batch_index + 1)))

                                for id, predict in zip(ids, predicts):
                                    prob_file.write('{},{}\n'.format(id.replace("../input/imet-2019-fgvc6/test/", "").replace(".png", ""), ','.join(predict.astype(np.str).tolist())))

                                del ids, image, labels_0, image_for_display, predicts
                                if config.TRAIN_GPU_ARG: torch.cuda.empty_cache()

                        # left, right = st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))
                        print("""
                        Mean Confidence = {}, STD = {}
                        """.format(np.mean(confidence_list), np.std(confidence_list)))
                        print("Prob_path: {}".format(prob_path))

