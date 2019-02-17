import os
import cv2
import numpy as np
from tqdm import tqdm

import config


class HisCancerPreprocess:
    def __init__(self, expected_img_size=(480, 640, 3)): # h, w, c
        self.expected_img_size = expected_img_size

        # make directory for preprocessed image
        if not os.path.exists(config.DIRECTORY_PREPROCESSED_IMG):
            os.makedirs(config.DIRECTORY_PREPROCESSED_IMG)

        """Preprocess train.csv by hand"""

        # searching files that need to be processed
        files = []
        for path, subdirs, f in os.walk(config.DIRECTORY_SELECTED_IMG):
            print("Searching in {}, {}, {}".format(path, subdirs, files))
            for name in f:
                files.append(os.path.join(path, name))
                print("Get Name: {}".format(name))

        print("files in selected {} in path {}".format(files, config.DIRECTORY_SELECTED_IMG))
        self.run(files)


    def run(self, videos):
        pbar = tqdm(videos)
        for dir in pbar:
            pth=os.path.join(config.DIRECTORY_PREPROCESSED_IMG, dir.split("/")[-1].split(".")[0]+".npy")
            pbar.set_description_str("Saving .npy images in {}".format(pth))
            np.save(pth, cv2.imread(dir, cv2.IMREAD_COLOR).astype(np.uint8))
