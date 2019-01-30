import os
import cv2
import numpy as np
from tqdm import tqdm

from os import listdir
from os.path import isfile, join

import config


class QUBOPreprocess:
    def __init__(self, expected_img_size=(480, 640, 3)): # h, w, c
        self.expected_img_size = expected_img_size
        if not os.path.exists(config.DIRECTORY_PREPROCESSED_IMG):
            os.makedirs(config.DIRECTORY_PREPROCESSED_IMG)
        # files = [config.DIRECTORY_SELECTED_IMG+f for f in listdir(config.DIRECTORY_SELECTED_IMG) if isfile(join(config.DIRECTORY_SELECTED_IMG, f))]
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
        # with open(config.DIRECTORY_CSV, 'a') as csv:
        #     csv.write('Id,Target,Max\n')
        with open(config.DIRECTORY_CSV, 'a') as csv:
            csv.write('Id,Target,Num\n')
        for video in pbar:
            npy_dirs = self.to_image(video)

            if "/bins/" in video:
                classes = "1"
            elif "/buoy/" in video:
                classes = "2"
            elif "/empty/" in video:
                classes = "3"
            elif "/gate/" in video:
                classes = "0"
            elif "/torpedo/" in video:
                classes = "4"
            else:
                print("Unexpected path: {}".format(video))
                continue
            with open(config.DIRECTORY_CSV, 'a') as csv:
                # for npy_dir in npy_dirs:
                #     csv.write('{},{},{}\n'.format(npy_dir, classes, len(npy_dirs)))
                csv.write('{},{},{}\n'.format(npy_dirs[0].replace("_0.npy", "_{}.npy"), classes, len(npy_dirs)))

    def to_image(self, dir):
        subdir = dir.split("/")[-2]+"/"
        name = dir.split("/")[-1]
        print("Reading {}".format(dir))
        vidcap = cv2.VideoCapture(dir)
        success, image = vidcap.read()
        count = 0

        dirs = []

        while success:
            cv2.imwrite("{}_{}.png".format(config.DIRECTORY_PREPROCESSED_IMG+subdir+name, count), image)
            image = image.astype(np.uint8)
            if self.expected_img_size != image.shape: raise ValueError("Expected image size:{} is not equal to actual image size:{}".format(self.expected_img_size, image.shape))
            npy_dir = "{}_{}.npy".format(config.DIRECTORY_PREPROCESSED_IMG+subdir+name, count)

            if not os.path.exists(config.DIRECTORY_PREPROCESSED_IMG + subdir):
                os.makedirs(config.DIRECTORY_PREPROCESSED_IMG + subdir)

            np.save(npy_dir, image)

            success, image = vidcap.read()
            print('Reading: {}, Saved: {}, Success: {}'.format(dir, "{}_{}.npy".format(config.DIRECTORY_PREPROCESSED_IMG+subdir+name, count), success))
            count += 1

            dirs.append(npy_dir)
        print("Finished. Count = {}".format(count))
        return dirs
