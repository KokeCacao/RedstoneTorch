import os
import cv2
import numpy as np
from tqdm import tqdm

class IMetPreprocess:
    def __init__(self, from_dir, to_dir): # h, w, c

        # make directory for preprocessed image
        if not os.path.exists(to_dir):
            os.makedirs(to_dir)

        """Preprocess train.csv by hand"""

        # searching files that need to be processed
        files = []
        for path, subdirs, f in os.walk(from_dir):
            print("Searching in {}, {}, {}".format(path, subdirs, files))
            for name in f:
                files.append(os.path.join(path, name))
                print("Get Name: {}".format(name))

        print("files in selected {} in path {}".format(files, from_dir))

        pbar = tqdm(files)
        for dir in pbar:
            pth=os.path.join(to_dir, dir.split("/")[-1].split(".")[0]+".npy")
            pbar.set_description_str("Saving .npy images in {}".format(pth))
            np.save(pth, cv2.imread(dir, cv2.IMREAD_COLOR).astype(np.uint8))