import os

import numpy as np
from tqdm import tqdm

import config
from dataset.hpa_dataset import HPAData


class HPAPreprocess:
    def __init__(self, calculate=False, expected_img_size=(4, 512, 512)):
        self.expected_img_size = expected_img_size
        self.calculate = calculate  # 6item/s when turn off calculation, 6item/s when turn on, 85item/s when loaded in memory (80 save + 85to_np = 6 load)
        if not os.path.exists(config.DIRECTORY_PREPROCESSED_IMG):
            os.makedirs(config.DIRECTORY_PREPROCESSED_IMG)
        mean, std, std1 = self.run(HPAData(config.DIRECTORY_CSV, load_img_dir=config.DIRECTORY_IMG, img_suffix=".png", load_strategy="train", load_preprocessed_dir=False))
        print("""
        Train Data:
            Mean = {}
            STD  = {}
            STD1 = {}
        """.format(mean, std, std1))
        mean, std, std1 = self.run(HPAData(config.DIRECTORY_CSV, load_img_dir=config.DIRECTORY_TEST, img_suffix=".png", load_strategy="test", load_preprocessed_dir=False))
        print("""
        Test Data:
            Mean = {}
            STD  = {}
            STD1 = {}
        """.format(mean, std, std1))

        """ https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/69462
        Hi lafoss, 
        just out of interest: How did you calculate these values? I am asking because I did the same a couple of days ago, on the original 512x512 images and got slightly different results, i.e.:
        Means for train image data (originals)

        Red average: 0.080441904331346
        Green average: 0.05262986230955176
        Blue average: 0.05474700710311806
        Yellow average: 0.08270895676048498

        Means for test image data (originals)

        Red average: 0.05908022413399168
        Green average: 0.04532851916280794
        Blue average: 0.040652325092460015
        Yellow average: 0.05923425759572161

        Did you resize the images before checking the means? 
        As I say, just out of interest, 
        cheers and thanks, 
        Wolfgang
        """

    def get_mean(self, dataset, save=False, overwrite=False):
        pbar = tqdm(dataset.id)
        length = len(pbar)
        sum = [0, 0, 0, 0]
        for id in pbar:
            img = dataset.get_load_image_by_id(id).astype(np.uint8)
            if self.expected_img_size != img.size: raise ValueError("Expected image size:{} is not equal to actual image size:{}".format(self.expected_img_size, img.size))
            img_mean = np.stack((img.astype(np.float32).mean(0).mean(0)) / 255.)
            sum = sum + img_mean
            pbar.set_description("{} Sum:[{:.2f},{:.2f},{:.2f},{:.2f}]".format(id, img_mean[0], img_mean[1], img_mean[2], img_mean[3]))
            if not os.path.exists(config.DIRECTORY_PREPROCESSED_IMG + id + ".npy") and save and overwrite:
                np.save(config.DIRECTORY_PREPROCESSED_IMG + id + ".npy", img)
            elif save and not overwrite:
                pbar.set_description("Pass: {}".format(id))
                continue
        mean = sum / length
        print("     Mean = {}".format(mean))
        return mean

    def get_std(self, dataset, mean):
        pbar = tqdm(dataset.id)
        length = len(pbar)
        sum_variance = [0, 0, 0, 0]
        for id in pbar:
            img = dataset.get_load_image_by_id(id).astype(np.uint8)
            if self.expected_img_size != img.size: raise ValueError("Expected image size:{} is not equal to actual image size:{}".format(self.expected_img_size, img.size))
            img_mean = np.stack((img.astype(np.float32).mean(0).mean(0)) / 255.)
            img_variance = (img_mean - mean) ** 2
            sum_variance = sum_variance + img_variance

            pbar.set_description("{} Var:[{:.2f},{:.2f},{:.2f},{:.2f}]".format(id, img_variance[0], img_variance[1], img_variance[2], img_variance[3]))
        std = (sum_variance / length) ** 0.5
        std1 = (sum_variance / (length - 1)) ** 0.5
        print("     STD  = {}".format(std))
        print("     STD1 = {}".format(std1))
        return mean, std, std1

    def normalize(self, dataset, mean, std, save=True, overwrite=False):
        """normalize and save data
        Not recomanded because uint8 can be load faster than float32
        """
        pbar = tqdm(dataset.id)
        length = len(pbar)
        for id in pbar:
            img = (dataset.get_load_image_by_id(id).astype(np.float32) / 225. - mean) / std
            if self.expected_img_size != img.size: raise ValueError("Expected image size:{} is not equal to actual image size:{}".format(self.expected_img_size, img.size))
            pbar.set_description("{}".format(id))
            if not os.path.exists(config.DIRECTORY_PREPROCESSED_IMG + id + ".npy") and save and overwrite:
                np.save(config.DIRECTORY_PREPROCESSED_IMG + id + ".npy", img)
            elif save and not overwrite:
                pbar.set_description("Pass: {}".format(id))
                continue

    def run(self, dataset):
        pbar = tqdm(dataset.id)
        length = len(pbar)
        sum = [0, 0, 0, 0]
        sum_variance = [0, 0, 0, 0]
        mean = [0, 0, 0, 0]
        for id in pbar:
            if os.path.exists(config.DIRECTORY_PREPROCESSED_IMG + id + ".npy") and not self.calculate:
                pbar.set_description("Pass: {}".format(id))
                continue
            img = dataset.get_load_image_by_id(id).astype(np.uint8)
            if self.expected_img_size != img.size: raise ValueError("Expected image size:{} is not equal to actual image size:{}".format(self.expected_img_size, img.size))
            # print(img.shape) # (512, 512, 4)
            if self.calculate:
                img_mean = np.stack((img.astype(np.float32).mean(0).mean(0)) / 255.)
                sum = sum + img_mean
                pbar.set_description("{} Sum:[{:.2f},{:.2f},{:.2f},{:.2f}]".format(id, img_mean[0], img_mean[1], img_mean[2], img_mean[3]))

            if not os.path.exists(config.DIRECTORY_PREPROCESSED_IMG + id + ".npy"): np.save(config.DIRECTORY_PREPROCESSED_IMG + id + ".npy", img)

        if self.calculate:
            mean = sum / length
            print("     Mean = {}".format(mean))
        if self.calculate:
            pbar = tqdm(dataset.id)
            for id in pbar:
                img = dataset.get_load_image_by_id(id).astype(np.uint8)
                if self.expected_img_size != img.size: raise ValueError("Expected image size:{} is not equal to actual image size:{}".format(self.expected_img_size, img.size))
                img_mean = np.stack((img.astype(np.float32).mean(0).mean(0)) / 255.)
                img_variance = (img_mean - mean) ** 2
                sum_variance = sum_variance + img_variance

                pbar.set_description("{} Var:[{:.2f},{:.2f},{:.2f},{:.2f}]".format(id, img_variance[0], img_variance[1], img_variance[2], img_variance[3]))
            std = (sum_variance / length) ** 0.5
            std1 = (sum_variance / (length - 1)) ** 0.5
            print("     STD  = {}".format(std))
            print("     STD1 = {}".format(std1))
            return mean, std, std1
        return 0, 0, 0