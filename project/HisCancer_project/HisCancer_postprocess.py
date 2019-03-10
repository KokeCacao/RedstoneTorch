import os

import pandas as pd
from tqdm import tqdm

import config
from dataset.HisCancer_dataset import HisCancerDataset

class HisCancerPostprocess:
    def __init__(self, writer):
        self.writer = writer
        self.thresholds = config.PREDICTION_CHOSEN_THRESHOLD
        self.test_dataset = HisCancerDataset(config.DIRECTORY_CSV, config.DIRECTORY_SAMPLE_CSV, load_strategy="predict", writer=self.writer, column='Label')
        self.run()

    def run(self):
            for threshold in self.thresholds:
                pred_path = "postprocess.csv".format(config.DIRECTORY_LOAD, config.PREDICTION_TAG, "#", threshold)
                if os.path.exists(pred_path):
                    os.remove(pred_path)
                    print("WARNING: delete file '{}'".format(pred_path))

                count = 0
                with open(pred_path, 'a') as pred_file:
                    pred_file.write('Id,Label\n')

                    pbar = tqdm(self.test_dataset.id)
                    for id in pbar:
                        image = self.test_dataset.get_load_image_by_id(id)
                        image = cv2.fromarray(image)
                        image = image[32:64, 32:64]
                        average = image.mean(axis=0).mean(axis=0)

                        write = 0 if average[0]> 229.64312066 and average[1]>219.28200955 and average[2]>225.03200955 else 1
                        count = count + write
                        pbar.set_description_str("Id:{} Write:{} Average:{}")
                        pred_file.write('{},{}\n'.format(id, write))

                print("""
                Job Finished with {} file post-processed
                """.format(count))

                """ORGANIZE"""
                def sort(dir_sample, dir_save):
                    f1 = pd.read_csv(dir_sample)
                    f1.drop('Label', axis=1, inplace=True)
                    f2 = pd.read_csv(dir_save)
                    f1 = f1.merge(f2, left_on='Id', right_on='Id', how='outer')
                    os.remove(dir_save)
                    f1.to_csv(dir_save, index=False)
                sort(config.DIRECTORY_SAMPLE_CSV, pred_path)
                print("Pred_path: {}".format(pred_path))