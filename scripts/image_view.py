from PIL import Image
import subprocess
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer


if __name__ == '__main__':
    dataframe = pd.read_csv('augment.csv', engine='python').set_index('Id')
    multilabel_binarizer = MultiLabelBinarizer().fit([list(range(28))])
    labelframe = multilabel_binarizer.transform([(int(i) for i in s.split()) for s in dataframe['Target']])

    path = "answer.csv"

    if os.path.exists(path):
        os.remove(path)
        print("WARNING: delete file '{}'".format(path))

    with open(path, 'a') as file:
        file.write('Id,Predicted\n')
    for i, labelframe in enumerate(labelframe):
        p = subprocess.Popen(["eog", "images/{}_rgb.jpg".format(i)])
        answer = input("Answer:").replace(", ", " ").replace(",", " ")
        print("Your Answer for #{}: {}".format(i, answer))
        with open(path, 'a') as file:
            file.write('{},{}\n'.format(i, answer))
        p.kill()
