import numpy as np

# Koke_Cacao: for testing in submission
from tqdm import tqdm
import cv2

def compute_kaggle_lb(test_id, test_truth, test_probability, threshold, min_size, tq=True, test_empty=None, empty_threshold=None):

    test_num    = len(test_truth)

    kaggle_pos = []
    kaggle_neg = []
    pbar = tqdm(range(test_num)) if tq else range(test_num)
    for b in pbar:
        truth       = test_truth[b,0]
        probability = test_probability[b,0]
        empty       = test_empty[b] if test_empty is not None else None

        if truth.shape!=(1024,1024):
            truth = cv2.resize(truth, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
            truth = (truth>0.5).astype(np.float32)

        if probability.shape!=(1024,1024):
            probability = cv2.resize(probability, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)

        #-----
        predict, num_component = post_process(probability, threshold, min_size, empty=empty, empty_threshold=empty_threshold)

        score = kaggle_metric_one(predict, truth)
        if tq: pbar.set_description_str('%3d  %-56s  %s   %0.5f  %0.5f'% (b, test_id[b], predict.shape, probability.mean(), probability.max()))

        if truth.sum()==0:
            kaggle_neg.append(score)
        else:
            kaggle_pos.append(score)

    kaggle_neg = np.array(kaggle_neg)
    kaggle_pos = np.array(kaggle_pos)
    kaggle_neg_score = kaggle_neg.mean()
    kaggle_pos_score = kaggle_pos.mean()
    kaggle_score = 0.7886*kaggle_neg_score + (1-0.7886)*kaggle_pos_score

    return kaggle_score, kaggle_neg_score, kaggle_pos_score

def kaggle_metric_one(predict, truth):

    if truth.sum() ==0:
        if predict.sum() ==0: return 1
        else:                 return 0

    #----
    predict = predict.reshape(-1)
    truth   = truth.reshape(-1)

    intersect = predict*truth
    union     = predict+truth
    dice      = 2.0*intersect.sum()/union.sum()
    return dice

# Koke_Cacao: deprecated, only for Kaggle LB prediction
def post_process(probability, threshold, min_size, empty=None, empty_threshold=None):

    predict = np.zeros((1024,1024), np.float32)
    num = 0
    if empty is not None and empty_threshold is not None and empty > empty_threshold:
        return predict, num

    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]

    num_component, component = cv2.connectedComponents(mask.astype(np.uint8)) # Koke_Cacao: return n-white in a region and those white pixels in an array

    for c in range(1,num_component):
        p = (component==c)
        if p.sum()>min_size:
            predict[p] = 1
            num += 1
    return predict, num