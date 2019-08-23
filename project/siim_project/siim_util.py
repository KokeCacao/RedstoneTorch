import numpy as np

# Koke_Cacao: for testing in submission
from tqdm import tqdm
import cv2

def compute_kaggle_lb(test_id, test_truth, test_probability, threshold, min_size, test_empty=None, empty_threshold=None):

    test_num    = len(test_truth)

    kaggle_pos = []
    kaggle_neg = []
    pbar = tqdm(range(test_num), leave=False)
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
        pbar.set_description_str('%3d  %-56s  %s   %0.5f  %0.5f'% (b, test_id[b], predict.shape, probability.mean(), probability.max()))

        if truth.sum()==0:
            kaggle_neg.append(score)
        else:
            kaggle_pos.append(score)
    kaggle_sum = kaggle_pos + kaggle_neg
    kaggle_neg = np.array(kaggle_neg)
    kaggle_pos = np.array(kaggle_pos)
    kaggle_neg_score = kaggle_neg.mean()
    kaggle_pos_score = kaggle_pos.mean()
    kaggle_score = 0.7886*kaggle_neg_score + (1-0.7886)*kaggle_pos_score # balancing kaggle
    # kaggle_score = np.array(kaggle_sum).mean() # validation used

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
# input must be (x, x)
def post_process(probability, threshold, min_size, empty=None, empty_threshold=None):
    if probability.shape[0] != probability.shape[1]: raise ValueError("{} != {}".format(probability.shape[0], probability.shape[1]))
    predict = np.zeros((probability.shape[0],probability.shape[1]), np.float32)
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

# search for a new threshold, should be smaller than 5000
def classification_based_post_process(probability, threshold, min_size, empty=None, empty_threshold=None):
    if probability.shape[0] != probability.shape[1]: raise ValueError("{} != {}".format(probability.shape[0], probability.shape[1]))
    predict = np.zeros((probability.shape[0],probability.shape[1]), np.float32)
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

    # trust the classifier to assign a mask?
    # adjust threshold until the pixel difference goes to the smallest (to make sure this is the most certain prediction by the model)
    if predict.sum() == 0:
        # choose another threshold
        pass

    return predict, num