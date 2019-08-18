from tqdm import tqdm

import numpy as np

# criteria must be (label, pred)
import config
import tensorboardwriter
from project.siim_project.siim_util import post_process


def calculate_shakeup(label, pred, criteria, shakeup_ratio, **kwargs):
    shakeup = dict()
    pbar = tqdm(range(shakeup_ratio))
    for i in pbar:
        public_lb = set(np.random.choice(range(len(pred)), int(len(pred) * 0.5), replace=False))
        private_lb = set(range(len(pred))) - public_lb
        public_lb = np.array(list(public_lb)).astype(dtype=np.int)
        # public_lb = metrics.roc_auc_score(label[public_lb], pred[public_lb])
        public_lb = criteria(label[public_lb], pred[public_lb], **kwargs)
        private_lb = np.array(list(private_lb)).astype(dtype=np.int)
        # private_lb = metrics.roc_auc_score(label[private_lb], pred[private_lb])
        private_lb = criteria(label[private_lb], pred[private_lb], **kwargs)
        score_diff = private_lb - public_lb
        shakeup[score_diff] = (public_lb, private_lb)
        pbar.set_description_str("""Public LB: {}, Private LB: {}, Difference: {}""".format(public_lb, private_lb, score_diff))
    shakeup_keys = sorted(shakeup)
    shakeup_mean, shakeup_std = np.mean(shakeup_keys), np.std(shakeup_keys)
    return shakeup, shakeup_keys, shakeup_mean, shakeup_std

def calculate_threshold(label, pred, criteria, threshold_check_list, writer, fold, n_class=1, test_empty=None, empty_threshold=None, **kwargs):
    best_threshold = 0.0
    best_val = 0.0
    bad_value = 0
    total_score = 0
    total_tried = 0

    best_threshold_dict = np.zeros(n_class)
    best_val_dict = np.zeros(n_class)

    pbar = tqdm(threshold_check_list)
    for threshold in pbar:

        if 1: # post-process
            config.log.write("Calculating Threshold Using post_process; min_pixel={};".format(int(config.PREDICTION_CHOSEN_MINPIXEL* label.shape[-1]/1024)), once=1)
            thresholded_pred = np.zeros((pred.shape[0], 1024, 1024))
            for i, p in enumerate(pred):
                p, _ = post_process(p, threshold, int(config.PREDICTION_CHOSEN_MINPIXEL* label.shape[-1]/1024), empty=test_empty[i] if test_empty is not None else None, empty_threshold=empty_threshold)
                thresholded_pred[i] = p
            thresholded_pred = np.asarray(thresholded_pred, dtype=np.int8)
        else: # no post-process
            config.log.write("Calculating Threshold Without post_process", once=1)
            thresholded_pred = (pred > threshold).astype(np.byte)

        print("===========", label.shape, thresholded_pred.shape)

        total_tried = total_tried + 1
        score = criteria(label, thresholded_pred, **kwargs)
        total_score = total_score + score
        tensorboardwriter.write_threshold(writer, -1, score, threshold * 1000.0, fold)
        if score > best_val:
            best_threshold = threshold
            best_val = score
            bad_value = 0
        else:
            bad_value = bad_value + 1
        pbar.set_description("Threshold: {}; Score: {}; AreaUnder: {}".format(threshold, score, total_score / total_tried))
        if bad_value > 100: break

        if n_class > 1: # do per-class threshold
            for c in range(n_class):
                score = criteria(label, thresholded_pred, **kwargs)
                # tensorboardwriter.write_threshold(self.writer, c, score, threshold * 1000.0, config.fold)
                if score > best_val_dict[c]:
                    best_threshold_dict[c] = threshold
                    best_val_dict[c] = score
    # removed calculation threshold vs. frequency
    return best_threshold, best_val, total_score, total_tried