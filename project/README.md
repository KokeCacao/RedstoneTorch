Psudolabeling:
* https://www.kaggle.com/youhanlee/unet-resnetblock-hypercolumn-deep-supervision-fold
Based on dicussions and kernels, I made function to make a Unet+ResNetBlock+Hypercolumn+Deep supervision.
Thanks for sharing to all of authors!
https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/68435
https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/68190
For scSE, I could't find the good position where scSE is added. You can add scSE in more good position!OC Net
https://zhangbin0917.github.io/2018/09/13/OCNet-Object-Context-Network-for-Scene-Parsing/
CV: random, stratified, group repeated
https://stats.stackexchange.com/questions/273749/majority-voting-in-ensemble-learning

Learning Rate Scheduler
https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers.py
https://github.com/pytorch/pytorch/issues/3790
https://discuss.pytorch.org/t/current-learning-rate-and-cosine-annealing/8952Fold
https://github.com/dnouri/skorch
https://stats.stackexchange.com/questions/14474/compendium-of-cross-validation-techniques

Classifier
https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65933
https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/68435Libs
https://github.com/fastai/fastai
https://github.com/albu/albumentationsjigsaw puzzle and postprocessing
https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/68993
https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69168

Structure
Dual Path Networks
Concurrent Spatial and Channel Squeeze & Excitation: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178

Psodu
https://www.kaggle.com/c/challenges-in-representation-learning-the-black-box-learning-challenge/discussion/4706

Top Players
1: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69291

5: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69051
Don't trust CV, trust stability; classification+segmentation; scSE+Unet+hypercolumn; lovasz+cosine annealing+snapshots

8: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69220#407805

9: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69053; https://github.com/radekosmulski/tgs_salt_solution
https://github.com/tugstugi/pytorch-saltnet
AdamW with a Noam scheduler
SWA after the training on the best loss, pixel accuracy, metric and the last models.
modifying Lovasz to symmetric which gives a good boost in the LB:

11: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69093
https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69165

25: BEST MODEL https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69179#40771134: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69160; https://github.com/mingmingDiii/kaggle_tgs

