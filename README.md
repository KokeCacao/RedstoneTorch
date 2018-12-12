# Redstone Torch
![RedstoneTorch](https://d1u5p3l4wpay3k.cloudfront.net/minecraft_gamepedia/d/da/Redstone_Torch.png)


## Models
```text
python train.py --projecttag base --versiontag base1 --resume False
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-02-20-23-22-547213-test/ --port=6006
//memory leak 8827 at step 1.3k from 4788
//10 fold 10 train
//BAD MODEL(droped)
=
python train.py --projecttag mem --versiontag mem1 --resume False
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-02-20-23-22-547213-test/ --port=6006
//memory leak from 8285 at step 1.4k, 6577
//10 fold 1 train
//3*4 epoch, loss=0.5556
=
python train.py --projecttag mem --versiontag mem1 --resume False
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-03-01-10-04-699788-mem/ --port=6006
//10 fold 10 train, put evaluation inside with each fold instead of epoch
//still memory leak between folds, but leak back when epoch
=
python train.py --projecttag mem2 --versiontag mem2 --resume False
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-03-02-21-55-372744-mem2/ --port=6006
python predict.py --projecttag 2018-11-03-02-21-55-372744-mem2 --versiontag mem2-pred --loadfile mem2-CP2.pth
//10 fold 2 train, put evaluation back, but save model using self.net and self.optimizers
//memory leak around 7G min stable, 2nd*4 epoch
=
python train.py --projecttag mem3 --versiontag mem3 --resume False
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-03-14-14-20-075060-mem3/ --port=6006  
//No memory leak at epoch 3*4 after delete all extraneous things. memory around 2.5G
//CPU memory leak, GPU fine
=
python train.py --projecttag mem4 --versiontag mem4 --resume False
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-03-19-02-40-014830-mem4/ --port=6006
python predict.py --projecttag 2018-11-03-19-02-40-014830-mem4 --versiontag mem4 --loadfile mem4-CP9.pth
//open extraneous things, clean-up loss.detach(), clean cache() outside of the epoch(), del more things
//add f1
//no GPU leak during training, but increasing GPU usage after eval
//Epoch: 10*4, Fold: 0 TrainLoss: 0.47 ValidLoss: 0.469516485929, ValidF1: 0.179454994182
//To Resume: python train.py --versiontag 'test' --projecttag 2018-11-04-04-19-26-236033-lr3--loadfile lr3-CP7.pth
//This model is good but it take 15h to get to focal loss 0.5. I guess that is was too small the lr
=
python train.py --projecttag mem5 --versiontag mem5 --resume False (on machine 2)
//add CPU memory monitor and evil monitor
=
python train.py --projecttag lr1 --versiontag lr1 --resume False (on machine 2)
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-04-03-22-36-637908-lr1/ --port=6006
//okay, but not significant lambda x: x/(100*np.mod(-x-1, 600))-0.000006*x
=
python train.py --projecttag lr2 --versiontag lr2 --resume False (on machine 2)
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-04-03-55-40-334831-lr2/ --port=6006
//lambda x: x/(8*np.mod(-x-1, 600)+0.1)-0.000207*x
//ln=2 is fine actually
=
python train.py --projecttag lr3 --versiontag lr3 --resume False (on machine 2)
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-04-04-19-26-236033-lr3/ --port=6006
//adjust batch to 32, start from lr=5
//Epoch: 8*4, Fold: 0 TrainLoss: 0.468069558797 ValidLoss: 0.453331559896, ValidF1: 0.190886673186
=
python train.py --projecttag normal1 --versiontag normal1 --resume False
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-05-22-05-03-738974-normal1/ --port=6006
python train.py --projecttag 2018-11-05-22-05-03-738974-normal1 --versiontag normal2 --resume True --loadfile normal1-CP1.pth
    python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-05-22-05-03-738974-normal1/ --port=6006
//normalize data, use both loss(f1, focal), lr=0.1
//lambda global_step: (0.1/2)*(np.cos(np.pi*(np.mod(global_step-1,10000)/(10000)))+1)
//loading speed = 2.46s/it in 32 batch (compare to 1.10s in 16 batch)
//8740k step, 0.6344BestF1, 0.43-45eval-focal, 0.20-21evalF1, 0.2083epochloss, 0.57-58trainF1, 0.44-45trainFocal
=
python train.py --projecttag gpu1 --versiontag gpu1 --resume False (on machine 2)
//around 0.7s per batch of 32
=
python train.py --projecttag tune1 --versiontag tune1 --resume False
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-07-06-51-09-190794-tune1/ --port=6006
//switch to only one fold, change to Adadelta, adjust lr=2 * 46808 / 32, start lr=1.0, add weighted_bce
//F1 goes up, not good
=
python train.py --projecttag tune2 --versiontag tune2 --resume False
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-07-07-14-32-212121-tune2/ --port=6006
//log bce, remove bce, maybe at the end of trianing
=
python train.py --projecttag tune3 --versiontag tune3 --resume False
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-07-07-22-33-042252-tune3/ --port=6006
=
python train.py --projecttag tune4 --versiontag tune4 --resume False
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-07-13-02-42-175714-tune4/ --port=6006
python predict.py --projecttag 2018-11-07-13-02-42-175714-tune4 --versiontag tune4 --loadfile tune4-CP11.pth
=
python train.py --projecttag 2018-11-07-13-02-42-175714-tune4 --versiontag tune5 --resume True --loadfile tune4-CP17.pth
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-07-13-02-42-175714-tune4/ --port=6006
//switch to beta=1, +weighted_bce
//Epoch: 17, Fold: 0
            TrainLoss: 14.8867569101, TrainF1:
            ValidLoss: 0.457575827837, ValidF1: 0.210946713931
//don't use beta=2, it is evil
=
python train.py --projecttag 2018-11-07-13-02-42-175714-tune4 --versiontag tune6 --resume True --loadfile tune5-CP23.pth
//try only with f1 loss
//focal up a lot, weighted bce down a lot, bce up little, f1 down little (from start)
//nothing happened (from CP23)
=
python train.py --projecttag tune5 --versiontag tune6 --resume False --loadfile tune5-CP18.pth --loaddir 2018-11-07-13-02-42-175714-tune4
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-08-03-24-12-983709-tune5/ --port=6006
python train.py --projecttag tune5 --loaddir 2018-11-08-03-24-12-983709-tune5 --versiontag tune7 --resume True --loadfile tune6-CP2.pth
//only focal now, add precision recall graph
python train.py --projecttag tune5 --loaddir 2018-11-08-03-24-12-983709-tune5 --versiontag tune8 --resume True --loadfile tune7-CP5.pth
//now add f1 (focall loss is bad when you just init the train)
// Epoch: 22, Fold: 0                                                                            
            TrainLoss: 13.4163611972, TrainF1: 0.999997869304
            ValidLoss: 0.449503481388, ValidF1: 0.226643079329 Thres:0.1837, 0.6439F1
F1 by sklearn = 0.196469649036
python predict.py --loaddir 2018-11-08-03-24-12-983709-tune5 --versiontag f1andsomefocal --loadfile tune8-CP22.pth
RedstoneTorch/model/2018-11-08-03-24-12-983709-tune5/tune8-CP22.pth-f1andsomefocal-0.csv
=
python train.py --projecttag tune6 --versiontag one --resume False
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-08-13-18-39-289369-tune6/ --port=6006
python predict.py --loaddir 2018-11-08-13-18-39-289369-tune6 --versiontag bcef1-1 --loadfile one-CP18.pth
RedstoneTorch/model/2018-11-08-13-18-39-289369-tune6/one-CP18.pth-bcef1-1-F0-T0.01602.csv
// train on bce and f1
python train.py --projecttag tune6 --versiontag two --resume True --loadfile one-CP25.pth --loaddir 2018-11-08-13-18-39-289369-tune6
// adjust down lr by a factor of 10, adjust batch size by 2(32->64)
// focal loss seem to need a bigger batch size, I will see how the loss fluctuate to decide whether to add focal or not
python predict.py --loaddir 2018-11-08-13-18-39-289369-tune6 --versiontag bcef1-2 --loadfile two-CP44.pth
//python train.py --projecttag tune6 --versiontag three --resume True --loadfile two-CP46.pth --loaddir 2018-11-08-13-18-39-289369-tune6
//train +with focal
=
python train.py --projecttag normal3 --versiontag three --resume False
//add stratify fold, change focal gamma to 4, combination of loss on different stage
            else
                loss = f1 + bce.sum() + focal.sum()
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-11-08-00-19-978745-normal3/ --port=6006
python train.py --projecttag normal4 --versiontag one --resume False
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-13-04-54-14-153732-normal4/ --port=6006
//normalize by all data instead of just train or val
python train.py --projecttag normal4 --versiontag one --resume False --loadfile two-CP46.pth --loaddir 2018-11-08-13-18-39-289369-tune6
=
python train.py --projecttag normal4 --versiontag one --resume False
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-13-14-20-22-604828-normal4/ --port=6006
=
python train.py --projecttag normal4 --versiontag two --resume False --loadfile one-CP8.pth --loaddir 2018-11-13-14-20-22-604828-normal4
=
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-13-18-29-22-424660-normal4/ --port=6006
python predict.py --loaddir 2018-11-13-18-29-22-424660-normal4 --versiontag 0.42 --loadfile two-CP52.pth
python train.py --projecttag test --versiontag test --resume False
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-14-18-47-41-277052-test/ --port=6006

python train.py --projecttag normal4 --versiontag four --resume True --loadfile two-CP52.pth --loaddir 2018-11-13-18-29-22-424660-normal4
//fix predict path, fix prediction
python predict.py --loaddir 2018-11-13-18-29-22-424660-normal4 --versiontag 0.42 --loadfile two-CP52.pth
download: RedstoneTorch/model/2018-11-13-18-29-22-424660-normal4/two-CP52.pth-0.42-F0-T0.01.csv
download: RedstoneTorch/model/2018-11-13-18-29-22-424660-normal4/two-CP52.pth-0.42-F0-T0.5.csv
=
//fix sigmoid
python train.py --projecttag normal4 --versiontag three --resume True --loadfile two-CP52.pth --loaddir 2018-11-13-18-29-22-424660-normal4
python train.py --projecttag normal4 --versiontag four --resume True --loadfile three-CP56.pth --loaddir 2018-11-13-18-29-22-424660-normal4
//fix display, predict, add f1-gamma5, remove bce
python predict.py --loaddir 2018-11-13-18-29-22-424660-normal4 --versiontag 0.1942 --loadfile four-CP64.pth (raw threshold)
download: RedstoneTorch/model/2018-11-13-18-29-22-424660-normal4/four-CP64.pth-0.1942-F0-T0.1942.csv
python predict.py --loaddir 2018-11-13-18-29-22-424660-normal4 --versiontag 0.2187 --loadfile four-CP64.pth (smothed threshold)
download: RedstoneTorch/model/2018-11-13-18-29-22-424660-normal4/four-CP64.pth-0.2187-F0-T0.2187.csv
python predict.py --loaddir 2018-11-13-18-29-22-424660-normal4 --versiontag 0.2187 --loadfile three-CP55.pth (CP55)
download: RedstoneTorch/model/2018-11-13-18-29-22-424660-normal4/three-CP55.pth-0.2187-F0-T0.2187.csv
=
python train.py --projecttag normal5 --versiontag one --resume False --loadfile two-CP52.pth --loaddir 2018-11-13-18-29-22-424660-normal4
//change augmentation, change optimzer, output more validatuon loss, image to size 512, batch to 32
python train.py --projecttag aug --versiontag one --resume False --loadfile two-CP35.pth --loaddir 2018-11-08-13-18-39-289369-tune6
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-19-04-02-48-790862-aug/ --port=6006
//add four times TTA, adjust weighted BCE to negatively weighted, use weighted BCE, create an LB versioin submission - dropping rare class,
python train.py --projecttag test --versiontag test --resume False
python train.py --projecttag aug2 --versiontag one --resume False --loadfile two-CP35.pth --loaddir 2018-11-08-13-18-39-289369-tune6
//start training aug with 3 fold -1,2,3
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-19-18-55-35-204127-aug2/ --port=6006
=
python train.py --projecttag augnew --versiontag one --resume False
python train.py --projecttag test --versiontag one --resume False
//2 fold
=
python train.py --projecttag seresnext-augnew-2pooling --versiontag one --resume False
python train.py --projecttag seresnext-augnew-2pooling --versiontag 9ff937c --resume True --loadfile one-CP23_F[1]_PT2018-11-22-05-08-27-139778-seresnext-augnew-2pooling_VTone_LR0.1_BS64_IMG224.pth --loaddir 2018-11-22-05-08-27-139778-seresnext-augnew-2pooling

python predict.py --loaddir 2018-11-22-05-08-27-139778-seresnext-augnew-2pooling --versiontag d418e9d_THRES0.1_SK0.4862 --loadfile 9ff937c-CP27_F[1]_PTseresnext-augnew-2pooling_VT9ff937c_LR0.1_BS64_IMG224.pth
python train.py --projecttag seresnext-augnew-2pooling --versiontag 20f4aed --resume True --loadfile 9ff937c-CP27_F[1]_PTseresnext-augnew-2pooling_VT9ff937c_LR0.1_BS64_IMG224.pth --loaddir 2018-11-22-05-08-27-139778-seresnext-augnew-2pooling

RedstoneTorch/model/2018-11-22-05-08-27-139778-seresnext-augnew-2pooling/9ff937c-CP27_F[1]_PTseresnext-augnew-2pooling_VT9ff937c_LR0.1_BS64_IMG224.pth-d418e9d_THRES0.1_SK0.4862-F0-T0.1-LB.csv
RedstoneTorch/model/2018-11-22-05-08-27-139778-seresnext-augnew-2pooling/9ff937c-CP27_F[1]_PTseresnext-augnew-2pooling_VT9ff937c_LR0.1_BS64_IMG224.pth-d418e9d_THRES0.1_SK0.4862-F0-T0.1.csv

python predict.py --versiontag 4edd2fd_THRES0.268_SK0.5436 --loaddir 2018-11-22-05-08-27-139778-seresnext-augnew-2pooling --loadfile 20f4aed-CP38_F[1]_PTseresnext-augnew-2pooling_VT20f4aed_LR0.1_BS64_IMG224.pth
Download: RedstoneTorch/model/2018-11-22-05-08-27-139778-seresnext-augnew-2pooling/20f4aed-CP38_F[1]_PTseresnext-augnew-2pooling_VT20f4aed_LR0.1_BS64_IMG224.pth-4edd2fd_THRES0.268_SK0.5436-F0-T0.268.csv
Download: RedstoneTorch/model/2018-11-22-05-08-27-139778-seresnext-augnew-2pooling/20f4aed-CP38_F[1]_PTseresnext-augnew-2pooling_VT20f4aed_LR0.1_BS64_IMG224.pth-4edd2fd_THRES0.268_SK0.5436-F0-T0.1.csv
Download: RedstoneTorch/model/2018-11-22-05-08-27-139778-seresnext-augnew-2pooling/20f4aed-CP38_F[1]_PTseresnext-augnew-2pooling_VT20f4aed_LR0.1_BS64_IMG224.pth-4edd2fd_THRES0.268_SK0.5436-F0-T0.01.csv

python train.py --projecttag seresnext-augnew-2pooling --versiontag 4ebebfe --resume True --loadfile 20f4aed-CP38_F[1]_PTseresnext-augnew-2pooling_VT20f4aed_LR0.1_BS64_IMG224.pth --loaddir 2018-11-22-05-08-27-139778-seresnext-augnew-2pooling
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-22-05-08-27-139778-seresnext-augnew-2pooling/ --port=6006

python predict.py --versiontag gsfv1 --loaddir 2018-11-22-05-08-27-139778-seresnext-augnew-2pooling --loadfile 20f4aed-CP38_F[1]_PTseresnext-augnew-2pooling_VT20f4aed_LR0.1_BS64_IMG224.pth
Download: RedstoneTorch/model/20f4aed-CP38_F[1]_PTseresnext-augnew-2pooling_VT20f4aed_LR0.1_BS64_IMG224.pth-gsfv1-F0-T0.268.csv

```

## Usage
The folowing instructions are made so that you can use this library

### Data
The dataset is provided by Kaggle  
However, kaggle api is not very easy to use on remote server

Please use this chrome plugging to get `cookie.txt` file: [Here](https://chrome.google.com/webstore/detail/cookiestxt/njabckikapfpffapmjgojcnbfjonfjfg?hl=zh-CN)

After you upload your `cookie.txt` file to your remote server, use command(provided by [CarlosSouza](https://www.kaggle.com/c/cdiscount-image-classification-challenge/discussion/39492))

Please run the following command in your ~/RedstoneTorch directory
```commandline
cd ~/RedstoneTorch
wget -x -c --load-cookies cookies.txt -P data -nH --cut-dirs=5 https://www.kaggle.com/c/DATASET/train.bson
wget -x -c --load-cookies cookies.txt -P data -nH --cut-dirs=5 https://www.kaggle.com/c/DATASET/download-all
```
The `DATASET` can be replaced with `human-protein-atlas-image-classification`  
The command above will create a file named `data` and put your file `download-all` in it.  
So you need to unzip the `doanload-all`  
To do so, run the following command
```commandline
unzip ~/RedstoneTorch/data/download-all -d ~/RedstoneTorch/data
```
and then you need to unzip the `train.zip` and `test.zip`
```commandline
unzip ~/RedstoneTorch/data/train.zip -d ~/RedstoneTorch/data/train
unzip ~/RedstoneTorch/data/test.zip -d ~/RedstoneTorch/data/test
```
Please use `sudo` in front of these command if the terminal says that you don't have permissions to do so

However, you may not have the full permission to read doanloaded file, use
```commandline
sudo chmod -R a+rwx train.csv
```
to give yourself permission to read.
### Preprocess
By using this command
```commandline
python preprocess.py
```
You can preprocess the data.
 - You can calculate the mean and standard deviation of train and test data
 - The image will transformed to .npy so that it load faster

### Train
You can start trainning by type command `python train.py`  
Make sure you have everything setup  
You can also use the following flags to train

| Flag        | Function | Default  |
|:-------------|:-------------|:-----|
| --projecttag | specify the project's tag | "" |
| --versiontag | specify the version's tag | "" |
| --loadfile | file name you want to load | None |
| --resume | resume or not  | False |  

We strongly recommend you use some tags to make sure the program runs correctly
```commandline
cd ~/RedstoneTorch
python train.py --projecttag mem --versiontag mem1 --resume False
```

If you want to load from previous model to continue trainning progress:
```commandline
python train.py --projecttag 2018-10-30-04-07-40-043900-test --versiontag test2 --resume True --loadfile test1-CP1.pth
```
The above information can be obtained in the command line during trainning, like this:
```commandline
Validation Dice Coeff: 0.0754207968712
Checkpoint: 1 epoch; 13.0-13.0 step; dir: model/2018-10-30-04-07-40-043900-test/test1-CP1.pth
```
(The epoch starts from #1, whereas fold start from #0. Only Epoch got saved.)
### Evaluate and Display
The program use tensorboardX to display tensors  
Use command
```commandline
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=~/RedstoneTorch/model/PROJECTTAG --port=6006
```
to open tensorboad's display on port `6006` of your server after you run `train.py` where `PROJECTTAG` can be replaced with your project tag.

### Predict
Use predict.py to get the submit data table
```commandline
python predict.py --projecttag 2018-10-30-04-07-40-043900-test --versiontag test2 --loadfile test1-CP1.pth

```
After the prediction, you probably want to download the .csv file, the directory is here:
```commandline
RedstoneTorch/model/2018-10-30-04-07-40-043900-test/test1-CP1.pth-test-0.csv
```
## GCP Monitor and Logging
```commandline
# To install the Stackdriver monitoring agent:
$ curl -sSO https://dl.google.com/cloudagents/install-monitoring-agent.sh
$ sudo bash install-monitoring-agent.sh

# To install the Stackdriver logging agent:
$ curl -sSO https://dl.google.com/cloudagents/install-logging-agent.sh
$ sudo bash install-logging-agent.sh
```

## Dependencies
This package depends on
```
matplotlib
pydensecrf
numpy
Pillow
torch
torchvision
augmentor
tensorboardX
psutil
tensorboard
tensorflow

```
Please use `pip install` to install these dependencies.

## Directory

```
.
├── config.py
├── data
│   ├── sample_submission.csv
│   ├── test
│   │   └── [A LOT OF PICTURES]
│   ├── trian.csv
│   └── train
│   │   └── [A LOT OF PICTURES]
├── dataset
│   ├── hpa_dataset.py
│   ├── __init__.py
│   └── tgs_dataset.py
├── loss
│   ├── dice.py
│   ├── focal.py
│   ├── __init__.py
│   ├── iou.py
│   └── loss.py
├── model
├── net
│   ├── block.py
│   ├── __init__.py
│   ├── proteinet
│   │   ├── __init__.py
│   │   ├── proteinet_model.py
│   │   └── proteinet_parts.py
│   ├── resnet
│   │   ├── __init__.py
│   │   ├── resnet_extractor.py
│   │   └── resnet_model.py
│   ├── resunet
│   │   ├── __init__.py
│   │   ├── resunet_model.py
│   │   └── resunet_parts.py
│   ├── seinception
│   │   ├── __init__.py
│   │   ├── seinception_model.py
│   │   └── seinception_parts.py
│   ├── seresnet
│   │   ├── __init__.py
│   │   ├── seresnet_model.py
│   │   └── seresnet_parts.py
│   └── unet
│       ├── __init__.py
│       ├── unet_model.py
│       └── unet_parts.py
├── optimizer
│   ├── __init__.py
│   └── sgdw.py
├── pretained_model
│   ├── bninception.py
│   ├── inceptionresnetv2.py
│   ├── inceptionv4.py
│   ├── __init__.py
│   ├── nasnet.py
│   ├── resnext_features
│   │   ├── __init__.py
│   │   ├── resnext101_32x4d_features.py
│   │   └── resnext101_64x4d_features.py
│   ├── resnext.py
│   ├── senet.py
│   ├── torchvision_models.py
│   ├── utils.py
│   ├── vggm.py
│   ├── wideresnet.py
│   └── xception.py
├── project
│   ├── hpa_project.py
│   ├── __init__.py
│   └── tgs_project.py
├── README.md
├── requirements.txt
├── tensorboardwriter.py
├── train.py
├── tree.txt
└── utils
    ├── encode.py
    ├── __init__.py
    ├── memory.py
    └── postprocess.py

16 directories, 60 files
```

## Network Model
```
module.layer0.conv1.weight
module.layer0.bn1.weight
module.layer0.bn1.bias
module.layer1.0.conv1.weight
module.layer1.0.bn1.weight
module.layer1.0.bn1.bias
module.layer1.0.conv2.weight
module.layer1.0.bn2.weight
module.layer1.0.bn2.bias
module.layer1.0.conv3.weight
module.layer1.0.bn3.weight
module.layer1.0.bn3.bias
module.layer1.0.se_module.fc1.weight
module.layer1.0.se_module.fc1.bias
module.layer1.0.se_module.fc2.weight
module.layer1.0.se_module.fc2.bias
module.layer1.0.downsample.0.weight
module.layer1.0.downsample.1.weight
module.layer1.0.downsample.1.bias
module.layer1.1.conv1.weight
module.layer1.1.bn1.weight
module.layer1.1.bn1.bias
module.layer1.1.conv2.weight
module.layer1.1.bn2.weight
module.layer1.1.bn2.bias
module.layer1.1.conv3.weight
module.layer1.1.bn3.weight
module.layer1.1.bn3.bias
module.layer1.1.se_module.fc1.weight
module.layer1.1.se_module.fc1.bias
module.layer1.1.se_module.fc2.weight
module.layer1.1.se_module.fc2.bias
module.layer1.2.conv1.weight
module.layer1.2.bn1.weight
module.layer1.2.bn1.bias
module.layer1.2.conv2.weight
module.layer1.2.bn2.weight
module.layer1.2.bn2.bias
module.layer1.2.conv3.weight
module.layer1.2.bn3.weight
module.layer1.2.bn3.bias
module.layer1.2.se_module.fc1.weight
module.layer1.2.se_module.fc1.bias
module.layer1.2.se_module.fc2.weight
module.layer1.2.se_module.fc2.bias
module.layer2.0.conv1.weight
module.layer2.0.bn1.weight
module.layer2.0.bn1.bias
module.layer2.0.conv2.weight
module.layer2.0.bn2.weight
module.layer2.0.bn2.bias
module.layer2.0.conv3.weight
module.layer2.0.bn3.weight
module.layer2.0.bn3.bias
module.layer2.0.se_module.fc1.weight
module.layer2.0.se_module.fc1.bias
module.layer2.0.se_module.fc2.weight
module.layer2.0.se_module.fc2.bias
module.layer2.0.downsample.0.weight
module.layer2.0.downsample.1.weight
module.layer2.0.downsample.1.bias
module.layer2.1.conv1.weight
module.layer2.1.bn1.weight
module.layer2.1.bn1.bias
module.layer2.1.conv2.weight
module.layer2.1.bn2.weight
module.layer2.1.bn2.bias
module.layer2.1.conv3.weight
module.layer2.1.bn3.weight
module.layer2.1.bn3.bias
module.layer2.1.se_module.fc1.weight
module.layer2.1.se_module.fc1.bias
module.layer2.1.se_module.fc2.weight
module.layer2.1.se_module.fc2.bias
module.layer2.2.conv1.weight
module.layer2.2.bn1.weight
module.layer2.2.bn1.bias
module.layer2.2.conv2.weight
module.layer2.2.bn2.weight
module.layer2.2.bn2.bias
module.layer2.2.conv3.weight
module.layer2.2.bn3.weight
module.layer2.2.bn3.bias
module.layer2.2.se_module.fc1.weight
module.layer2.2.se_module.fc1.bias
module.layer2.2.se_module.fc2.weight
module.layer2.2.se_module.fc2.bias
module.layer2.3.conv1.weight
module.layer2.3.bn1.weight
module.layer2.3.bn1.bias
module.layer2.3.conv2.weight
module.layer2.3.bn2.weight
module.layer2.3.bn2.bias
module.layer2.3.conv3.weight
module.layer2.3.bn3.weight
module.layer2.3.bn3.bias
module.layer2.3.se_module.fc1.weight
module.layer2.3.se_module.fc1.bias
module.layer2.3.se_module.fc2.weight
module.layer2.3.se_module.fc2.bias
module.layer3.0.conv1.weight
module.layer3.0.bn1.weight
module.layer3.0.bn1.bias
module.layer3.0.conv2.weight
module.layer3.0.bn2.weight
module.layer3.0.bn2.bias
module.layer3.0.conv3.weight
module.layer3.0.bn3.weight
module.layer3.0.bn3.bias
module.layer3.0.se_module.fc1.weight
module.layer3.0.se_module.fc1.bias
module.layer3.0.se_module.fc2.weight
module.layer3.0.se_module.fc2.bias
module.layer3.0.downsample.0.weight
module.layer3.0.downsample.1.weight
module.layer3.0.downsample.1.bias
module.layer3.1.conv1.weight
module.layer3.1.bn1.weight
module.layer3.1.bn1.bias
module.layer3.1.conv2.weight
module.layer3.1.bn2.weight
module.layer3.1.bn2.bias
module.layer3.1.conv3.weight
module.layer3.1.bn3.weight
module.layer3.1.bn3.bias
module.layer3.1.se_module.fc1.weight
module.layer3.1.se_module.fc1.bias
module.layer3.1.se_module.fc2.weight
module.layer3.1.se_module.fc2.bias
module.layer3.2.conv1.weight
module.layer3.2.bn1.weight
module.layer3.2.bn1.bias
module.layer3.2.conv2.weight
module.layer3.2.bn2.weight
module.layer3.2.bn2.bias
module.layer3.2.conv3.weight
module.layer3.2.bn3.weight
module.layer3.2.bn3.bias
module.layer3.2.se_module.fc1.weight
module.layer3.2.se_module.fc1.bias
module.layer3.2.se_module.fc2.weight
module.layer3.2.se_module.fc2.bias
module.layer3.3.conv1.weight
module.layer3.3.bn1.weight
module.layer3.3.bn1.bias
module.layer3.3.conv2.weight
module.layer3.3.bn2.weight
module.layer3.3.bn2.bias
module.layer3.3.conv3.weight
module.layer3.3.bn3.weight
module.layer3.3.bn3.bias
module.layer3.3.se_module.fc1.weight
module.layer3.3.se_module.fc1.bias
module.layer3.3.se_module.fc2.weight
module.layer3.3.se_module.fc2.bias
module.layer3.4.conv1.weight
module.layer3.4.bn1.weight
module.layer3.4.bn1.bias
module.layer3.4.conv2.weight
module.layer3.4.bn2.weight
module.layer3.4.bn2.bias
module.layer3.4.conv3.weight
module.layer3.4.bn3.weight
module.layer3.4.bn3.bias
module.layer3.4.se_module.fc1.weight
module.layer3.4.se_module.fc1.bias
module.layer3.4.se_module.fc2.weight
module.layer3.4.se_module.fc2.bias
module.layer3.5.conv1.weight
module.layer3.5.bn1.weight
module.layer3.5.bn1.bias
module.layer3.5.conv2.weight
module.layer3.5.bn2.weight
module.layer3.5.bn2.bias
module.layer3.5.conv3.weight
module.layer3.5.bn3.weight
module.layer3.5.bn3.bias
module.layer3.5.se_module.fc1.weight
module.layer3.5.se_module.fc1.bias
module.layer3.5.se_module.fc2.weight
module.layer3.5.se_module.fc2.bias
module.layer3.6.conv1.weight
module.layer3.6.bn1.weight
module.layer3.6.bn1.bias
module.layer3.6.conv2.weight
module.layer3.6.bn2.weight
module.layer3.6.bn2.bias
module.layer3.6.conv3.weight
module.layer3.6.bn3.weight
module.layer3.6.bn3.bias
module.layer3.6.se_module.fc1.weight
module.layer3.6.se_module.fc1.bias
module.layer3.6.se_module.fc2.weight
module.layer3.6.se_module.fc2.bias
module.layer3.7.conv1.weight
module.layer3.7.bn1.weight
module.layer3.7.bn1.bias
module.layer3.7.conv2.weight
module.layer3.7.bn2.weight
module.layer3.7.bn2.bias
module.layer3.7.conv3.weight
module.layer3.7.bn3.weight
module.layer3.7.bn3.bias
module.layer3.7.se_module.fc1.weight
module.layer3.7.se_module.fc1.bias
module.layer3.7.se_module.fc2.weight
module.layer3.7.se_module.fc2.bias
module.layer3.8.conv1.weight
module.layer3.8.bn1.weight
module.layer3.8.bn1.bias
module.layer3.8.conv2.weight
module.layer3.8.bn2.weight
module.layer3.8.bn2.bias
module.layer3.8.conv3.weight
module.layer3.8.bn3.weight
module.layer3.8.bn3.bias
module.layer3.8.se_module.fc1.weight
module.layer3.8.se_module.fc1.bias
module.layer3.8.se_module.fc2.weight
module.layer3.8.se_module.fc2.bias
module.layer3.9.conv1.weight
module.layer3.9.bn1.weight
module.layer3.9.bn1.bias
module.layer3.9.conv2.weight
module.layer3.9.bn2.weight
module.layer3.9.bn2.bias
module.layer3.9.conv3.weight
module.layer3.9.bn3.weight
module.layer3.9.bn3.bias
module.layer3.9.se_module.fc1.weight
module.layer3.9.se_module.fc1.bias
module.layer3.9.se_module.fc2.weight
module.layer3.9.se_module.fc2.bias
module.layer3.10.conv1.weight
module.layer3.10.bn1.weight
module.layer3.10.bn1.bias
module.layer3.10.conv2.weight
module.layer3.10.bn2.weight
module.layer3.10.bn2.bias
module.layer3.10.conv3.weight
module.layer3.10.bn3.weight
module.layer3.10.bn3.bias
module.layer3.10.se_module.fc1.weight
module.layer3.10.se_module.fc1.bias
module.layer3.10.se_module.fc2.weight
module.layer3.10.se_module.fc2.bias
module.layer3.11.conv1.weight
module.layer3.11.bn1.weight
module.layer3.11.bn1.bias
module.layer3.11.conv2.weight
module.layer3.11.bn2.weight
module.layer3.11.bn2.bias
module.layer3.11.conv3.weight
module.layer3.11.bn3.weight
module.layer3.11.bn3.bias
module.layer3.11.se_module.fc1.weight
module.layer3.11.se_module.fc1.bias
module.layer3.11.se_module.fc2.weight
module.layer3.11.se_module.fc2.bias
module.layer3.12.conv1.weight
module.layer3.12.bn1.weight
module.layer3.12.bn1.bias
module.layer3.12.conv2.weight
module.layer3.12.bn2.weight
module.layer3.12.bn2.bias
module.layer3.12.conv3.weight
module.layer3.12.bn3.weight
module.layer3.12.bn3.bias
module.layer3.12.se_module.fc1.weight
module.layer3.12.se_module.fc1.bias
module.layer3.12.se_module.fc2.weight
module.layer3.12.se_module.fc2.bias
module.layer3.13.conv1.weight
module.layer3.13.bn1.weight
module.layer3.13.bn1.bias
module.layer3.13.conv2.weight
module.layer3.13.bn2.weight
module.layer3.13.bn2.bias
module.layer3.13.conv3.weight
module.layer3.13.bn3.weight
module.layer3.13.bn3.bias
module.layer3.13.se_module.fc1.weight
module.layer3.13.se_module.fc1.bias
module.layer3.13.se_module.fc2.weight
module.layer3.13.se_module.fc2.bias
module.layer3.14.conv1.weight
module.layer3.14.bn1.weight
module.layer3.14.bn1.bias
module.layer3.14.conv2.weight
module.layer3.14.bn2.weight
module.layer3.14.bn2.bias
module.layer3.14.conv3.weight
module.layer3.14.bn3.weight
module.layer3.14.bn3.bias
module.layer3.14.se_module.fc1.weight
module.layer3.14.se_module.fc1.bias
module.layer3.14.se_module.fc2.weight
module.layer3.14.se_module.fc2.bias
module.layer3.15.conv1.weight
module.layer3.15.bn1.weight
module.layer3.15.bn1.bias
module.layer3.15.conv2.weight
module.layer3.15.bn2.weight
module.layer3.15.bn2.bias
module.layer3.15.conv3.weight
module.layer3.15.bn3.weight
module.layer3.15.bn3.bias
module.layer3.15.se_module.fc1.weight
module.layer3.15.se_module.fc1.bias
module.layer3.15.se_module.fc2.weight
module.layer3.15.se_module.fc2.bias
module.layer3.16.conv1.weight
module.layer3.16.bn1.weight
module.layer3.16.bn1.bias
module.layer3.16.conv2.weight
module.layer3.16.bn2.weight
module.layer3.16.bn2.bias
module.layer3.16.conv3.weight
module.layer3.16.bn3.weight
module.layer3.16.bn3.bias
module.layer3.16.se_module.fc1.weight
module.layer3.16.se_module.fc1.bias
module.layer3.16.se_module.fc2.weight
module.layer3.16.se_module.fc2.bias
module.layer3.17.conv1.weight
module.layer3.17.bn1.weight
module.layer3.17.bn1.bias
module.layer3.17.conv2.weight
module.layer3.17.bn2.weight
module.layer3.17.bn2.bias
module.layer3.17.conv3.weight
module.layer3.17.bn3.weight
module.layer3.17.bn3.bias
module.layer3.17.se_module.fc1.weight
module.layer3.17.se_module.fc1.bias
module.layer3.17.se_module.fc2.weight
module.layer3.17.se_module.fc2.bias
module.layer3.18.conv1.weight
module.layer3.18.bn1.weight
module.layer3.18.bn1.bias
module.layer3.18.conv2.weight
module.layer3.18.bn2.weight
module.layer3.18.bn2.bias
module.layer3.18.conv3.weight
module.layer3.18.bn3.weight
module.layer3.18.bn3.bias
module.layer3.18.se_module.fc1.weight
module.layer3.18.se_module.fc1.bias
module.layer3.18.se_module.fc2.weight
module.layer3.18.se_module.fc2.bias
module.layer3.19.conv1.weight
module.layer3.19.bn1.weight
module.layer3.19.bn1.bias
module.layer3.19.conv2.weight
module.layer3.19.bn2.weight
module.layer3.19.bn2.bias
module.layer3.19.conv3.weight
module.layer3.19.bn3.weight
module.layer3.19.bn3.bias
module.layer3.19.se_module.fc1.weight
module.layer3.19.se_module.fc1.bias
module.layer3.19.se_module.fc2.weight
module.layer3.19.se_module.fc2.bias
module.layer3.20.conv1.weight
module.layer3.20.bn1.weight
module.layer3.20.bn1.bias
module.layer3.20.conv2.weight
module.layer3.20.bn2.weight
module.layer3.20.bn2.bias
module.layer3.20.conv3.weight
module.layer3.20.bn3.weight
module.layer3.20.bn3.bias
module.layer3.20.se_module.fc1.weight
module.layer3.20.se_module.fc1.bias
module.layer3.20.se_module.fc2.weight
module.layer3.20.se_module.fc2.bias
module.layer3.21.conv1.weight
module.layer3.21.bn1.weight
module.layer3.21.bn1.bias
module.layer3.21.conv2.weight
module.layer3.21.bn2.weight
module.layer3.21.bn2.bias
module.layer3.21.conv3.weight
module.layer3.21.bn3.weight
module.layer3.21.bn3.bias
module.layer3.21.se_module.fc1.weight
module.layer3.21.se_module.fc1.bias
module.layer3.21.se_module.fc2.weight
module.layer3.21.se_module.fc2.bias
module.layer3.22.conv1.weight
module.layer3.22.bn1.weight
module.layer3.22.bn1.bias
module.layer3.22.conv2.weight
module.layer3.22.bn2.weight
module.layer3.22.bn2.bias
module.layer3.22.conv3.weight
module.layer3.22.bn3.weight
module.layer3.22.bn3.bias
module.layer3.22.se_module.fc1.weight
module.layer3.22.se_module.fc1.bias
module.layer3.22.se_module.fc2.weight
module.layer3.22.se_module.fc2.bias
module.layer4.0.conv1.weight
module.layer4.0.bn1.weight
module.layer4.0.bn1.bias
module.layer4.0.conv2.weight
module.layer4.0.bn2.weight
module.layer4.0.bn2.bias
module.layer4.0.conv3.weight
module.layer4.0.bn3.weight
module.layer4.0.bn3.bias
module.layer4.0.se_module.fc1.weight
module.layer4.0.se_module.fc1.bias
module.layer4.0.se_module.fc2.weight
module.layer4.0.se_module.fc2.bias
module.layer4.0.downsample.0.weight
module.layer4.0.downsample.1.weight
module.layer4.0.downsample.1.bias
module.layer4.1.conv1.weight
module.layer4.1.bn1.weight
module.layer4.1.bn1.bias
module.layer4.1.conv2.weight
module.layer4.1.bn2.weight
module.layer4.1.bn2.bias
module.layer4.1.conv3.weight
module.layer4.1.bn3.weight
module.layer4.1.bn3.bias
module.layer4.1.se_module.fc1.weight
module.layer4.1.se_module.fc1.bias
module.layer4.1.se_module.fc2.weight
module.layer4.1.se_module.fc2.bias
module.layer4.2.conv1.weight
module.layer4.2.bn1.weight
module.layer4.2.bn1.bias
module.layer4.2.conv2.weight
module.layer4.2.bn2.weight
module.layer4.2.bn2.bias
module.layer4.2.conv3.weight
module.layer4.2.bn3.weight
module.layer4.2.bn3.bias
module.layer4.2.se_module.fc1.weight
module.layer4.2.se_module.fc1.bias
module.layer4.2.se_module.fc2.weight
module.layer4.2.se_module.fc2.bias
module.last_linear.weight
module.last_linear.bias
```
| Class | BestThreshold(Raw) | BestThreshold(Smoothed) |
|-------|--------------------|-------------------------|
| All   | 0.2332             | 0.2196                  |
| 0     | 0.07007            | 0.1547                  |
| 1     | 0.9650             | 0.1571                  |
| 2     | 0.8579             | 0.1798                  |
| 3     | 0.1662             | 0.1931                  |
| 4     | 0.7728             | 0.1324                  |
| 5     | 0.01001            | 0.1926                  |
| 6     | 0.01201            | 0.09215                 |
| 7     | 0.0030030          | 0.1843                  |
| 8     | 0.7978             | 0.1669                  |
| 9     | 0.01602            | 0.09612                 |
| 10    | 0.1982             | 0.1602                  |
| 11    | 0.5325             | 0.1286                  |
| 12    | 0.2152             | 0.1722                  |
| 13    | 0.03103            | 0.1544                  |
| 14    | 0.004004           | 0.04645                 |
| 15    | 0.04304            | 0.06961                 |
| 16    | 0.005005           | 0.1499                  |
| 17    | 0.003003           | 0.06373                 |
| 18    | 0.09810            | 0.1001                  |
| 19    | 0.04204            | 0.1706                  |
| 20    | 0.01101            | 0.1264                  |
| 21    | 0.01101            | 0.1121                  |
| 22    | 0.01702            | 0.08679                 |
| 23    | 0.000              | 0.000                   |
| 24    | 0.03504            | 0.08634                 |
| 25    | 0.01502            | 0.1221                  |
| 26    | 0.0050050          | 0.1943                  |
| 27    | 0.01502            | 0.1180                  |
