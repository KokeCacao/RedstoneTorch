# Redstone Torch
![RedstoneTorch](https://d1u5p3l4wpay3k.cloudfront.net/minecraft_gamepedia/d/da/Redstone_Torch.png)


## Models
```text
python train.py --projecttag base --versiontag base1 --resume False
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-02-20-23-22-547213-test/ --port=6006
//memory leak 8827 at step 1.3k from 4788
//10 fold 10 train
//BAD MODEL
=
python train.py --projecttag mem --versiontag mem1 --resume False
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-02-20-23-22-547213-test/ --port=6006
//memory leak from 8285 at step 1.4k, 6577
//10 fold 1 train
//3 epoch, loss=0.5556
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
//memory leak around 7G min stable, 2nd epoch
=
python train.py --projecttag mem3 --versiontag mem3 --resume False
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-03-14-14-20-075060-mem3/ --port=6006  
//No memory leak at epoch 3 after delete all extraneous things. memory around 2.5G
//CPU memory leak, GPU fine
=
python train.py --projecttag mem4 --versiontag mem4 --resume False
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=RedstoneTorch/model/2018-11-03-18-58-57-766148-mem4/ --port=6006
//open extraneous things, clean-up loss.detach(), clean cache() outside of the epoch(), del more things
//add f1

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
