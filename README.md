# Redstone Torch
![RedstoneTorch](https://d1u5p3l4wpay3k.cloudfront.net/minecraft_gamepedia/d/da/Redstone_Torch.png)


## Models


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

We strongly recommand you use some tags to make sure the program runs correctly
```commandline
cd ~/RedstoneTorch
python train.py --projecttag test1 --versiontag test1 --resume False
```

If you want to load from previous model to continue trainning progress:
```commandline
python train.py --projecttag 2018-10-30-04-07-40-043900-test --versiontag test2 --resume True --loadfile test1CP1.pth
```
The above information can be obtained in the command line during trainning, like this:
```commandline
Validation Dice Coeff: 0.0754207968712
Checkpoint: 1 epoch; 13.0-13.0 step; dir: model/2018-10-30-04-07-40-043900-test/test1CP1.pth
```
### Evaluate and Display
The program use tensorboardX to display tensors  
Use command
```commandline
python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=~/RedstoneTorch/model/PROJECTTAG --port=6006
```
to open tensorboad's display on port `6006` of your server after you run `train.py` where `PROJECTTAG` can be replaced with your project tag.

### Predict
Use predict.py to get the sumbit datatable
```commandline
python train.py --projecttag test --versiontag test --loadfile
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
