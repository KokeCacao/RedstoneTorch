# Redstone Torch

## Usage
The folowing instructions are made so that you can use this library

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

### Evaluate and Display
The program use tensorboardX to display tensors  
Use command `python .local/lib/python2.7/site-packages/tensorboard/main.py --logdir=XXX --port=6006`
to open tensorboad's display on port `6006` of your server after you run `train.py`

### Predict
Use predict.py to get the sumbit datatable

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
│   ├── tgs.zip
│   └── training_upload.csv
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
