# Redstone Torch
![RedstoneTorch](https://vignette.wikia.nocookie.net/minecraft/images/c/c2/RedstoneTorchNew.png/revision/latest)


## Models
HPA Project  
Qubo Project

## Usage
The folowing instructions are made so that you can use this library

### Data
#### Kaggle Data
The dataset is provided by Kaggle  
However, kaggle api is not very easy to use on remote server

Please use this chrome plugging to get `cookie.txt` file: [Here](https://chrome.google.com/webstore/detail/cookiestxt/njabckikapfpffapmjgojcnbfjonfjfg?hl=zh-CN)

After you upload your `cookie.txt` file to your remote server, use command(provided by [CarlosSouza](https://www.kaggle.com/c/cdiscount-image-classification-challenge/discussion/39492))

Please run the following command in your ~/RedstoneTorch directory
```commandline
cd ~/RedstoneTorch/data/DATASET
wget -x --load-cookies ~/cookies.txt -nH --cut-dirs=5 LINK

```
The `DATASET` can be replaced with `human-protein-atlas-image-classification`  
The command above will create a file named `data` and put your file `download-all` in it.  
So you need to unzip the `doanload-all`  
To do so, run the following command
```commandline
unzip ~/RedstoneTorch/data/download-all -d ~/RedstoneTorch/data/DATASET
```
and then you need to unzip the `train.zip` and `test.zip`
```commandline
unzip ~/RedstoneTorch/data/DATASET/train.zip -d ~/RedstoneTorch/data/DATASET/train
unzip ~/RedstoneTorch/data/DATASET/test.zip -d ~/RedstoneTorch/data/DATASET/test
```
Please use `sudo` in front of these command if the terminal says that you don't have permissions to do so

However, you may not have the full permission to read download file, use
```commandline
sudo chmod -R a+rwx train.csv
```
to give yourself permission to read.  
If you want to connect to your machine
```
ssh -i '/home/koke_cacao/.ssh/google_compute_engine' koke_cacao@35.229.123.118
```
#### Upload Data
You can also use `rsync` to upload data to your server like:
```
rsync -P --rsh=ssh -r /home/koke_cacao/Documents/Koke_Cacao/Python/WorkSpace/RedstoneTorch/data/qubo_dataset/preprocessed koke_cacao@xxx.xxx.xxx.xxx:/home/k1412042720/RedstoneTorch/data/qubo_dataset/preprocessed
```
If you get errors about mkdir, you probably does not have access to other user's account using ssh.  
So you should upload to the folder you have access to and then copy back on cloud.  

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

| Input Image Size | Speed       | Batch Size | Format | Device                     |
|------------------|-------------|------------|--------|----------------------------|
| 4x1728x1728      | 1.16s/img   | 1          | jpg    | 16CPU, 1 Nvidia Tesla P100 |
| 4x512x512        | 0.0128s/img | 64         | npy    | 16CPU, 1 Nvidia Tesla P100 |
| 4x512x512        | 0.0769s/img | 1          | npy    | 16CPU, 1 Nvidia Tesla P100 |

|         | Correct Label | Total Label | Binary Accuracy | F1-Macro Score | Precision | Recall | IOU Score |
|---------|---------------|-------------|-----------------|----------------|-----------|--------|-----------|
| Human   | 5360          | 5880        | 91.15%          | 0.1124         | 44.67%    | 27.46% | 27.29%    |
| Machine | 301384        | 311108      | 96.87%          | 0.3407         | 67.29%    | 69.23% | 63.07%    |
