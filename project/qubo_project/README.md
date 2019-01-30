```text
oard/main.py --logdir=RedstoneTorch/model/2018-11-22-05-08-27-139778-seresnext-augnew-2pooling/ --port=6006

python predict.py --versiontag gsfv1 --loaddir 2018-11-22-05-08-27-139778-seresnext-augnew-2pooling --loadfile 4ebebfe-CP44_F[1]_PTseresnext-augnew-2pooling_VT4ebebfe_LR0.1_BS64_IMG224.pth
Download: RedstoneTorch/model/2018-11-22-05-08-27-139778-seresnext-augnew-2pooling/4ebebfe-CP44_F[1]_PTseresnext-augnew-2pooling_VT4ebebfe_LR0.1_BS64_IMG224.pth-gsfv1-F0-T0.268.csv
python predict.py --versiontag gsfv2 --loaddir 2018-11-22-05-08-27-139778-seresnext-augnew-2pooling --loadfile 4ebebfe-CP44_F[1]_PTseresnext-augnew-2pooling_VT4ebebfe_LR0.1_BS64_IMG224.pth

python train.py --projecttag first --versiontag 74e39ce1
python train.py --projecttag second --versiontag 23d1a59e

python train.py --projecttag second --versiontag 23d1a59e --loaddir 2019-01-30-20-31-05-899748-second --loadfile 23d1a59e-CP18_F[1]_PT2019-01-30-20-31-05-899748-second_VT23d1a59e_LR0.1_BS32_IMG224.pth
```