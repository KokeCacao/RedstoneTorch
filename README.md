# Pytorch-UNet
Customized implementation of the [U-Net](https://arxiv.org/pdf/1505.04597.pdf) in Pytorch for Kaggle's [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge).

This model scored a [dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) of 0.988423 (511 out of 735), which is bad but could be improved with more training, data augmentation, fine tuning, and playing with CRF post-processing.

The model used for the last submission is stored in the `MODEL.pth` file, if you wish to play with it. The data is available on the [Kaggle website](https://www.kaggle.com/c/carvana-image-masking-challenge/data).

## Usage

### Prediction

You can easily test the output masks on your images via the CLI.
To see all options:
`python predict.py -h`

To predict a single image and save it:

`python predict.py -i image.jpg -o ouput.jpg`

To predict a multiple images and show them without saving them:

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

You can use the cpu-only version with `--cpu`.

You can specify which model file to use with `--model MODEL.pth`.

## Warning
In order to process the image, it is splitted into two squares (a left on and a right one), and each square is passed into the net. The two square masks are then merged again to produce the final image. As a consequence, the height of the image must be strictly superior than half the width. Make sure the width is even too.
