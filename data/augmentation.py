import sys
import os
import Augmentor
from optparse import OptionParser




def get_args():
    parser = OptionParser()
    # parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int', help='number of epochs')
    # parser.add_option('-b', '--batch-size', dest='batchsize', default=10, type='int', help='batch size')
    # parser.add_option('-l', '--learning-rate', dest='lr', default=0.1, type='float', help='learning rate')
    # # parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default="", help='use cuda, please put all gpu id here')
    # parser.add_option('-g', '--gpu', dest='gpu', default="", help='use cuda, please put all gpu id here')
    # parser.add_option('-c', '--load', dest='load', default=False, help='load file model')
    # parser.add_option('-w', '--weight_init', dest='weight_init', default=0.01, type='float',
    #                   help='weight initialization number')
    # parser.add_option('-v', '--val_percent', dest='val_percent', default=0.05, type='float',
    #                   help='percent for validation')
    # parser.add_option('-p', '--dir_prefix', dest='dir_prefix', default='', help='the root directory')
    # parser.add_option('-d', '--data_percent', dest='data_percent', default=1.0, type='float', help='the root directory')
    # parser.add_option('-i', '--visualization', dest='visualization', action='store_true', default="False",
    #                   help='visualization the data')
    # parser.add_option('-t', '--tag', dest='tag', default="", help='tag for tensorboard-log')
    parser.add_option('-i', '--image', dest='image_dir', default='/train/images', help='the image directory to running directory')
    parser.add_option('-m', '--mask', dest='mask_dir', default='/train/masks', help='the mask directory related to running directory')
    parser.add_option('-s', '--sample', dest='sample', default=4000, help='the number of picture generated')

    (options, args) = parser.parse_args()
    return options



if __name__ == '__main__':
    args = get_args()
    # init artgs
    try:
        p = Augmentor.Pipeline(args.image_dir)
        # Point to a directory containing ground truth data.
        # Images with the same file names will be added as ground truth data
        # and augmented in parallel to the original data.
        p.ground_truth(args.mask_dir)

        p.greyscale(probability=1.0) # I don't need grey scale because resnet does not support, but add it anyway in case it says wrong mode
        p.resize(1, 224, 224, resample_filter=u'BICUBIC') #BICUBIC, BILINEAR, ANTIALIAS, or NEAREST. https://i.stack.imgur.com/orwNd.png
        p.random_brightness(probability=0.5, min_factor=0.95, max_factor=1.05)
        # p.black_and_white(probability=1.0, threshold=128)
        p.flip_random(probability=0.75)
        p.rotate_random_90(probability=0.75)
        p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
        # p.crop_random(probability=0.5, percentage_area, randomise_percentage_area=False) already 'croped' by skew
        p.skew(probability=0.5, magnitude=0.1)
        p.random_distortion(probability=0.8, grid_width=16, grid_height=16, magnitude=4)

        # 1.5625% data will not be modified


        # Add operations to the pipeline as normal:
        p.sample(args.sample)


    except KeyboardInterrupt as e:
        print (e)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
# python data/augmentation.py --image 'data/train/images' --mask 'data/train/masks' --sample '40000'