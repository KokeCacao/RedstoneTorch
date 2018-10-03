# full assembly of the sub-parts to form the complete net
import torchvision

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        # add depth data here
        self.join = join()
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x, depth):
        # print("x size = {}".format(x.size())) [32, 3, 224, 224]
        x1 = self.inc(x)
        # print("x1 size = {}".format(x1.size())) [32, 64, 224, 224]
        x2 = self.down1(x1)
        # print("x2 size = {}".format(x2.size())) [32, 128, 112, 112]
        x3 = self.down2(x2)
        # print("x3 size = {}".format(x3.size())) [32, 256, 56, 56]
        x4 = self.down3(x3)
        # print("x4 size = {}".format(x4.size())) [32, 512, 28, 28]

        # add depth data here
        join = self.join(x4, depth)
        # print("join size = {}".format(join.size())) [32, 512, 28, 28]

        x5 = self.down4(join)
        # print("x5 size = {}".format(x5.size())) [32, 512, 14, 14]
        x = self.up1(x5, x4)
        # print("after up1 size = {}".format(x.size())) [32, 256, 28, 28]
        x = self.up2(x, x3)
        # print("after up2 size = {}".format(x.size())) [32, 128, 56, 56]
        x = self.up3(x, x2)
        # print("after up3 size = {}".format(x.size())) [32, 64, 112, 112]
        x = self.up4(x, x1)
        # print("after up4 size = {}".format(x.size())) [32, 64, 224, 224]
        x = self.outc(x)
        # print("outc size = {}".format(x.size())) [32, 1, 224, 224]
        return x

class UNetResNet(nn.Module):
    """PyTorch U-Net model using ResNet(34, 101 or 152) encoder.

    UNet: https://arxiv.org/abs/1505.04597
    ResNet: https://arxiv.org/abs/1512.03385
    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/

    Args:
            encoder_depth (int): Depth of a ResNet encoder (34, 101 or 152).
            num_classes (int): Number of output classes.
            num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
            dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
            pretrained (bool, optional):
                False - no pre-trained weights are being used.
                True  - ResNet encoder is pre-trained on ImageNet.
                Defaults to False.
            is_deconv (bool, optional):
                False: bilinear interpolation is used in decoder.
                True: deconvolution is used in decoder.
                Defaults to False.

    """

    def __init__(self, encoder_depth, num_classes, num_filters=32, dropout_2d=0.2,
                 pretrained=False, is_deconv=False):
        super(UNetResNet, self).__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 50: # 224*224 input
            self.encoder = torchvision.models.resnet50(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152, 50 version of Resnet are implemented')


        self.pool = nn.MaxPool2d(2, 2)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1) #kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(bottom_channel_nr,
                                     num_filters * 8 * 2,
                                     num_filters * 8,
                                     is_deconv, padding=2)
        self.dec5 = DecoderBlockV2(bottom_channel_nr + num_filters * 8,
                                   num_filters * 8 * 2,
                                   num_filters * 8,
                                   is_deconv)
        self.dec4 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8,
                                   num_filters * 8 * 2,
                                   num_filters * 8,
                                   is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8,
                                   num_filters * 4 * 2,
                                   num_filters * 2,
                                   is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2,
                                   num_filters * 2 * 2,
                                   num_filters * 2 * 2,
                                   is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2,
                                   num_filters * 2 * 2,
                                   num_filters,
                                   is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        print (x.size()) # (32, 3, 101, 101)(32, 3, 224, 224)
        conv1 = self.conv1(x)
        print (conv1.size()) # (32, 64, 25, 25)(32, 64, 56, 56)(32, 64, 57, 57)
        conv2 = self.conv2(conv1)
        print (conv2.size()) # (32, 256, 25, 25)(32, 256, 56, 56)(32, 256, 57, 57)
        conv3 = self.conv3(conv2)
        print (conv3.size()) # (32, 512, 13, 13)(32, 512, 28, 28)(32, 512, 29, 29)
        conv4 = self.conv4(conv3)
        print (conv4.size()) # (32, 1024, 7, 7)=(32, 1024, 14, 14)(32, 1024, 15, 15)
        conv5 = self.conv5(conv4)
        print (conv5.size()) # (32, 2048, 4, 4)(32, 2048, 7, 7)(32, 2048, 8, 8)

        pool = self.pool(conv5)
        print (pool.size()) # (32, 2048, 2, 2)(32, 2048, 3, 3)(32, 2048, 5, 5)
        center = self.center(pool)
        print (center.size()) # (32, 256, 4, 4)(32, 256, 6, 6)(32, 256, 10, 10)

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        print (dec5.size()) # (32, 256, 8, 8)=

        dec4 = self.dec4(torch.cat([dec5, conv4], 1)) #=
        print (dec4.size())
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        print (dec3.size())
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        print (dec2.size())
        dec1 = self.dec1(dec2)
        print (dec1.size())
        dec0 = self.dec0(dec1)
        print (dec0.size())






        # (32, 3, 101, 101)
        # (32, 64, 25, 25)
        # (32, 256, 25, 25)
        # (32, 512, 13, 13)
        # (32, 1024, 7, 7)=
        # (32, 2048, 4, 4)
        # ======
        # (32, 2048, 2, 2)
        # (32, 256, 4, 4)

        # (32, 256, 8, 8)=




        # (32, 3, 152, 152)
        # (32, 64, 38, 38)
        # (32, 256, 38, 38)
        # (32, 512, 19, 19)
        # (32, 1024, 10, 10)
        # (32, 2048, 5, 5)=
        # ======
        # (32, 2048, 2, 2)
        # (32, 256, 4, 4)=

        return self.final(F.dropout2d(dec0, p=self.dropout_2d))

class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True, padding=1):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=padding),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)