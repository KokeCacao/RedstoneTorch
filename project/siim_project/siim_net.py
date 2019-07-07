# full assembly of the sub-parts to form the complete net
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: change the backbone to SEResNeXt

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        # x = [self.conv(x[0]), x[1]]
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        # x = [self.mpconv(x[0]), x[1]]
        return x

class join(nn.Module):
    def __init__(self):
        super(join, self).__init__()

    def forward(self, x, data):
        # x = torch.stack((x, self.data), dim=-1)
        # I am not sure whether this step is differentiable by autograd
        # print (data.size())
        # print (torch.ones(x.size(), dtype=torch.double).size())
        # copy = torch.ones(x.size(), dtype=torch.double)
        # copy[0] = copy[0]*data[0]
        # copy[1] = copy[1]*data[1]
        # copy[2] = copy[2]*data[2]
        # copy[3] = copy[3]*data[3]
        # copy[4] = copy[4]*data[4]
        # copy[5] = copy[5]*data[5]
        # copy[6] = copy[6]*data[6]
        # copy[7] = copy[7]*data[7]
        # copy[8] = copy[8]*data[8]
        # copy[9] = copy[9]*data[9]
        #
        # # x = torch.cat((x, new), 0)
        # x = torch.mul(x, copy.float())
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
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

    def __init__(self, encoder_depth=50, num_classes=1, num_filters=32, dropout_2d=0.2,
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
                                     is_deconv, kernel_size=5, stride=2, padding=1)
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

        # WORKING
        # self.inter_conv = nn.Conv2d(in_, out, 1)
        # self.linear1 = nn.Linear(64, num_classes)
        # self.linear2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # print (x.size()) # (32, 3, 101, 101)(32, 3, 224, 224)
        conv1 = self.conv1(x)
        # print (conv1.size()) # (32, 64, 25, 25)(32, 64, 56, 56)(32, 64, 57, 57)
        conv2 = self.conv2(conv1)
        # print (conv2.size()) # (32, 256, 25, 25)(32, 256, 56, 56)(32, 256, 57, 57)
        conv3 = self.conv3(conv2)
        # print (conv3.size()) # (32, 512, 13, 13)(32, 512, 28, 28)(32, 512, 29, 29)
        conv4 = self.conv4(conv3)
        # print (conv4.size()) # (32, 1024, 7, 7)=(32, 1024, 14, 14)(32, 1024, 15, 15)
        conv5 = self.conv5(conv4)
        # print (conv5.size()) # (32, 2048, 4, 4)(32, 2048, 7, 7)(32, 2048, 8, 8)

        pool = self.pool(conv5)
        # print (pool.size()) # (32, 2048, 2, 2)(32, 2048, 3, 3)(32, 2048, 5, 5)
        center = self.center(pool)
        # print (center.size()) # (32, 256, 4, 4)(32, 256, 6, 6)(32, 256, 10, 10)

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        # print (dec5.size()) # (32, 256, 8, 8)=

        dec4 = self.dec4(torch.cat([dec5, conv4], 1)) #=
        # print (dec4.size())
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        # print (dec3.size())
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        # print (dec2.size())
        dec1 = self.dec1(dec2)
        # print (dec1.size())
        dec0 = self.dec0(dec1)
        # print (dec0.size())

        # WORKING
        # inter = F.avg_pool2d(pool, pool.size()[3]) #average
        #
        # inter = self.inter_conv(inter)
        # inter = self.linear1(inter)
        # inter = self.linear2(inter)
        # inter = F.softmax(inter, )
        # # softmax
        # pool = pool.view(pool.size(0), -1)
        # pool = self.linear(pool)



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
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True, kernel_size=4, stride=2, padding=1):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=kernel_size, stride=stride,
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
        # print("conv3x3:", x.size())
        x = self.activation(x)
        return x

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

def resunet(encoder_depth=50, num_classes=1, num_filters=32, dropout_2d=0.2,
                 pretrained=False, is_deconv=False):
    return UNetResNet(encoder_depth, num_classes, num_filters, dropout_2d, pretrained, is_deconv)