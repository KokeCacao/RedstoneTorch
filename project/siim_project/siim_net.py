# full assembly of the sub-parts to form the complete net
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: change the backbone to SEResNeXt
from torch.utils import model_zoo

from net import senet154, se_resnet152, se_resnext101_32x4d, se_resnext50_32x4d
from net.ibnnet import IBN, resnext101_ibn_a
from net.pytorch_resnet import resnet34


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

        """EDIT CHANNEL TO 1"""
        self.encoder.conv1 = self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

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


"""========================================================================"""
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

def conv3x3(in_, out, bias=True):
    return nn.Conv2d(in_, out, 3, padding=1, bias=bias)

def conv7x7(in_, out, bias=True):
    return nn.Conv2d(in_, out, 7, padding=3, bias=bias)

def conv5x5(in_, out, bias=True):
    return nn.Conv2d(in_, out, 5, padding=2, bias=bias)

def conv1x1(in_, out, bias=True):
    return nn.Conv2d(in_, out, 1, padding=0, bias=bias)

class ConvRelu(nn.Module):
    def __init__(self, in_, out, kernel_size, norm_type = None):
        super(ConvRelu,self).__init__()

        is_bias = True
        self.norm_type = norm_type
        if norm_type == 'batch_norm':
            self.norm = nn.BatchNorm2d(out)
            is_bias = False

        elif norm_type == 'instance_norm':
            self.norm = nn.InstanceNorm2d(out)
            is_bias = True

        if kernel_size == 3:
            self.conv = conv3x3(in_, out, is_bias)
        elif kernel_size == 7:
            self.conv = conv7x7(in_, out, is_bias)
        elif kernel_size == 5:
            self.conv = conv5x5(in_, out, is_bias)
        elif kernel_size == 1:
            self.conv = conv1x1(in_, out, is_bias)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.norm_type is not None:
            x = self.conv(x)
            x = self.norm(x)
            x = self.activation(x)
        else:
            x = self.conv(x)
            x = self.activation(x)
        return x

class ImprovedIBNaDecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(ImprovedIBNaDecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = IBN(in_channels // 4)
        self.relu = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 4, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, int(channel/reduction), bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(int(channel/reduction), channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel//reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)

class Decoder(nn.Module):
    def __init__(self,in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=3,padding=1),
                                   nn.BatchNorm2d(channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(channels, out_channels, kernel_size=3,padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))
        self.SCSE = SCSEBlock(out_channels)

    def forward(self, x, e = None):
        x = F.upsample(x, scale_factor=2, mode='bilinear')
        if e is not None:
            x = torch.cat([x,e],1)
            x = F.dropout2d(x, p = 0.50)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.SCSE(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes):
        super(Bottleneck, self).__init__()
        planes = inplanes // 4

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)

        self.relu = nn.ReLU(inplace=True)

        self.is_skip = True
        if inplanes != outplanes:
            self.is_skip = False


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.is_skip:
            out += residual
        out = self.relu(out)

        return out

class Decoder_bottleneck(nn.Module):
    def __init__(self,in_channels, channels, out_channels):
        super(Decoder_bottleneck, self).__init__()

        self.block1 = Bottleneck(in_channels, channels)
        self.block2 = Bottleneck(channels, out_channels)
        self.SCSE = SCSEBlock(out_channels)

    def forward(self, x, e = None):
        x = F.upsample(x, scale_factor=2, mode='bilinear')
        if e is not None:
            x = torch.cat([x,e],1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.SCSE(x)
        return x

class model34_DeepSupervion(nn.Module):
    """Change mask_class from 2 to 1 since it is binary classification"""
    def __init__(self, num_classes=1, mask_class = 1):
        super(model34_DeepSupervion, self).__init__()

        self.num_classes = num_classes



        """Change Input Architecture"""
        self.encoder = resnet34(pretrained=True)

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
        self.center_conv1x1 = nn.Conv2d(512, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, mask_class)

        self.center = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2,stride=2))

        self.decoder5 = Decoder(256 + 512, 512, 64)
        self.decoder4 = Decoder(64 + 256, 256, 64)
        self.decoder3 = Decoder(64 + 128, 128, 64)
        self.decoder2 = Decoder(64 + 64, 64, 64)
        self.decoder1 = Decoder(64, 32, 64)

        self.logits_no_empty = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 1, kernel_size=1, padding=0))

        self.logits_final = nn.Sequential(nn.Conv2d(320+64, 64, kernel_size=3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 1, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)     #1/4
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        center_512 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_512)
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten)

        f = self.center(conv5)
        d5 = self.decoder5(f, conv5)
        d4 = self.decoder4(d5, conv4)
        d3 = self.decoder3(d4, conv3)
        d2 = self.decoder2(d3, conv2)
        d1 = self.decoder1(d2)

        hypercol = torch.cat((
            d1,
            F.upsample(d2, scale_factor=2,mode='bilinear'),
            F.upsample(d3, scale_factor=4, mode='bilinear'),
            F.upsample(d4, scale_factor=8, mode='bilinear'),
            F.upsample(d5, scale_factor=16, mode='bilinear')),1)
        hypercol = F.dropout2d(hypercol, p = 0.50)

        x_no_empty = self.logits_no_empty(hypercol)
        hypercol_add_center = torch.cat((
            hypercol,
            F.upsample(center_64, scale_factor=hypercol.shape[2],mode='bilinear')),1)

        x_final = self.logits_final( hypercol_add_center)
        return center_fc, x_no_empty, x_final

class model50A_DeepSupervion(nn.Module):
    def __init__(self, num_classes=1, test=False):
        super(model50A_DeepSupervion, self).__init__()

        self.num_classes = num_classes
        self.test = test
        """Pretrain edited to false for 1 channel input"""
        self.encoder = se_resnext50_32x4d()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                   self.encoder.layer0.bn1,
                                   self.encoder.layer0.relu1)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center_global_pool = nn.AdaptiveAvgPool2d([1, 1])
        self.center_conv1x1 = nn.Conv2d(512 * 4, 64, kernel_size=1)
        if not self.test:
            """Change out_feature from 2 to 1 since it is binary classification"""
            self.center_fc = nn.Linear(64, 1)

        self.center = nn.Sequential(nn.Conv2d(512*4, 512, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2,stride=2))

        self.decoder5 = Decoder(256 + 512*4, 512, 64)
        self.decoder4 = Decoder(64 + 256*4, 256, 64)
        self.decoder3 = Decoder(64 + 128*4, 128, 64)
        self.decoder2 = Decoder(64 + 64*4, 64, 64)
        self.decoder1 = Decoder(64, 32, 64)

        self.logits_no_empty = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 1, kernel_size=1, padding=0))

        self.logits_final = nn.Sequential(nn.Conv2d(320+64, 64, kernel_size=3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 1, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)     #1/4
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        center_512 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_512)
        if not self.test:
            center_64_flatten = center_64.view(center_64.size(0), -1)
            center_fc = self.center_fc(center_64_flatten)

        f = self.center(conv5)
        d5 = self.decoder5(f, conv5)
        d4 = self.decoder4(d5, conv4)
        d3 = self.decoder3(d4, conv3)
        d2 = self.decoder2(d3, conv2)
        d1 = self.decoder1(d2)

        hypercol = torch.cat((
            d1,
            F.upsample(d2, scale_factor=2,mode='bilinear'),
            F.upsample(d3, scale_factor=4, mode='bilinear'),
            F.upsample(d4, scale_factor=8, mode='bilinear'),
            F.upsample(d5, scale_factor=16, mode='bilinear')),1)
        hypercol = F.dropout2d(hypercol, p = 0.50)

        x_no_empty = self.logits_no_empty(hypercol)
        hypercol_add_center = torch.cat((
            hypercol,
            F.upsample(center_64, scale_factor=hypercol.shape[2],mode='bilinear')),1)

        x_final = self.logits_final(hypercol_add_center)
        if not self.test:
            return center_fc, x_no_empty, x_final
        return None, x_no_empty, x_final

class model50A_slim_DeepSupervion(nn.Module):
    def __init__(self, num_classes=1, test=False):
        super(model50A_slim_DeepSupervion, self).__init__()

        self.num_classes = num_classes
        self.test = test
        self.encoder = se_resnext50_32x4d()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                   self.encoder.layer0.bn1,
                                   self.encoder.layer0.relu1)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        if not self.test:
            self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
            self.center_conv1x1 = nn.Conv2d(512*4, 64, kernel_size=1)
            self.center_fc = nn.Linear(64, 2)

        self.center = nn.Sequential(nn.Conv2d(512*4, 512, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2,stride=2))

        self.dec5_1x1 = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=1), nn.BatchNorm2d(512),nn.ReLU(inplace=True))
        self.decoder5 = Decoder_bottleneck(256 + 512, 512, 64)

        self.dec4_1x1 = nn.Sequential(nn.Conv2d(256 * 4, 256, kernel_size=1), nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.decoder4 = Decoder_bottleneck(64 + 256, 256, 64)

        self.dec3_1x1 = nn.Sequential(nn.Conv2d(128 * 4, 128, kernel_size=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.decoder3 = Decoder_bottleneck(64 + 128, 128, 64)

        self.dec2_1x1 = nn.Sequential(nn.Conv2d(64 * 4, 64, kernel_size=1), nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.decoder2 = Decoder_bottleneck(64 + 64, 64, 64)

        self.decoder1 = Decoder_bottleneck(64, 32, 64)

        self.logits_no_empty = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 1, kernel_size=1, padding=0))

        self.logits_final = nn.Sequential(nn.Conv2d(320+64, 64, kernel_size=3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 1, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)     #1/4
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        if not self.test:
            center_512 = self.center_global_pool(conv5)
            center_64 = self.center_conv1x1(center_512)
            center_64_flatten = center_64.view(center_64.size(0), -1)
            center_fc = self.center_fc(center_64_flatten)

        f = self.center(conv5)

        conv5 = self.dec5_1x1(conv5)
        d5 = self.decoder5(f, conv5)

        conv4 = self.dec4_1x1(conv4)
        d4 = self.decoder4(d5, conv4)

        conv3 = self.dec3_1x1(conv3)
        d3 = self.decoder3(d4, conv3)

        conv2 = self.dec2_1x1(conv2)
        d2 = self.decoder2(d3, conv2)

        d1 = self.decoder1(d2)

        hypercol = torch.cat((
            d1,
            F.upsample(d2, scale_factor=2,mode='bilinear'),
            F.upsample(d3, scale_factor=4, mode='bilinear'),
            F.upsample(d4, scale_factor=8, mode='bilinear'),
            F.upsample(d5, scale_factor=16, mode='bilinear')),1)
        hypercol = F.dropout2d(hypercol, p = 0.50)

        x_no_empty = self.logits_no_empty(hypercol)
        x_no_empty_sig = F.sigmoid(x_no_empty)

        hypercol_add_center = torch.cat((
            hypercol,
            F.upsample(center_64, scale_factor=hypercol.shape[2],mode='bilinear')),1)

        x_final = self.logits_final(hypercol_add_center)
        x_final_sig = F.sigmoid(x_final)
        if not self.test:
            return center_fc, x_no_empty, x_final
        return None, x_no_empty, x_final

class model101A_DeepSupervion(nn.Module):
    def __init__(self, num_classes=1):
        super(model101A_DeepSupervion, self).__init__()

        self.num_classes = num_classes
        num_filters = 32
        baseWidth = 4
        cardinality = 32
        self.encoder = resnext101_ibn_a(baseWidth, cardinality, pretrained = True)


        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
        self.center_conv1x1 = nn.Conv2d(2048, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, 2)

        self.center_se = SELayer(512*4)
        self.center = ImprovedIBNaDecoderBlock(512*4,  num_filters * 8)

        self.dec5_se = SELayer(512*4 + num_filters * 8)
        self.dec5 = ImprovedIBNaDecoderBlock(512*4 + num_filters * 8, num_filters * 8)

        self.dec4_se = SELayer(256*4 + num_filters * 8)
        self.dec4 = ImprovedIBNaDecoderBlock(256*4 + num_filters * 8, num_filters * 8)

        self.dec3_se = SELayer(128*4 + num_filters * 8)
        self.dec3 = ImprovedIBNaDecoderBlock(128*4 + num_filters * 8, num_filters * 4)

        self.dec2_se = SELayer(64*4 + num_filters * 4)
        self.dec2 = ImprovedIBNaDecoderBlock(64*4 + num_filters * 4, num_filters * 4)

        self.logits_no_empty = nn.Sequential(StConvRelu(num_filters * 4, num_filters, 3),
                                             nn.Dropout2d(0.5),
                                             nn.Conv2d(num_filters, 1, kernel_size=1, padding=0))


        self.logits_final = nn.Sequential(StConvRelu(num_filters * 4 + 64, num_filters, 3),
                                          nn.Dropout2d(0.5),
                                          nn.Conv2d(num_filters, 1, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)     #1/2
        conv2 = self.conv2(conv1) #1/2
        conv3 = self.conv3(conv2) #1/4
        conv4 = self.conv4(conv3) #1/8
        conv5 = self.conv5(conv4) #1/16

        center_2048 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_2048)
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten)

        center = self.center(self.center_se(self.pool(conv5)))#1/16

        dec5 = self.dec5(self.dec5_se(torch.cat([center, conv5], 1)))#1/8
        dec4 = self.dec4(self.dec4_se(torch.cat([dec5, conv4], 1)))  #1/4
        dec3 = self.dec3(self.dec3_se(torch.cat([dec4, conv3], 1)))  #1/2
        dec2 = self.dec2(self.dec2_se(torch.cat([dec3, conv2], 1)))  #1

        x_no_empty = self.logits_no_empty(dec2)
        dec0_add_center = torch.cat((
            dec2,
            F.upsample(center_64, scale_factor=128, mode='bilinear')), 1)
        x_final = self.logits_final(dec0_add_center)

        return center_fc, x_no_empty, x_final

class model101B_DeepSupervion(nn.Module):
    def __init__(self, num_classes=1):
        super(model101B_DeepSupervion, self).__init__()

        self.num_classes = num_classes

        num_filters = 32

        self.encoder = se_resnext101_32x4d()

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                   self.encoder.layer0.bn1,
                                   self.encoder.layer0.relu1)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
        self.center_conv1x1 = nn.Conv2d(2048, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, 2)

        self.center_se = SELayer(512*4)
        self.center = ImprovedIBNaDecoderBlock(512*4,  num_filters * 8)

        self.dec5_se = SELayer(512*4 + num_filters * 8)
        self.dec5 = ImprovedIBNaDecoderBlock(512*4 + num_filters * 8, num_filters * 8)

        self.dec4_se = SELayer(256*4 + num_filters * 8)
        self.dec4 = ImprovedIBNaDecoderBlock(256*4 + num_filters * 8, num_filters * 8)

        self.dec3_se = SELayer(128*4 + num_filters * 8)
        self.dec3 = ImprovedIBNaDecoderBlock(128*4 + num_filters * 8, num_filters * 4)

        self.dec2_se = SELayer(64*4 + num_filters * 4)
        self.dec2 = ImprovedIBNaDecoderBlock(64*4 + num_filters * 4, num_filters * 4)

        self.logits_no_empty = nn.Sequential(ConvRelu(num_filters * 4, num_filters, 3),
                                             nn.Dropout2d(0.5),
                                             nn.Conv2d(num_filters, 1, kernel_size=1, padding=0))


        self.logits_final = nn.Sequential(ConvRelu(num_filters * 4 + 64, num_filters, 3),
                                          nn.Dropout2d(0.5),
                                          nn.Conv2d(num_filters, 1, kernel_size=1, padding=0))


    def forward(self, x):
        conv1 = self.conv1(x)     #1/2
        conv2 = self.conv2(conv1) #1/2
        conv3 = self.conv3(conv2) #1/4
        conv4 = self.conv4(conv3) #1/8
        conv5 = self.conv5(conv4) #1/16

        center_2048 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_2048)
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten)

        center = self.center(self.center_se(self.pool(conv5)))#1/16

        dec5 = self.dec5(self.dec5_se(torch.cat([center, conv5], 1)))#1/8
        dec4 = self.dec4(self.dec4_se(torch.cat([dec5, conv4], 1)))  #1/4
        dec3 = self.dec3(self.dec3_se(torch.cat([dec4, conv3], 1)))  #1/2
        dec2 = self.dec2(self.dec2_se(torch.cat([dec3, conv2], 1)))  #1

        x_no_empty = self.logits_no_empty(dec2)

        dec0_add_center = torch.cat((
            dec2,
            F.upsample(center_64, scale_factor=128, mode='bilinear')), 1)

        x_final = self.logits_final(dec0_add_center)

        return center_fc, x_no_empty, x_final

class model152_DeepSupervion(nn.Module):
    def __init__(self, num_classes=1):
        super(model152_DeepSupervion, self).__init__()

        self.num_classes = num_classes
        self.encoder = se_resnet152()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                   self.encoder.layer0.bn1,
                                   self.encoder.layer0.relu1)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
        self.center_conv1x1 = nn.Conv2d(512*4, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, 2)

        self.center = nn.Sequential(nn.Conv2d(512*4, 512, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2,stride=2))

        self.dec5_1x1 = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=1), nn.BatchNorm2d(512),nn.ReLU(inplace=True))
        self.decoder5 = Decoder_bottleneck(256 + 512, 512, 64)

        self.dec4_1x1 = nn.Sequential(nn.Conv2d(256 * 4, 256, kernel_size=1), nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.decoder4 = Decoder_bottleneck(64 + 256, 256, 64)

        self.dec3_1x1 = nn.Sequential(nn.Conv2d(128 * 4, 128, kernel_size=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.decoder3 = Decoder_bottleneck(64 + 128, 128, 64)

        self.dec2_1x1 = nn.Sequential(nn.Conv2d(64 * 4, 64, kernel_size=1), nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.decoder2 = Decoder_bottleneck(64 + 64, 64, 64)

        self.decoder1 = Decoder_bottleneck(64, 32, 64)

        self.logits_no_empty = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 1, kernel_size=1, padding=0))

        self.logits_final = nn.Sequential(nn.Conv2d(320+64, 64, kernel_size=3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 1, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)     #1/4
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        center_512 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_512)
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten)

        f = self.center(conv5)

        conv5 = self.dec5_1x1(conv5)
        d5 = self.decoder5(f, conv5)

        conv4 = self.dec4_1x1(conv4)
        d4 = self.decoder4(d5, conv4)

        conv3 = self.dec3_1x1(conv3)
        d3 = self.decoder3(d4, conv3)

        conv2 = self.dec2_1x1(conv2)
        d2 = self.decoder2(d3, conv2)

        d1 = self.decoder1(d2)

        hypercol = torch.cat((
            d1,
            F.upsample(d2, scale_factor=2,mode='bilinear'),
            F.upsample(d3, scale_factor=4, mode='bilinear'),
            F.upsample(d4, scale_factor=8, mode='bilinear'),
            F.upsample(d5, scale_factor=16, mode='bilinear')),1)

        hypercol = F.dropout2d(hypercol, p = 0.50)

        x_no_empty = self.logits_no_empty(hypercol)

        hypercol_add_center = torch.cat((
            hypercol,
            F.upsample(center_64, scale_factor=hypercol.shape[2],mode='bilinear')),1)

        x_final = self.logits_final(hypercol_add_center)

        return center_fc, x_no_empty, x_final

class model154_DeepSupervion(nn.Module):
    def __init__(self, num_classes=1):
        super(model154_DeepSupervion, self).__init__()

        self.num_classes = num_classes
        self.encoder = senet154()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                   self.encoder.layer0.bn1,
                                   self.encoder.layer0.relu1,
                                   self.encoder.layer0.conv2,
                                   self.encoder.layer0.bn2,
                                   self.encoder.layer0.relu2,
                                   self.encoder.layer0.conv3,
                                   self.encoder.layer0.bn3,
                                   self.encoder.layer0.relu3
                                   )

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center_global_pool = nn.AdaptiveAvgPool2d([1, 1])
        self.center_conv1x1 = nn.Conv2d(512 * 4, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, 2)

        self.center = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.dec5_1x1 = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=1), nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True))
        self.decoder5 = Decoder_bottleneck(256 + 512, 512, 64)

        self.dec4_1x1 = nn.Sequential(nn.Conv2d(256 * 4, 256, kernel_size=1), nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True))
        self.decoder4 = Decoder_bottleneck(64 + 256, 256, 64)

        self.dec3_1x1 = nn.Sequential(nn.Conv2d(128 * 4, 128, kernel_size=1), nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True))
        self.decoder3 = Decoder_bottleneck(64 + 128, 128, 64)

        self.dec2_1x1 = nn.Sequential(nn.Conv2d(64 * 4, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.decoder2 = Decoder_bottleneck(64 + 64, 64, 64)

        self.decoder1 = Decoder_bottleneck(64, 32, 64)

        self.logits_no_empty = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(64, 1, kernel_size=1, padding=0))

        self.logits_final = nn.Sequential(nn.Conv2d(320 + 64, 64, kernel_size=3, padding=1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(64, 1, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)  # 1/4
        conv2 = self.conv2(conv1)  # 1/4
        conv3 = self.conv3(conv2)  # 1/8
        conv4 = self.conv4(conv3)  # 1/16
        conv5 = self.conv5(conv4)  # 1/32

        center_512 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_512)
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten)

        f = self.center(conv5)

        conv5 = self.dec5_1x1(conv5)
        d5 = self.decoder5(f, conv5)

        conv4 = self.dec4_1x1(conv4)
        d4 = self.decoder4(d5, conv4)

        conv3 = self.dec3_1x1(conv3)
        d3 = self.decoder3(d4, conv3)

        conv2 = self.dec2_1x1(conv2)
        d2 = self.decoder2(d3, conv2)

        d1 = self.decoder1(d2)

        hypercol = torch.cat((
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear'),
            F.upsample(d3, scale_factor=4, mode='bilinear'),
            F.upsample(d4, scale_factor=8, mode='bilinear'),
            F.upsample(d5, scale_factor=16, mode='bilinear')), 1)

        hypercol = F.dropout2d(hypercol, p=0.50)
        x_no_empty = self.logits_no_empty(hypercol)

        hypercol_add_center = torch.cat((
            hypercol,
            F.upsample(center_64, scale_factor=hypercol.shape[2], mode='bilinear')), 1)

        x_final = self.logits_final(hypercol_add_center)

        return center_fc, x_no_empty,  x_final