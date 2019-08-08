# https://raw.githubusercontent.com/liaopeiyuan/ml-arsenal-public/master/models/TGS_salt/se_resnet34.py
import math

import torch
import torch.nn.functional as F
from torch import nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

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
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, input_channel=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def se_resnet18(num_classes, input_channel):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, input_channel=input_channel)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet34(num_classes, input_channel):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes, input_channel=input_channel)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes, input_channel):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes, input_channel=input_channel)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet101(num_classes, input_channel):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes, input_channel=input_channel)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet152(num_classes, input_channel):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes, input_channel=input_channel)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        # self.bn = SynchronizedBatchNorm2d(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x


class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()
        self.conv = ConvBn2d(in_channels=out_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        # print('spatial',x.size())
        x = F.sigmoid(x)
        return x


class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()
        self.conv1 = ConvBn2d(in_channels=out_channels, out_channels=int(out_channels / 2), kernel_size=1, padding=0)
        self.conv2 = ConvBn2d(in_channels=int(out_channels / 2), out_channels=out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = nn.AvgPool2d(x.size()[2:])(x)
        # print('channel',x.size())
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = ConvBn2d(in_channels, channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
        self.spatial_gate = sSE(out_channels)
        self.channel_gate = cSE(out_channels)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        # print('x',x.size())
        # print('e',e.size())
        if e is not None:
            x = torch.cat([x, e], 1)

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        # print('x_new',x.size())
        g1 = self.spatial_gate(x)
        # print('g1',g1.size())
        g2 = self.channel_gate(x)
        # print('g2',g2.size())
        x = g1 * x + g2 * x

        return x


class SEResUNetscSEHyper34(nn.Module):
    def __init__(self, num_classes=1):
        super(SEResUNetscSEHyper34, self).__init__()
        self.resnet = se_resnet34(num_classes=num_classes, input_channel=num_classes)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )

        self.encoder2 = self.resnet.layer1  # 64
        self.encoder3 = self.resnet.layer2  # 128
        self.encoder4 = self.resnet.layer3  # 256
        self.encoder5 = self.resnet.layer4  # 512

        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder5 = Decoder(256 + 512, 512, 64)
        self.decoder4 = Decoder(64 + 256, 256, 64)
        self.decoder3 = Decoder(64 + 128, 128, 64)
        self.decoder2 = Decoder(64 + 64, 64, 64)
        self.decoder1 = Decoder(64, 32, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(384, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )
        self.center_fc = nn.Linear(256, num_classes)

    def forward(self, x, flip):
        e1 = self.conv1(x)
        # print(e1.size())
        e2 = self.encoder2(e1)
        # print('e2',e2.size())
        e3 = self.encoder3(e2)
        # print('e3',e3.size())
        e4 = self.encoder4(e3)
        # print('e4',e4.size())
        e5 = self.encoder5(e4)
        # print('e5',e5.size())

        f = self.center(e5)

        print(f.shape)
        classification = self.center_fc(F.max_pool2d(f, kernel_size=8).view(f.size(0), -1))

        # print('f',f.size())
        d5 = self.decoder5(f, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2)
        # print('d1',d1.size())

        # Koke_Cacao: Change dropout from all 0.5 to 0.2 and 0.8
        f = torch.cat((
            F.dropout2d(F.upsample(e1, scale_factor=2, mode='bilinear', align_corners=False), 0.8),
            F.dropout2d(d1, 0.2),
            F.dropout2d(F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False), 0.8),
            F.dropout2d(F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False), 0.8),
            F.dropout2d(F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False), 0.8),
            F.dropout2d(F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False), 0.8),
        ), 1)

        logit = self.logit(f)
        return classification, None, logit


class ResUNetscSEHyper32(nn.Module):
    def __init__(self, num_classes=1):
        super(ResUNetscSEHyper32, self).__init__()
        self.resnet = ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes, input_channel=num_classes)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            nn.ReLU(inplace=True),
        )

        self.encoder2 = self.resnet.layer1  # 16
        self.encoder3 = self.resnet.layer2  # 32
        self.encoder4 = self.resnet.layer3  # 64
        # self.encoder5 = self.resnet.layer4 #512

        self.center = nn.Sequential(
            ConvBn2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # self.decoder5 = Decoder(256+512,512,64)
        self.decoder4 = Decoder(64 + 64, 64, 64)
        self.decoder3 = Decoder(64 + 32, 32, 64)
        self.decoder2 = Decoder(64 + 16, 16, 64)
        self.decoder1 = Decoder(64 + 16, 8, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(272, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )
        self.center_fc = nn.Linear(256, num_classes)

    def forward(self, x, flip):
        e1 = self.conv1(x)
        # print("e1",e1.size())
        e2 = self.encoder2(e1)
        # print('e2',e2.size())
        e3 = self.encoder3(e2)
        # print('e3',e3.size())
        e4 = self.encoder4(e3)
        # print('e4',e4.size())
        # e5 = self.encoder5(e4)
        # print('e5',e5.size())

        f = self.center(e4)

        print(f.shape)
        classification = self.center_fc(F.max_pool2d(f, kernel_size=8).view(f.size(0), -1))

        # print('f',f.size())
        # d5 = self.decoder5(f, e5)
        d4 = self.decoder4(f, e4)
        # print('d4',d4.size())
        d3 = self.decoder3(d4, e3)
        # print('d3',d3.size())
        d2 = self.decoder2(d3, e2)
        # print('d2',d4.size())
        d1 = self.decoder1(d2, e1)
        # print('d1',d1.size())
        # print('e1',e1.size())
        # print('d2',d2.size())
        # print('d3',d3.size())
        # print('d4',d4.size())
        # print('d1',d1.size())

        # Koke_Cacao: Change dropout from all 0.5 to 0.2 and 0.8
        f = torch.cat((
            F.dropout2d(F.upsample(e1, scale_factor=1, mode='bilinear', align_corners=False), 0.8),
            F.dropout2d(d1, 0.2),
            F.dropout2d(F.upsample(d2, scale_factor=1, mode='bilinear', align_corners=False), 0.8),
            F.dropout2d(F.upsample(d3, scale_factor=2, mode='bilinear', align_corners=False), 0.8),
            F.dropout2d(F.upsample(d4, scale_factor=4, mode='bilinear', align_corners=False), 0.8),
            # F.upsample(d5,scale_factor=16, mode='bilinear',align_corners=False),
        ), 1)
        # print('f',f.size())
        logit = self.logit(f)
        # print('logit',logit.size())
        return classification, None, logit
