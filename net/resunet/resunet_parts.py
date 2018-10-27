# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


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
