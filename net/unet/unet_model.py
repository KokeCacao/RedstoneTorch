# full assembly of the sub-parts to form the complete net

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes):
#         super(UNet, self).__init__()
#         self.inc = inconv(n_channels, 64)
#         self.down1 = down(64, 128)
#         self.down2 = down(128, 256)
#         self.down3 = down(256, 512)
#         # add depth data here
#         self.join = join()
#         self.down4 = down(512, 512)
#         self.up1 = up(1024, 256)
#         self.up2 = up(512, 128)
#         self.up3 = up(256, 64)
#         self.up4 = up(128, 64)
#         self.outc = outconv(64, n_classes)
#
#     def forward(self, x, depth):
#         # print("x size = {}".format(x.size())) [32, 3, 224, 224]
#         x1 = self.inc(x)
#         # print("x1 size = {}".format(x1.size())) [32, 64, 224, 224]
#         x2 = self.down1(x1)
#         # print("x2 size = {}".format(x2.size())) [32, 128, 112, 112]
#         x3 = self.down2(x2)
#         # print("x3 size = {}".format(x3.size())) [32, 256, 56, 56]
#         x4 = self.down3(x3)
#         # print("x4 size = {}".format(x4.size())) [32, 512, 28, 28]
#
#         # add depth data here
#         join = self.join(x4, depth)
#         # print("join size = {}".format(join.size())) [32, 512, 28, 28]
#
#         x5 = self.down4(join)
#         # print("x5 size = {}".format(x5.size())) [32, 512, 14, 14]
#         x = self.up1(x5, x4)
#         # print("after up1 size = {}".format(x.size())) [32, 256, 28, 28]
#         x = self.up2(x, x3)
#         # print("after up2 size = {}".format(x.size())) [32, 128, 56, 56]
#         x = self.up3(x, x2)
#         # print("after up3 size = {}".format(x.size())) [32, 64, 112, 112]
#         x = self.up4(x, x1)
#         # print("after up4 size = {}".format(x.size())) [32, 64, 224, 224]
#         x = self.outc(x)
#         # print("outc size = {}".format(x.size())) [32, 1, 224, 224]
#         return x