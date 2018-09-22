import torchvision
from torch import nn

from resnet import BasicBlock
from resnet.resnet_model import _weights_init
import torch.nn.functional as F


class ResnetExtractor(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResnetExtractor, self).__init__()
        self.in_planes = 16
        # transform input
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out

def resnet20():
    return ResnetExtractor(BasicBlock, [3, 3, 3])
def resnet32():
    return ResnetExtractor(BasicBlock, [5, 5, 5])
def resnet44():
    return ResnetExtractor(BasicBlock, [7, 7, 7])
def resnet56():
    return ResnetExtractor(BasicBlock, [9, 9, 9])
def resnet110():
    return ResnetExtractor(BasicBlock, [18, 18, 18])
def resnet1202():
    return ResnetExtractor(BasicBlock, [200, 200, 200])