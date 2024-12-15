import torch
import torch.nn as nn
from torch import cat
from torch.nn import Parameter

from torch.nn import functional as F
import sys
from model.Resnet_CCBC import *


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        cosine = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        #out = x.mm(self.weight)
        return cosine
    
class Network(nn.Module):

    def __init__(self, classes=1000, feature_size=512):
        super(Network, self).__init__()

        self.conv = resnet18(pretrained=True)

        # self.max1 = nn.MaxPool2d(kernel_size=14, stride=14)
        # self.max2 = nn.MaxPool2d(kernel_size=14, stride=7)
        # self.max3 = nn.MaxPool2d(kernel_size=7, stride=7)
        self.max1 = nn.AdaptiveAvgPool2d((2, 2))
        self.max2 = nn.AdaptiveAvgPool2d((1, 2))
        self.max3 = nn.AdaptiveAvgPool2d((1, 1))

        self.num_ftrs = 512

        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        # self.classifier3 = nn.Sequential(
        #     # nn.Linear(512, 64),
        #     # nn.Tanh(),
        #     nn.BatchNorm1d(512),
        #     nn.Linear(512, 32),
        #     nn.Tanh(),
        #     nn.BatchNorm1d(32),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(32, classes),
        #     # NormedLinear(32, classes),
        # )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, classes)
        )
        
        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs//2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, classes)
        )

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs//4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, classes)
        )

        self.fc_training = False


    def load(self, checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict and 'fc8' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])

    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def forward(self, x, block=[0,0,0,0]):

        B, C, H, W = x.size()
        x1, x2, x3, x4, x5 = self.conv(x, block=block)

        if self.fc_training:
            return x3, x4, x5
        
        xs1 = x3 # self.conv_block1(x3)
        xs2 = x4 # self.conv_block2(x4)
        xs3 = x5 # self.conv_block3(x5)
        # print(xs1.size(), xs2.size(), xs3.size())
        xl1 = self.max1(xs1)
        xl1 = xl1.view(xl1.size(0), -1)
        xc1 = self.classifier1(xl1)

        xl2 = self.max2(xs2)
        xl2 = xl2.view(xl2.size(0), -1)
        xc2 = self.classifier2(xl2)

        xl3 = self.max3(xs3)
        xl3 = xl3.view(xl3.size(0), -1)
        xc3 = self.classifier3(xl3)
        if self.training:
            return xc1, xc2, xc3
        else:
            return (xc1+xc2+xc3) / 3


def weights_init(model):
    if type(model) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_normal(model.weight.data)
        nn.init.constant(model.bias.data, 0.1)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

