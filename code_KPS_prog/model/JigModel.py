# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:57:01 2017

@author: Biagio Brattoli
"""
import torch
import torch.nn as nn
from torch import cat

import sys

sys.path.append('Utils')


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


class Network(nn.Module):

    def __init__(self, classes=1000):
        super(Network, self).__init__()

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv1_s1', nn.Conv2d(1, 32, kernel_size=3, padding=1))
        self.layer1.add_module('relu1_s1', nn.ReLU(inplace=True))
        self.layer1.add_module('BN1_s1', nn.BatchNorm2d(32))
        self.layer1.add_module('pool1_s1', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.layer1.add_module('drop1_s1', nn.Dropout2d(0.3))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv2_s1', nn.Conv2d(32, 16, kernel_size=3, padding=1))
        self.layer2.add_module('relu2_s1', nn.ReLU(inplace=True))
        self.layer2.add_module('BN2_s1', nn.BatchNorm2d(16))
        self.layer2.add_module('pool2_s1', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.layer2.add_module('drop2_s1', nn.Dropout(0.3))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv3_s1', nn.Conv2d(16, 8, kernel_size=3, padding=1))
        self.layer3.add_module('relu3_s1', nn.ReLU(inplace=True))
        self.layer3.add_module('BN3_s1', nn.BatchNorm2d(8))
        self.layer3.add_module('pool3_s1', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.layer3.add_module('drop3_s1', nn.Dropout2d(0.3))

        self.downsample = nn.Sequential()
        self.downsample.add_module('conv1_d1', nn.Conv2d(1, 8, kernel_size=4, stride=8, bias=False))
        self.downsample.add_module('BN1_d1', nn.BatchNorm2d(8))

        # self.fc6 = nn.Sequential()
        # self.fc6.add_module('fc6_s1', nn.Linear(200, 64))
        # self.fc6.add_module('relu6_s1', nn.ReLU(inplace=True))
        # self.fc6.add_module('BN6_s1', nn.BatchNorm1d(64))
        # 32 48 48
        # 16 24 24
        # 8 12 12 
        self.max1 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.max2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.max3 = nn.MaxPool2d(kernel_size=1, stride=1)

        self.conv1 = nn.Sequential(
            BasicConv(32, 8, kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(8, 8, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(1152, 64), # 1152
            nn.Tanh(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.3),
            nn.Linear(32, classes)
        )

        self.conv2 = nn.Sequential(
            BasicConv(16, 8, kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(8, 8, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(1024, 64), # 1024
            nn.Tanh(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.3),
            nn.Linear(32, classes)
        )

        self.conv3 = nn.Sequential(
            BasicConv(8, 8, kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(8, 8, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(1152, 64), # 1152
            nn.Tanh(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.3),
            nn.Linear(32, classes)
        )

        self.last_linear = nn.Sequential()
        self.last_linear.add_module('fc6', nn.Linear(1152, 64))
        self.last_linear.add_module('relu6', nn.Tanh())
        self.last_linear.add_module('BN6', nn.BatchNorm1d(64))

        self.last_linear.add_module('fc7', nn.Linear(64, 32))
        self.last_linear.add_module('relu7', nn.Tanh())
        self.last_linear.add_module('BN7', nn.BatchNorm1d(32))
        self.last_linear.add_module('drop7', nn.Dropout(p=0.3))

        self.last_linear.add_module('fc8', nn.Linear(32, classes))

        self.training = True
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

    def split_block(self, x, div_parts):
        if div_parts == 0:
            return x
        n, c, w, h = x.size()
        block_size = w // div_parts
        l = []
        for i in range(div_parts):
            for j in range(div_parts):
                l.append(x[:, :, i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size])
        x = torch.cat(l, 0)
        return x
    
    def concat_block(self, x, div_parts):
        if div_parts == 0:
            return x
        n, c, w, h = x.size()
        n = n // div_parts ** 2
        r = []
        for i in range(div_parts):
            c = []
            for j in range(div_parts):
                c.append(x[(i * div_parts + j) * n: (i * div_parts + (j + 1)) * n])
            c = torch.cat(c, -1)
            r.append(c)
        x = torch.cat(r, -2)
        return x
    
    def get_dis_block(self, x):
        x = self.block_wise_pool(x)
        f = self.maxpool2(x) * 1

        f_min = torch.min(f, 1)[0].unsqueeze(1)
        f_max = torch.max(f, 1)[0].unsqueeze(1)
        f = (f - f_min) / (f_max - f_min + 1e-9)

        return f

    def forward(self, x, block=[0, 0, 0]):
        B, C, H, W = x.size()
        # x = x.transpose(0, 1)
        # print(x.size())
        # sys.exit(100)
        # x_list = []
        # for i in range(9):
        #     z = self.conv(x[i,:,0:1,:,:])
        #     # z = self.fc6(z.view(B, -1))
        #     z = z.view([B, 1, -1])
        #     x_list.append(z)
        if self.training:
            z = self.split_block(x[:,:1,:,:], block[0])
            z = self.layer1(z)
            x1 = self.concat_block(z, block[0])
            
            z = self.split_block(x1, block[1])
            z = self.layer2(z)
            x2 = self.concat_block(z, block[1])

            z = self.split_block(x2, block[2])
            z = self.layer3(z)
            x3 = self.concat_block(z, block[2])
        else:
            x1 = self.layer1(x[:,:1,:,:])
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
        
        if self.fc_training:
            return x1, x2, x3

        xs1 = x1 # self.conv1(x1)
        xs2 = x2 # self.conv2(x2)
        xs3 = x3 # self.conv3(x3)

        xl1 = self.max1(xs1)
        xl2 = self.max2(xs2)
        xl3 = self.max3(xs3)

        xc1 = self.classifier1(xl1.view(xl1.size(0), -1))
        xc2 = self.classifier2(xl2.view(xl2.size(0), -1))
        xc3 = self.classifier3(xl3.view(xl3.size(0), -1))
        if self.training:
            return xc1, xc2, xc3
        else:
            return (xc1 + xc2 + xc3) / 3


def weights_init(model):
    if type(model) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_normal(model.weight.data)
        nn.init.constant(model.bias.data, 0.1)

if __name__ == "__main__":
    model = Network(11)
    # pretrained_weight = torch.load("D:\研究生\Data\Sonar\marine-debris-fls-datasets\md_fls_dataset\model.pt")
    #
    # partial_weights = {}
    # for key, value in pretrained_weight.items():
    #     if key.startswith('conv'):
    #        partial_weights[key] = value
    #
    # model.load_state_dict(partial_weights, strict=False)

    print(model)