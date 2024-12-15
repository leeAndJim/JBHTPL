# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:57:01 2017

@author: Biagio Brattoli
"""
import torch
import torch.nn as nn
from torch import cat
from torchsummary import summary

import sys

sys.path.append('Utils')
# from Layers import LRN


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

        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1', nn.Linear(1152, 64))
        self.fc6.add_module('relu6_s1', nn.ReLU(inplace=True))
        self.fc6.add_module('BN6_s1', nn.BatchNorm1d(64))

        self.fc7 = nn.Sequential()
        self.fc7.add_module('fc7', nn.Linear(64, 32))
        self.fc7.add_module('relu7', nn.ReLU(inplace=True))
        self.fc7.add_module('BN7', nn.BatchNorm1d(32))
        self.fc7.add_module('drop7', nn.Dropout(p=0.3))

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc8', nn.Linear(32, classes))
        # self.classifier.add_module('softmax1', nn.Softmax(dim=1))

        # self.apply(weights_init)

    def load(self, checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict and 'fc8' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])

    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def forward(self, x):

        B, T, C, H, W = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(9):
            # z = self.conv(x[i,:,0:1,:,:])
            z = self.layer1(x[i,:,0:1,:,:])
            z = self.layer2(z)
            z = self.layer3(z)
            indentity = self.downsample(x[i,:,0:1,:,:])
            z += indentity
            # z = self.fc6(z.view(B, -1))
            z = z.view([B, 1, -1])
            x_list.append(z)

        x = cat(x_list, 1)
        x = self.fc6(x.view(B, -1))
        x = self.fc7(x)
        x = self.classifier(x)

        return x


def weights_init(model):
    if type(model) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_normal(model.weight.data)
        nn.init.constant(model.bias.data, 0.1)

if __name__ == "__main__":
    model = Network(5)
    model.forward(torch.ones((128, 9, 3, 32, 32)))
    # input_size = (3, 32, 32)
    model = Network(5).to('cuda')

    input_size = (9, 3, 32, 32)

    summary(model, input_size=input_size)



