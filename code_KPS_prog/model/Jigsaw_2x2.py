import torch
import torch.nn as nn
from torch import cat

import sys

class Network(nn.Module):

    def __init__(self, classes=1000):
        super(Network, self).__init__()

        self.conv1_s1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1_s1 = nn.ReLU(inplace=True)
        self.BN1_s1 = nn.BatchNorm2d(32)
        self.pool1_s1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.drop1_s1 = nn.Dropout2d(0.3)

        self.conv2_s1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.relu2_s1 = nn.ReLU(inplace=True)
        self.BN2_s1 = nn.BatchNorm2d(16)
        self.pool2_s1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.drop2_s1 = nn.Dropout2d(0.3)

        self.conv3_s1 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.relu3_s1 = nn.ReLU(inplace=True)
        self.BN3_s1 = nn.BatchNorm2d(8)
        self.pool3_s1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.drop3_s1 = nn.Dropout2d(0.3)

        self.conv4_s1 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.relu4_s1 = nn.ReLU(inplace=True)
        self.BN4_s1 = nn.BatchNorm2d(8)
        self.pool4_s1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.drop4_s1 = nn.Dropout2d(0.3)

        self.downsample1 = nn.Conv2d(1, 32, kernel_size=1)
        self.downsample2 = nn.Conv2d(32, 16, kernel_size=1)
        self.downsample3 = nn.Conv2d(16, 8, kernel_size=1)
        self.downsample4 = nn.Conv2d(8, 8, kernel_size=1)

        self.last_linear = nn.Sequential()
        self.last_linear.add_module('fc6', nn.Linear(1568, 64))
        self.last_linear.add_module('tanh6', nn.Tanh())
        self.last_linear.add_module('BN6', nn.BatchNorm1d(64))
        
        self.last_linear.add_module('fc7', nn.Linear(64, 32))
        self.last_linear.add_module('tnh7', nn.Tanh())
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

    def forward(self, x):

        B, C, H, W = x.size()

        z1 = self.downsample1(x[:,:1,:,:])
        z = self.conv1_s1(x[:,:1,:,:])
        z = z + z1
        z = self.relu1_s1(z)
        z = self.BN1_s1(z)
        z = self.pool1_s1(z)
        z = self.drop1_s1(z)

        z2 = self.downsample2(z)
        z = self.conv2_s1(z)
        z = z + z2
        z = self.relu2_s1(z)
        z = self.BN2_s1(z)
        z = self.pool2_s1(z)
        z = self.drop2_s1(z)

        z3 = self.downsample3(z)
        z = self.conv3_s1(z)
        z = z + z3
        z = self.relu3_s1(z)
        z = self.BN3_s1(z)
        z = self.pool3_s1(z)
        z = self.drop3_s1(z)

        z4 = self.downsample4(z)
        z = self.conv4_s1(z)
        z = z + z4
        z = self.relu4_s1(z)
        z = self.BN4_s1(z)
        z = self.pool4_s1(z)
        z = self.drop4_s1(z)
        x = self.last_linear(z.view(B, -1))

        return x


def weights_init(model):
    if type(model) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_normal(model.weight.data)
        nn.init.constant(model.bias.data, 0.1)