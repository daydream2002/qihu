#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/25 8:50
# @Author  : Joisen
# @File    : densenet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1)

        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()

        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        # print(out.shape)
        # out = F.avg_pool2d(out, 2)
        # print(out.shape)
        return out


class DenseNet(nn.Module):
    def __init__(self, nblocks, growth_rate, reduction, num_classes):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 128
        self.basic_conv = nn.Sequential(
            nn.Conv2d(85, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_planes),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.dense1 = self._make_dense_layers(num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.linear = nn.Linear(num_planes, num_classes)


        self.linear = nn.Linear(37152, num_classes)

    def _make_dense_layers(self, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(Bottleneck(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.basic_conv(x)
        # print(out.shape)
        out = self.trans1(self.dense1(out))
        # print(out.shape)
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        # print(out.shape)
        # out = self.gap(out)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.linear(out)
        return out


def DenseNet121():
    return DenseNet([6, 12, 24, 16], growth_rate=32, reduction=0.5, num_classes=2)


def DenseNet4():
    return DenseNet([6,6,6,6], growth_rate=16, reduction=0.5,num_classes=2)




if __name__ == '__main__':
    input = torch.randn(1, 69, 4, 9)
    net121 = DenseNet121()
    net4 = DenseNet4()
    y = net121(input)
    print(y)
