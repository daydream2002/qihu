#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/17 9:09
# @Author  : Joisen
# @File    : pong_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Basic_Layer(nn.Module):
    def __init__(self):
        super(Basic_Layer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer(x) + x
        return out


class Model_Me(nn.Module):
    def __init__(self, Basic_Layer, out_channel, nums):
        super(Model_Me, self).__init__()
        self.conv1 = nn.Conv2d(418, 256, 3, 1, 1)
        self.relu = nn.ReLU()
        self.conv2 = self._make_layer(Basic_Layer, nums)
        self.conv3 = nn.Conv2d(256, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 64, 3, 1, 1)
        self.fc = nn.Linear(64 * 34, out_channel)

    def _make_layer(self, basic, nums):
        layers = []
        for i in range(nums):
            layers.append(basic())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1, 1], padding=1) -> None:
        super(BasicBlock, self).__init__()
        # 残差部分
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # 原地替换 节省内存开销
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if stride[0] != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                # 注意跳变时 都是stride==2的时候 也就是每次输出信道升维的时候
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=2) -> None:
        super(ResNet, self).__init__()
        self.in_channels = 256
        # 第一层作为单独的 因为没有残差快
        self.conv1 = nn.Sequential(
            nn.Conv2d(418, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            # nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        # conv2_x
        self.conv2 = self._make_layer(BasicBlock, 256, [[1, 1], [1, 1]])

        # conv3_x
        self.conv3 = self._make_layer(BasicBlock, 256, [[1, 1], [1, 1]])

        # conv4_x
        self.conv4 = self._make_layer(BasicBlock, 256, [[1, 1], [1, 1]])

        # conv5_x
        self.conv5 = self._make_layer(BasicBlock, 256, [[1, 1], [1, 1]])

        self.conv = nn.Conv2d(256, 128, 3, 1, 1)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4352, num_classes)

    # 这个函数主要是用来，重复同一个残差块
    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        # out = self.avgpool(out)
        out = self.conv(out)
        # print(out.shape)
        out = out.reshape(x.shape[0], -1)
        # print(out.shape)
        out = self.fc(out)
        return out





if __name__ == '__main__':
    model = Model_Me(Basic_Layer, 2, 20)
    input = torch.randn(1, 418, 34, 1)
    print(model(input))
