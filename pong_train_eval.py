#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/17 9:08
# @Author  : Joisen
# @File    : pong_train_eval.py

from torch.optim import Adam
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.models as models
from dataset import *
from pong_model import *

if __name__ == '__main__':
    writer = SummaryWriter('log')
    # 设置启用设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据路径
    hu_dir = '/home/tonnn/.nas/wjh/qihu/hu/hu'
    nhu_dir = '/home/tonnn/.nas/wjh/qihu/hu/nhu'

    # 加载数据集
    hu_dataset = HuDataset(hu_dir)
    nhu_dataset = HuDataset(nhu_dir)
    print(len(hu_dataset))
    print(len(nhu_dataset))


    # 设置划分比例
    train_ratio = 0.8

    # 计算训练和验证集的大小
    hu_train_size = int(train_ratio * len(hu_dataset))
    hu_val_size = len(hu_dataset) - hu_train_size
    nhu_train_size = int(train_ratio * len(nhu_dataset))
    nhu_val_size = len(nhu_dataset) - nhu_train_size

    # 划分数据集
    hu_train_dataset, hu_val_dataset = random_split(hu_dataset, [hu_train_size, hu_val_size])
    nhu_train_dataset, nhu_val_dataset = random_split(nhu_dataset, [nhu_train_size, nhu_val_size])

    # 合并训练集和验证集
    train_dataset = hu_train_dataset + nhu_train_dataset
    val_dataset = hu_val_dataset + nhu_val_dataset

    # 打印数据集大小
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # 设置batch_size
    batch_size = 128

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # # 设置batch_size
    # batch_size = 128
    # # 加载数据集
    # dataset = HuDataset(data_dir)
    #
    # # 划分数据集
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
    # # 创建 DataLoader
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
    #                           num_workers=2)  # train需要shuffle
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    # 定义网络
    # 定义模型并添加Dropout层
    class CustomResNet(nn.Module):
        def __init__(self):
            super(CustomResNet, self).__init__()
            self.model = models.resnet34(pretrained=True)
            self.model.conv1 = nn.Conv2d(418, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 2)
            self.dropout = nn.Dropout(0.5)  # 添加Dropout层

        def forward(self, x):
            x = self.model(x)
            x = self.dropout(x)
            return x


    model = CustomResNet()
    # 定义损失函数
    loss = nn.CrossEntropyLoss()

    # 设置学习率
    LR = 0.0008

    # 定义优化器(随机梯度下降)
    optim = Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # 训练轮次
    train_round = 30

    # 将loss和神经网络放到指定设备上执行
    loss.to(device)
    model.to(device)
    model_dir = "./model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    for i in range(train_round):
        # 开始训练
        loss_sum = 0
        model.train()
        print(f"------训练第{i + 1}开始------")
        print('训练数据长度：', len(train_dataset))
        for feature, label in tqdm(train_loader):
            # 初始化梯度
            optim.zero_grad()
            # 将数据放到指定设备上运算
            inputs = feature.to(device)
            targets = label.to(device)
            # 进行训练
            outputs = model(inputs)
            # 计算损失，并计算梯度
            train_loss = loss(outputs, targets)
            train_loss.backward()
            # 更新参数
            optim.step()

            # 计算总损失
            loss_sum += train_loss.item()
        print(f"训练损失为:{loss_sum}")
        writer.add_scalar('sum loss', loss_sum, i)
        # 开始验证
        print(f"----开始第{i + 1}次验证----")
        model.eval()
        sum_data = 0  # 记录测试样本数
        acc = 0  # 记录准确率
        with torch.no_grad():
            print('验证数据长度：', len(val_dataset))
            for feature, label in tqdm(val_loader):
                # 将数据放到指定设备上运算
                inputs = feature.to(device)
                targets = label.to(device)
                # 预测
                outputs = model(inputs)
                # 计算测试样本数
                sum_data += outputs.shape[0]
                # 计算正确数
                acc = (outputs.argmax(1) == targets).sum() + acc
            acc = acc / sum_data
            print(f"准确率为:{acc}")
            writer.add_scalar('Accuracy', acc, i)
        torch.save(model, f"./model/model{i}.pth")
    writer.close()
    print("")
# tensorboard --logdir=D:\product\Python\project\RL-Group\works\pong\log
