import torch
import torch.nn as nn


# Bottleneck模块（与原论文一致）
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 自定义ResNet50（无任何池化层）支持输入尺寸 [1, 85, 4, 9]
class ResNet50NoPool(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50NoPool, self).__init__()
        self.inplanes = 64
        # 修改初始卷积层：输入通道数改为85
        self.conv1 = nn.Conv2d(86, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 删除maxpool层

        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # 删除全局平均池化层
        # 根据计算，最终特征图尺寸为 (2048, 1, 1)
        self.fc = nn.Linear(2048 * 1 * 1, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 如果下采样需要或通道数不匹配，则构建downsample模块
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入尺寸: [1, 85, 4, 9]
        x = self.conv1(x)  # 输出尺寸: [1, 64, 2, 5]
        x = self.bn1(x)
        x = self.relu(x)
        # 不使用maxpool层

        x = self.layer1(x)  # 输出尺寸: [1, 256, 2, 5]
        x = self.layer2(x)  # 输出尺寸: [1, 512, 1, 3]
        x = self.layer3(x)  # 输出尺寸: [1, 1024, 1, 2]
        x = self.layer4(x)  # 输出尺寸: [1, 2048, 1, 1]

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# 示例：创建网络并测试前向传播
# if __name__ == "__main__":
#     model = ResNet50NoPool(num_classes=2)  # 例如：10个类别
#     print(model)
#     # 模拟一个batch大小为1的输入，尺寸为 [1, 85, 4, 9]
#     x = torch.randn(2, 85, 4, 9)
#     out = model(x)
#     print("输出尺寸：", out.shape)
