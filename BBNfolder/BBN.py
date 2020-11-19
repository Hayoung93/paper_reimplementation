import torch
import torch.nn as nn


class Resblock_basic(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Resblock_basic, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class Resblock_bottle(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Resblock_bottle, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        mid_channels = int(out_channels / 4)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity

        return self.relu(out)


class block_resnet_shared(nn.Module):
    def __init__(self, in_channels, block=Resblock_basic):
        super(block_resnet_shared, self).__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
        self.layer1 = nn.Sequential(
            block(64, 64, 1),
            block(64, 64, 1)
        )
        self.layer2 = nn.Sequential(
            block(64, 128, 2),
            block(128, 128, 1)
        )
        self.layer3 = nn.Sequential(
            block(128, 256, 2),
            block(256, 256, 1),
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class block_resnet_separated(nn.Module):
    def __init__(self, in_channels, block=Resblock_basic):
        super(block_resnet_separated, self).__init__()
        self.layer = nn.Sequential(
            block(in_channels, in_channels * 2, 2),
            block(in_channels * 2, in_channels * 2, 1)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.layer(x)
        x = self.gap(x)

        return x


class classifier(nn.Module):
    def __init__(self, in_channels, num_classes, batchnorm=True):
        super(classifier, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_channels, in_channels)
        self.fc2 = nn.Linear(in_channels, num_classes)
        if batchnorm:
            self.bn = nn.BatchNorm1d(in_channels)
        else:
            self.bn = None

    def forward(self, x):
        x = self.relu(self.fc1(x))
        if self.bn is not None:
            x = self.bn(x)
        x = self.fc2(x)

        return x
