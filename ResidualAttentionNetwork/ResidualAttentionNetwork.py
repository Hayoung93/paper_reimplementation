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


class Attention_module(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=(1, 1, 1, 2, 1, 1, 1), block=Resblock_basic):
        super(Attention_module, self).__init__()
        self.first_block = block(in_channels, out_channels, stride=1)

        self.trunk_branch = nn.Sequential(
            block(out_channels, out_channels, 1),
            block(out_channels, out_channels, 1)
        )

        self.mpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = []
        layers_mid = []
        for _ in range(blocks[0]):
            layers_mid.append(block(out_channels, out_channels, 1))
        layers.append(nn.Sequential(*layers_mid))
        layers_mid = []
        for b in blocks[1:]:
            for _ in range(b):
                layers_mid.append(block(out_channels, out_channels, 1))
            layers.append(nn.Sequential(*layers_mid))
            layers_mid = []
        conv_blocks = nn.Sequential(*layers)
        self.conv_blocks_down = conv_blocks[:int(len(blocks) / 2)]
        self.conv_blocks_mid = conv_blocks[int(len(blocks) / 2)]
        self.conv_blocks_up = conv_blocks[-int(len(blocks) / 2):]

        layers = []
        for i in range(int(len(blocks) / 2)):
            layers.append(block(out_channels, out_channels, 1))
        self.conv_blocks_skip = nn.Sequential(*layers)

        self.last_soft_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

        self.last_block = block(out_channels, out_channels, stride=1)

    def forward(self, x):
        x = self.first_block(x)

        out_trunk = self.trunk_branch(x)

        # --- soft mask branch ---
        soft_out = self.mpool(x)
        soft_outs = []
        # down path
        for down_layer in self.conv_blocks_down:
            soft_out = down_layer(soft_out)
            soft_outs.append(soft_out)
            soft_out = self.mpool(soft_out)
        # mid path
        for mid_layer in self.conv_blocks_mid:
            soft_out = mid_layer(soft_out)
        # up path
        for i, (skip_layer, up_layer) in enumerate(zip(self.conv_blocks_skip, self.conv_blocks_up)):
            skip_out = skip_layer(soft_outs[len(soft_outs) - 1 - i])
            up_out = nn.functional.interpolate(soft_out, size=skip_out.size()[-2:], mode='bilinear', align_corners=True)
            soft_out = up_layer((skip_out + up_out))

        soft_out = nn.functional.interpolate(soft_out, size=out_trunk.size()[-2:], mode='bilinear', align_corners=True)
        soft_out = self.last_soft_block(soft_out)
        # --- soft mask branch ended ---

        out = soft_out * out_trunk + out_trunk

        return self.last_block(out)


class ResidualAttentionNetwork(nn.Module):
    def __init__(self, in_channels, num_classes=7, block=Resblock_basic, blocks=(1, 1, 4),
                 channels=(64, 128, 256, 512), am_per_block=(1, 1, 1), drop=0.1):
        super(ResidualAttentionNetwork, self).__init__()
        self.first_conv_block = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        self.mpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = []
        for i, (b, a) in enumerate(zip(blocks, am_per_block)):
            for _ in range(a):
                layers.append(Attention_module(channels[i], channels[i + 1]))
            for _ in range(b):
                layers.append(block(channels[i + 1], channels[i + 1], stride=2))
        self.body = nn.Sequential(*layers)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        if drop is not None:
            self.dropout = nn.Dropout(drop)
        else:
            self.dropout = None
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.first_conv_block(x)
        x = self.mpool(x)

        x = self.body(x)

        x = self.gap(x)

        x = self.fc1(torch.flatten(x, 1))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
