from torch import nn
import torch
from torchvision.models import (
    resnext101_64x4d,
    resnext50_32x4d,
)
from torchvision.models.resnet import ResNet
from models.cbam import CBAM
from functools import partial


class Bottleneck_CBAM(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None,
                 use_cbam=False):
        super(Bottleneck_CBAM, self).__init__()

        width = int(planes * (base_width / 64.)) * groups
        norm_layer = norm_layer or nn.BatchNorm2d

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            dilation=dilation,
            bias=False
        )
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(
            width,
            planes * self.expansion,
            kernel_size=1,
            bias=False
        )
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM(planes * self.expansion, 16)
        else:
            self.cbam = None

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

        if self.cbam is not None:
            out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNeXt50(nn.Module):
    def __init__(self, pretrained=True, use_cbam=True, num_classes=100):
        super(ResNeXt50, self).__init__()
        block = partial(Bottleneck_CBAM, use_cbam=use_cbam)
        block.expansion = Bottleneck_CBAM.expansion

        self.model = ResNet(
            block,
            [3, 4, 6, 3],
            groups=32,
            width_per_group=4
        )

        if pretrained:
            pretrained_model = resnext50_32x4d(weights='IMAGENET1K_V1')
            self.model.load_state_dict(
                pretrained_model.state_dict(),
                strict=False
            )

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, num_classes),
            nn.LogSoftmax()
        )

    def forward(self, x):
        return self.model(x)


class ResNeXt101(nn.Module):
    def __init__(self, pretrained=True, use_cbam=False, num_classes=100):
        super(ResNeXt101, self).__init__()
        block = partial(Bottleneck_CBAM, use_cbam=use_cbam)
        block.expansion = Bottleneck_CBAM.expansion

        self.model = ResNet(
            block,
            [3, 4, 23, 3],
            groups=64,
            width_per_group=4
        )

        if pretrained:
            pretrained_model = resnext101_64x4d(weights='IMAGENET1K_V1')
            self.model.load_state_dict(
                pretrained_model.state_dict(),
                strict=False
            )

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, num_classes),
            nn.LogSoftmax()
        )

    def forward(self, x):
        return self.model(x)


class ResNeXt101_CBAM(nn.Module):
    def __init__(self, pretrained=True, use_cbam=True, num_classes=100):
        super(ResNeXt101_CBAM, self).__init__()
        block = partial(Bottleneck_CBAM, use_cbam=use_cbam)
        block.expansion = Bottleneck_CBAM.expansion

        self.model = ResNet(
            block,
            [3, 4, 23, 3],
            groups=64,
            width_per_group=4
        )

        if pretrained:
            pretrained_model = resnext101_64x4d(weights='IMAGENET1K_V1')
            self.model.load_state_dict(
                pretrained_model.state_dict(),
                strict=False
            )

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, num_classes),
            nn.LogSoftmax()
        )

    def forward(self, x):
        return self.model(x)


class ResNeXt101_CBAM_Layer(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNeXt101_CBAM_Layer, self).__init__()

        self.model = resnext101_64x4d(weights='IMAGENET1K_V1')
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, num_classes),
            nn.LogSoftmax()
        )
        self.cbam1 = CBAM(256, 16)
        self.cbam2 = CBAM(512, 16)
        self.cbam3 = CBAM(1024, 16)
        self.cbam4 = CBAM(2048, 16)

    def forward(self, x):
        # 前面部分與原模型相同
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.cbam1(x)

        x = self.model.layer2(x)
        x = self.cbam2(x)

        x = self.model.layer3(x)
        x = self.cbam3(x)

        x = self.model.layer4(x)
        x = self.cbam4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x
