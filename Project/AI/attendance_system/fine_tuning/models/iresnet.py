"""IResNet-100 Architecture for Face Recognition.

Improved ResNet (IResNet) is the standard backbone for modern face recognition
systems including ArcFace and AdaFace. Key differences from regular ResNet:
- BatchNorm BEFORE convolution (pre-activation)
- PReLU activation instead of ReLU
- 512-dimensional output embedding
- Optimized for 112x112 face crops

Reference:
    Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    CVPR 2019
"""

import torch
import torch.nn as nn


class IBasicBlock(nn.Module):
    """Improved Basic Block with pre-activation and PReLU."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out


class IResNet(nn.Module):
    """Improved ResNet for face recognition (ArcFace/AdaFace backbone)."""

    def __init__(self, block, layers, dropout=0.0, num_features=512, fp16=False):
        super().__init__()
        self.fp16 = fp16
        self.inplanes = 64

        # Stem: 112x112 -> 56x56
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)

        # Stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Head: feature embedding
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * 7 * 7, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-5),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)

        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x


def iresnet50(num_features=512, **kwargs):
    """IResNet-50."""
    return IResNet(IBasicBlock, [3, 4, 14, 3], num_features=num_features, **kwargs)


def iresnet100(num_features=512, **kwargs):
    """IResNet-100 - standard for ArcFace/AdaFace."""
    return IResNet(IBasicBlock, [3, 13, 30, 3], num_features=num_features, **kwargs)


if __name__ == "__main__":
    # Quick test
    model = iresnet100()
    x = torch.randn(2, 3, 112, 112)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
