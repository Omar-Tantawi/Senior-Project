"""
EdgeFace Architecture - Lightweight Face Recognition for Edge Devices
======================================================================

Reference:
    George et al., "EdgeFace: Efficient Face Recognition Model for Edge Devices"
    IEEE Transactions on Biometrics, Behavior, and Identity Science (T-BIOM) 2024
    Winner: IJCB 2023 Efficient Face Recognition Competition (compact track)

Official repo: https://github.com/otroshi/edgeface

KEY POINTS:
- ~1.77M parameters (25x smaller than IResNet-50)
- Hybrid CNN + Transformer (uses EdgeNeXt as backbone)
- 99.73% LFW accuracy
- Designed for edge/CCTV deployment

NOTE: This is a SIMPLIFIED implementation. The official EdgeFace uses
EdgeNeXt blocks which require the timm library. For full fidelity:
    pip install timm
    Then import: timm.create_model('edgenext_xx_small')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv — the building block of efficient nets."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        x = self.bn1(self.dw(x))
        x = self.act(self.bn2(self.pw(x)))
        return x


class EdgeBlock(nn.Module):
    """Efficient block inspired by EdgeNeXt — DWConv + pointwise + residual."""
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, dim * expansion, 1, 1, 0)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * expansion, dim, 1, 1, 0)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + residual


class EdgeFace(nn.Module):
    """Simplified EdgeFace-XS for 112x112 face recognition.

    For the OFFICIAL implementation, use timm + EdgeNeXt from
    https://github.com/otroshi/edgeface
    """

    def __init__(self, num_features=512, fp16=True):
        super().__init__()
        self.fp16 = fp16

        # Stem: 112 -> 56
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, 2, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.PReLU(24),
        )

        # Stage 1: 56 -> 28, dim 48
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(24, 48, stride=2),
            EdgeBlock(48),
            EdgeBlock(48),
        )

        # Stage 2: 28 -> 14, dim 96
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(48, 96, stride=2),
            EdgeBlock(96),
            EdgeBlock(96),
            EdgeBlock(96),
        )

        # Stage 3: 14 -> 7, dim 160
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(96, 160, stride=2),
            EdgeBlock(160),
            EdgeBlock(160),
            EdgeBlock(160),
            EdgeBlock(160),
        )

        # Stage 4: 7 -> 7, dim 304
        self.stage4 = nn.Sequential(
            DepthwiseSeparableConv(160, 304, stride=1),
            EdgeBlock(304),
            EdgeBlock(304),
        )

        # Head: GAP -> FC -> 512-dim embedding
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(304, num_features)
        self.bn_out = nn.BatchNorm1d(num_features)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=self.fp16):
            x = self.stem(x)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.gap(x)
            x = self.flatten(x)
            x = self.fc(x.float() if self.fp16 else x)
            x = self.bn_out(x)
        return x


def edgeface_xs(num_features=512, fp16=True):
    """Construct EdgeFace-XS (extra small, ~1.7M params)."""
    return EdgeFace(num_features=num_features, fp16=fp16)


if __name__ == "__main__":
    model = edgeface_xs()
    x = torch.randn(2, 3, 112, 112)
    out = model(x)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Output: {out.shape}")
    print(f"Params: {params:.2f}M")
