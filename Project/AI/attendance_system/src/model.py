"""
src/model.py — ArcFace Model with IResNet Backbone

Architecture:
  Backbone: IResNet-50 (pretrained on MS1MV2 or VGGFace2)
  Loss:     ArcFace (Additive Angular Margin Loss)
  Head:     512-d embedding → cosine similarity for recognition

Why ArcFace?
  - Best performance on angled faces (geometric margin in angular space)
  - State-of-the-art on LFW (99.83%), AgeDB-30, CFP-FP
  - More discriminative than softmax for open-set recognition (new students)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# ─── IResNet Backbone ─────────────────────────────────────────────────────────
class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes,   eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes,   planes, 3, stride=stride, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes,   eps=1e-5)
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
    """
    IResNet backbone for face recognition.
    Configurations:
      iresnet18:  [2, 2,  2, 2]
      iresnet34:  [3, 4,  6, 3]
      iresnet50:  [3, 4, 14, 3]  ← recommended
      iresnet100: [3, 13,30, 3]
    """
    def __init__(self, layers, embedding_dim=512, dropout=0.0):
        super().__init__()
        self.inplanes = 64
        self.embedding_dim = embedding_dim

        self.conv1  = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu  = nn.PReLU(64)

        self.layer1 = self._make_layer(64,  layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.bn2    = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout)
        self.fc     = nn.Linear(512 * 7 * 7, embedding_dim)  # 112×112 → 7×7 after 4×stride-2
        self.features = nn.BatchNorm1d(embedding_dim, eps=1e-5)

        self._init_weights()

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, eps=1e-5),
            )
        layers = [IBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(IBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.features(x)
        return x  # Raw 512-d embedding (not normalized here)


def iresnet18(embedding_dim=512,  **kw): return IResNet([2,  2,  2,  2],  embedding_dim, **kw)
def iresnet34(embedding_dim=512,  **kw): return IResNet([3,  4,  6,  3],  embedding_dim, **kw)
def iresnet50(embedding_dim=512,  **kw): return IResNet([3,  4, 14,  3],  embedding_dim, **kw)
def iresnet100(embedding_dim=512, **kw): return IResNet([3, 13, 30,  3],  embedding_dim, **kw)


# ─── ArcFace Loss ─────────────────────────────────────────────────────────────
class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss (Deng et al., CVPR 2019)

    Adds a fixed angular margin m to the angle between the embedding
    and the target class weight, making the decision boundary more strict.

    Key advantage for attendance: embeddings of different students are
    separated by a clear angular margin → fewer false positives.
    """

    def __init__(self, embedding_dim: int, num_classes: int,
                 margin: float = 0.5, scale: float = 64.0, easy_margin: bool = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes   = num_classes
        self.margin        = margin
        self.scale         = scale
        self.easy_margin   = easy_margin

        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m  = math.cos(margin)
        self.sin_m  = math.sin(margin)
        self.th     = math.cos(math.pi - margin)
        self.mm     = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize embeddings and weights → cosine similarity
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        w_norm   = F.normalize(self.weight,    p=2, dim=1)
        cosine   = F.linear(emb_norm, w_norm)

        # Add angular margin to target class
        sine         = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        phi          = cosine * self.cos_m - sine * self.sin_m   # cos(θ + m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale

        loss = F.cross_entropy(logits, labels)
        return loss


# ─── CosFace Loss (alternative) ───────────────────────────────────────────────
class CosFaceLoss(nn.Module):
    """CosFace: Large Margin Cosine Loss (simpler alternative to ArcFace)."""
    def __init__(self, embedding_dim, num_classes, margin=0.35, scale=64.0):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale  = scale

    def forward(self, embeddings, labels):
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        w_norm   = F.normalize(self.weight,    p=2, dim=1)
        cosine   = F.linear(emb_norm, w_norm)
        phi      = cosine - self.margin

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale
        return F.cross_entropy(logits, labels)


# ─── Full Model Wrapper ────────────────────────────────────────────────────────
class FaceRecognitionModel(nn.Module):
    """
    Full model = backbone + loss head.
    During inference, only the backbone is needed (no loss head).
    """
    def __init__(self, backbone_name: str = "iresnet50",
                 num_classes: int = 8631,
                 embedding_dim: int = 512,
                 loss_type: str = "arcface"):
        super().__init__()

        backbones = {
            "iresnet18":  iresnet18,
            "iresnet34":  iresnet34,
            "iresnet50":  iresnet50,
            "iresnet100": iresnet100,
        }
        assert backbone_name in backbones, f"Unknown backbone: {backbone_name}"
        self.backbone = backbones[backbone_name](embedding_dim=embedding_dim)

        if loss_type == "arcface":
            self.head = ArcFaceLoss(embedding_dim, num_classes)
        elif loss_type == "cosface":
            self.head = CosFaceLoss(embedding_dim, num_classes)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, x, labels=None):
        embeddings = self.backbone(x)
        if labels is not None:
            loss = self.head(embeddings, labels)
            return embeddings, loss
        return embeddings

    def get_embedding(self, x):
        """Get normalized L2 embedding for inference."""
        self.eval()
        with torch.no_grad():
            emb = self.backbone(x)
            emb = F.normalize(emb, p=2, dim=1)
        return emb


# ─── Load Pretrained from InsightFace ─────────────────────────────────────────
def load_pretrained_insightface(model: IResNet, pretrained_path: str = None):
    """
    Load InsightFace pretrained weights (MS1MV2 trained IResNet50).
    These are publicly available and dramatically speed up convergence.
    """
    if pretrained_path and Path(pretrained_path).exists():
        state_dict = torch.load(pretrained_path, map_location="cpu")
        # InsightFace saves backbone weights directly
        model.load_state_dict(state_dict, strict=False)
        print(f"[OK] Pretrained weights loaded from {pretrained_path}")
    else:
        print("[WARN] No pretrained weights found. Training from scratch (slower).")
        print("  Download from: https://github.com/deepinsight/insightface/tree/master/model_zoo")
    return model


# ─── Quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running on {device}")

    model = FaceRecognitionModel(
        backbone_name="iresnet50",
        num_classes=8631,
        embedding_dim=512,
        loss_type="arcface"
    ).to(device)

    dummy_input  = torch.randn(4, 3, 112, 112).to(device)
    dummy_labels = torch.randint(0, 8631, (4,)).to(device)

    embeddings, loss = model(dummy_input, dummy_labels)
    print(f"[OK] Embedding shape: {embeddings.shape}")   # [4, 512]
    print(f"[OK] Loss: {loss.item():.4f}")

    total_params = sum(p.numel() for p in model.backbone.parameters()) / 1e6
    print(f"[OK] Backbone parameters: {total_params:.1f}M")
