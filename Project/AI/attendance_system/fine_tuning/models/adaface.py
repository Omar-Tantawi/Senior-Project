"""AdaFace Loss for Face Recognition.

AdaFace adapts the angular margin based on the quality of the image (norm of the
feature vector). High-quality images get a stricter margin; low-quality images
(blurry, low-resolution, occluded - typical of CCTV) get a more relaxed margin.

This makes AdaFace especially powerful for surveillance scenarios where image
quality varies dramatically.

Reference:
    Kim et al., "AdaFace: Quality Adaptive Margin for Face Recognition"
    CVPR 2022 (Oral)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaFace(nn.Module):
    """AdaFace Loss.

    Args:
        embedding_size: Dimension of face embeddings (typically 512)
        classnum: Number of identities in training set
        m: Margin parameter (default 0.4)
        h: Concentration parameter (default 0.333)
        s: Scale parameter (default 64)
        t_alpha: EMA rate for tracking feature norm statistics
    """

    def __init__(
        self,
        embedding_size=512,
        classnum=70722,
        m=0.4,
        h=0.333,
        s=64.0,
        t_alpha=1.0,
    ):
        super().__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, classnum))

        # Initialize kernel (classifier weights)
        nn.init.normal_(self.kernel, std=0.01)

        self.m = m
        self.eps = 1e-3
        self.h = h
        self.s = s

        # Running statistics for batch norm (image quality tracking)
        self.t_alpha = t_alpha
        self.register_buffer("t", torch.zeros(1))
        self.register_buffer("batch_mean", torch.ones(1) * 20)
        self.register_buffer("batch_std", torch.ones(1) * 100)

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, embedding_size) face features from backbone
            labels: (B,) ground truth identity labels

        Returns:
            cos_theta: (B, classnum) modified logits with AdaFace margin
        """
        # Compute norm of embeddings (proxy for image quality)
        norms = torch.norm(embeddings, 2, -1, keepdim=True)
        embeddings = embeddings / norms  # Normalize

        # Normalize kernel
        kernel_norm = F.normalize(self.kernel, dim=0)

        # Compute cosine similarity
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1 + self.eps, 1 - self.eps)

        # Update running batch statistics
        safe_norms = torch.clip(norms, min=0.001, max=100).clone().detach()

        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        # Compute margin scaler based on image quality (norm)
        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)
        margin_scaler = margin_scaler * self.h
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        # Angular margin (g_angular)
        m_arc = torch.zeros_like(cos_theta)
        m_arc.scatter_(1, labels.view(-1, 1), 1)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cos_theta.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi - self.eps)
        cos_theta = theta_m.cos()

        # Additive margin (g_additive)
        m_cos = torch.zeros_like(cos_theta)
        m_cos.scatter_(1, labels.view(-1, 1), 1)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cos_theta = cos_theta - m_cos

        # Scale
        scaled_cos_theta = cos_theta * self.s

        return scaled_cos_theta


if __name__ == "__main__":
    # Quick test
    loss_fn = AdaFace(embedding_size=512, classnum=1000)
    embeddings = torch.randn(8, 512)
    labels = torch.randint(0, 1000, (8,))
    logits = loss_fn(embeddings, labels)
    print(f"Embeddings: {embeddings.shape}")
    print(f"Logits: {logits.shape}")
    loss = F.cross_entropy(logits, labels)
    print(f"Loss: {loss.item():.4f}")
