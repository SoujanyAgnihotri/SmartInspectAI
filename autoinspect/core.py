"""Minimal PatchCore-style anomaly detector.

Idea: train ONLY on defect-free ("good") images. Store patch features of good parts
in a memory bank. At inference, a patch that is far from every stored patch is
anomalous -> gives a defect heatmap and an image-level score.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision import transforms as T
from torchvision.transforms import functional as TF

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)


def greedy_coreset(feats: torch.Tensor, n: int, proj_dim: int = 128, seed: int = 0) -> torch.Tensor:
    """Pick n representative patches (k-center greedy) so the memory bank stays small."""
    N, D = feats.shape
    n = min(n, N)
    g = torch.Generator().manual_seed(seed)
    proj = (torch.randn(D, proj_dim, generator=g) / proj_dim**0.5).to(feats.device)
    z = feats @ proj
    first = int(torch.randint(N, (1,), generator=g))
    idx = [first]
    min_d = torch.cdist(z, z[first : first + 1]).squeeze(1)
    for _ in range(n - 1):
        i = int(torch.argmax(min_d))
        idx.append(i)
        min_d = torch.minimum(min_d, torch.cdist(z, z[i : i + 1]).squeeze(1))
    return feats[idx]


class PatchCore:
    def __init__(self, image_size=256, crop_size=224, coreset_ratio=0.1,
                 pretrained=True, device=None):
        self.image_size, self.crop_size = image_size, crop_size
        self.coreset_ratio = coreset_ratio
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = models.wide_resnet50_2(weights=weights).eval().to(self.device)
        self.geo = T.Compose([T.Resize(image_size), T.CenterCrop(crop_size)])
        self.norm = T.Compose([T.ToTensor(),
                               T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.bank: torch.Tensor | None = None
        self.threshold: float | None = None

    # ---------- features ----------
    def _load(self, items) -> torch.Tensor:
        imgs = [i if isinstance(i, Image.Image) else Image.open(i) for i in items]
        return torch.stack([self.norm(self.geo(im.convert("RGB"))) for im in imgs])

    @torch.no_grad()
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        f2 = n.layer2(n.layer1(x))
        f3 = n.layer3(f2)
        f2, f3 = F.avg_pool2d(f2, 3, 1, 1), F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)  # B x 1536 x H x W

    # ---------- train ----------
    def fit(self, paths, batch_size=16):
        feats = []
        for i in range(0, len(paths), batch_size):
            f = self._embed(self._load(paths[i : i + batch_size]).to(self.device))
            feats.append(f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]).cpu())
        feats = torch.cat(feats).to(self.device)
        self.bank = greedy_coreset(feats, max(1, int(len(feats) * self.coreset_ratio)))
        return self

    # ---------- inference ----------
    @torch.no_grad()
    def predict(self, items, batch_size=16):
        """Returns (image_scores [N], anomaly_maps [N, crop, crop])."""
        assert self.bank is not None, "fit() or load() first"
        scores, maps = [], []
        for i in range(0, len(items), batch_size):
            f = self._embed(self._load(items[i : i + batch_size]).to(self.device))
            B, C, H, W = f.shape
            q = f.permute(0, 2, 3, 1).reshape(-1, C)
            d = torch.cat([torch.cdist(c, self.bank).min(1).values for c in q.split(4096)])
            m = F.interpolate(d.reshape(B, 1, H, W), size=(self.crop_size,) * 2,
                              mode="bilinear", align_corners=False)
            m = TF.gaussian_blur(m, kernel_size=33, sigma=4.0)
            maps.append(m.squeeze(1).cpu())
            scores.append(m.flatten(1).max(1).values.cpu())
        return torch.cat(scores).numpy(), torch.cat(maps).numpy()

    def calibrate(self, good_items, margin=1.05):
        """Threshold = max score on held-out good images * margin."""
        s, _ = self.predict(good_items)
        self.threshold = float(s.max() * margin)
        return self.threshold

    def is_defective(self, score: float) -> bool:
        return self.threshold is not None and score > self.threshold

    # ---------- io ----------
    def save(self, path):
        torch.save({"bank": self.bank.cpu(), "threshold": self.threshold,
                    "image_size": self.image_size, "crop_size": self.crop_size}, path)

    @classmethod
    def load(cls, path, device=None, pretrained=True):
        ck = torch.load(path, map_location="cpu")
        m = cls(ck["image_size"], ck["crop_size"], pretrained=pretrained, device=device)
        m.bank, m.threshold = ck["bank"].to(m.device), ck["threshold"]
        return m
