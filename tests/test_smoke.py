"""Smoke test with random weights + synthetic images (no downloads)."""
import numpy as np
from PIL import Image, ImageDraw

from autoinspect import PatchCore, overlay


def make(defect=False, seed=0):
    rng = np.random.default_rng(seed)
    a = (rng.normal(128, 4, (256, 256, 3))).clip(0, 255).astype("uint8")
    im = Image.fromarray(a)
    if defect:
        ImageDraw.Draw(im).ellipse((100, 100, 150, 150), fill=(255, 0, 0))
    return im


def test_pipeline():
    pc = PatchCore(pretrained=False, coreset_ratio=0.05, device="cpu")
    pc.fit([make(seed=i) for i in range(8)], batch_size=4)
    pc.calibrate([make(seed=100 + i) for i in range(3)])
    s, m = pc.predict([make(seed=200), make(True, seed=201)])
    assert m.shape == (2, 224, 224)
    assert s[1] > s[0]
    overlay(pc.geo(make(True)), m[1], pc.threshold)
