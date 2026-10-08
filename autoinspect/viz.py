from __future__ import annotations

import numpy as np
from matplotlib import cm
from PIL import Image


def overlay(base: Image.Image, amap: np.ndarray, threshold: float | None = None, alpha=0.45):
    """Blend an anomaly heatmap on top of the (resized/cropped) image."""
    hi = (threshold * 1.5) if threshold else float(amap.max() + 1e-8)
    norm = np.clip(amap / (hi + 1e-8), 0, 1)
    heat = (cm.jet(norm)[..., :3] * 255).astype(np.uint8)
    base_arr = np.asarray(base.convert("RGB").resize(heat.shape[1::-1]))
    out = (base_arr * (1 - alpha) + heat * alpha).astype(np.uint8)
    return Image.fromarray(out)
