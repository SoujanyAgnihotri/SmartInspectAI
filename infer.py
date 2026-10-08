"""Run inspection on one image or a folder. Saves heatmap overlays."""
import argparse
from pathlib import Path

from PIL import Image

from autoinspect import PatchCore, list_images, overlay

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/model.pt")
ap.add_argument("--input", required=True, help="image file or folder")
ap.add_argument("--out", default="results")
a = ap.parse_args()

pc = PatchCore.load(a.model)
inp = Path(a.input)
paths = list_images(inp) if inp.is_dir() else [inp]
scores, maps = pc.predict(paths)
Path(a.out).mkdir(exist_ok=True)
for p, s, m in zip(paths, scores, maps):
    verdict = "DEFECT" if pc.is_defective(s) else "OK"
    print(f"{p.name}: score={s:.3f} -> {verdict}")
    overlay(pc.geo(Image.open(p).convert("RGB")), m, pc.threshold).save(Path(a.out) / f"{p.stem}_{verdict}.png")
