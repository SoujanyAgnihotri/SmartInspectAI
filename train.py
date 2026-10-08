"""Train on defect-free images, calibrate, (optionally) evaluate on MVTec-style test set.

Expected layout (MVTec AD / VisA-converted):
  <category_dir>/train/good/*.png
  <category_dir>/test/good/*.png        (optional, for evaluation)
  <category_dir>/test/<defect>/*.png    (optional)
  <category_dir>/ground_truth/<defect>/<name>_mask.png   (optional, pixel AUROC)
Any folder of good images also works via --train-dir.
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from autoinspect import PatchCore, list_images


def load_mask(path, pc: PatchCore):
    m = pc.geo(Image.open(path).convert("L"))  # same resize + center-crop as images
    return (np.asarray(m) > 0).astype(np.uint8)


def evaluate(pc: PatchCore, cat: Path):
    test = cat / "test"
    paths, labels, masks = [], [], []
    for p in list_images(test):
        defect = p.parent.name
        paths.append(p)
        labels.append(0 if defect == "good" else 1)
        gt = cat / "ground_truth" / defect / f"{p.stem}_mask.png"
        masks.append(load_mask(gt, pc) if gt.exists() else np.zeros((pc.crop_size,) * 2, np.uint8))
    scores, maps = pc.predict(paths)
    out = {"image_auroc": roc_auc_score(labels, scores)}
    if any(m.any() for m in masks):
        out["pixel_auroc"] = roc_auc_score(np.concatenate([m.ravel() for m in masks]),
                                           maps.reshape(-1))
    preds = scores > pc.threshold
    out["accuracy@threshold"] = float((preds == np.array(labels)).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category-dir", help="MVTec-style category folder")
    ap.add_argument("--train-dir", help="folder of defect-free images (alternative)")
    ap.add_argument("--out", default="models/model.pt")
    ap.add_argument("--coreset-ratio", type=float, default=0.1)
    ap.add_argument("--holdout", type=float, default=0.1, help="fraction of good imgs for threshold")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cat = Path(a.category_dir) if a.category_dir else None
    train_dir = Path(a.train_dir) if a.train_dir else cat / "train" / "good"
    imgs = list_images(train_dir)
    assert len(imgs) >= 5, f"need >=5 good images in {train_dir}"
    random.Random(a.seed).shuffle(imgs)
    k = max(1, int(len(imgs) * a.holdout))
    held, train = imgs[:k], imgs[k:]

    pc = PatchCore(coreset_ratio=a.coreset_ratio)
    print(f"Fitting on {len(train)} images ({k} held out for threshold)...")
    pc.fit(train)
    print(f"Memory bank: {tuple(pc.bank.shape)} | threshold: {pc.calibrate(held):.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pc.save(a.out)
    print("Saved ->", a.out)

    if cat and (cat / "test").exists():
        print("Evaluating:", evaluate(pc, cat))


if __name__ == "__main__":
    main()
