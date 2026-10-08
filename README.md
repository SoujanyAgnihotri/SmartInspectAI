# 🔍 AutoInspect AI

AI-powered visual quality inspection for automotive component manufacturing, built for the **Synapse** hackathon.

AutoInspect learns what a *good* part looks like and flags anything that deviates, with a heatmap showing **where** the defect is. It needs **no defect labels** for training, which matters on a real assembly line where defect samples are rare.

## How it works

1. **Train** on defect-free images only.
2. A pretrained **Wide-ResNet50** extracts patch-level features; a compact **memory bank** of "normal" patches is built (k-center greedy coreset), following the PatchCore approach.
3. At inference, each patch of a new image is compared to the memory bank. Patches far from anything seen in good parts are anomalous.
4. Output: image-level **PASS / DEFECT** verdict + pixel-level **defect heatmap**.
5. The threshold is calibrated automatically on held-out good images (adjustable in the demo).

```
image -> WideResNet50 (layer2+layer3) -> patch features
      -> nearest-neighbour distance to memory bank -> anomaly map -> verdict
```

## Datasets

Designed for public industrial datasets: **MVTec AD**, **MVTec AD 2**, **VisA**, and **NEU** surface defects.
Download them from their official sites (licences differ, so they are not included here).
Any folder of defect-free images also works.

> Note: NEU has no defect-free class, so use it for qualitative testing against a model trained on good surface images rather than as training data.

## Setup

```bash
git clone https://github.com/CodeJunkie101/AutoInspect-AI.git
cd AutoInspect-AI
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Train (MVTec-style folder)**
```bash
python train.py --category-dir data/mvtec/metal_nut --out models/metal_nut.pt
```
Expected layout: `train/good/`, optional `test/<good|defect>/` and `ground_truth/`. If `test/` exists, image AUROC and pixel AUROC are printed.

**Train (any folder of good images)**
```bash
python train.py --train-dir my_good_parts/ --out models/model.pt
```

**Inspect images**
```bash
python infer.py --model models/metal_nut.pt --input some_image.png
python infer.py --model models/metal_nut.pt --input folder/ --out results
```

**Demo UI**
```bash
streamlit run app.py
```
Upload images, see the verdict and heatmap, and tune sensitivity from the sidebar.

## Project structure

```
autoinspect/core.py   PatchCore model (features, coreset, scoring, save/load)
autoinspect/viz.py    heatmap overlay
train.py              train + calibrate + evaluate
infer.py              CLI inference
app.py                Streamlit demo
tests/                smoke test (no downloads needed)
```

## Results

_Add your numbers here after running `train.py` (image AUROC / pixel AUROC per category) and a couple of screenshots from the demo in `assets/`._

## Limitations & next steps

- One model per product category (no cross-category generalisation yet).
- Memory bank grows with training set size; coreset ratio trades accuracy for speed.
- Next: ONNX/TensorRT export for line-speed inference, camera stream input, defect-type classification.

## Team

Built by Kunal Meena (SIT Pune) for Synapse. Add teammates here.

## License

MIT
