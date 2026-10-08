"""Streamlit demo:  streamlit run app.py"""
import streamlit as st
from PIL import Image

from autoinspect import PatchCore, overlay

st.set_page_config(page_title="AutoInspect AI", page_icon="🔍", layout="wide")
st.title("🔍 AutoInspect AI")
st.caption("Visual quality inspection for automotive components - trained on defect-free parts only.")


@st.cache_resource
def get_model(path):
    return PatchCore.load(path)


model_path = st.sidebar.text_input("Model path", "models/model.pt")
try:
    pc = get_model(model_path)
except Exception as e:
    st.error(f"Could not load model: {e}\nTrain one first with train.py")
    st.stop()

sens = st.sidebar.slider("Sensitivity (threshold scale)", 0.5, 1.5, 1.0, 0.05)
files = st.file_uploader("Upload component image(s)", type=["png", "jpg", "jpeg", "bmp"],
                         accept_multiple_files=True)
for f in files or []:
    img = Image.open(f).convert("RGB")
    score, amap = pc.predict([img])
    thr = pc.threshold * sens
    bad = score[0] > thr
    c1, c2 = st.columns(2)
    c1.image(pc.geo(img), caption=f.name)
    c2.image(overlay(pc.geo(img), amap[0], thr), caption="Defect heatmap")
    (st.error if bad else st.success)(f"{'DEFECT DETECTED' if bad else 'PASS'} - score {score[0]:.2f} (threshold {thr:.2f})")
