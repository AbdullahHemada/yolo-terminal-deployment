import streamlit as st
import numpy as np
import cv2
from PIL import Image

from utils.detector import YOLOModel
from utils.visualization import draw_boxes


st.set_page_config(page_title="Parking Space Detection", layout="centered")

MODEL_PATH = "model/best.pt"


@st.cache_resource
def load_model():
    return YOLOModel(MODEL_PATH)


def load_labels(path):
    with open(path) as f:
        return [l.strip() for l in f.readlines()]


model = load_model()
class_names = load_labels("model/labels.txt")

st.title("🚗 Parking Space Detection (YOLO)")
st.write("Upload image or use demo.")

uploaded = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
else:
    image = Image.open("assets/demo.png").convert("RGB")

image_np = np.array(image)

if st.button("Run Detection"):
    with st.spinner("Detecting..."):
        detections = model.predict(image_np)
        result = draw_boxes(image_np, detections, class_names)

    st.image(result, caption="Detection Result", use_column_width=True)

