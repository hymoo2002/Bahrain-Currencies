"""
Bahraini Currency Detector - Streamlit app.

Run:
    streamlit run app.py

Requires:
    model.keras and class_names.json produced by train.py
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model.keras"
LABELS_PATH = ROOT / "class_names.json"
IMG_SIZE = (224, 224)

PRETTY_LABEL = {
    "One": "1 BHD",
    "Five": "5 BHD",
    "Ten": "10 BHD",
    "Twenty": "20 BHD",
    "5": "5 fils",
    "25": "25 fils",
    "50": "50 fils",
    "100": "100 fils",
    "500": "500 fils",
}


st.set_page_config(page_title="Bahraini Currency Detector", page_icon="💵", layout="centered")
st.title("Bahraini Currency Detector")
st.caption("Take a photo of a Bahraini note or coin and I'll predict what it is.")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        class_names = json.load(f)
    return model, class_names


model, class_names = load_model()

if model is None:
    st.error(
        "Model not found. Train it first with:\n\n"
        "```bash\npython train.py\n```\n\n"
        "That produces `model.keras` and `class_names.json` next to this file."
    )
    st.stop()


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict(pil_img: Image.Image):
    x = preprocess_image(pil_img)
    probs = model.predict(x, verbose=0)[0]
    order = np.argsort(probs)[::-1]
    return [(class_names[i], float(probs[i])) for i in order]


def render_prediction(pil_img: Image.Image):
    ranked = predict(pil_img)
    top_label, top_prob = ranked[0]
    pretty = PRETTY_LABEL.get(top_label, top_label)

    st.image(pil_img, caption="Input", use_container_width=True)
    st.subheader(f"Prediction: {pretty}")
    st.metric("Confidence", f"{top_prob * 100:.1f}%")

    with st.expander("All class probabilities"):
        for label, p in ranked:
            display = PRETTY_LABEL.get(label, label)
            st.write(f"**{display}** ({label}) — {p * 100:.2f}%")
            st.progress(min(max(p, 0.0), 1.0))


mode = st.radio("Input method", ["Camera", "Upload image"], horizontal=True)

if mode == "Camera":
    photo = st.camera_input("Point the camera at the currency")
    if photo is not None:
        render_prediction(Image.open(photo))
else:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        render_prediction(Image.open(uploaded))

with st.sidebar:
    st.header("About")
    st.write(
        "This app classifies Bahraini currency using a MobileNetV2 model fine-tuned on "
        "your `augmented/` dataset."
    )
    st.write(f"**Classes:** {', '.join(class_names)}")
    st.write("**Tips for better predictions**")
    st.write("- Fill the frame with the note/coin")
    st.write("- Use even lighting, avoid glare")
    st.write("- Plain background helps")
