"""Streamlit interface for the trained brain MRI classifier."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from src.inference import load_h5


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


@st.cache_resource
def load_classifier() -> tuple[tf.keras.Model, dict]:
    metadata = json.loads((MODEL_DIR / "deployment.json").read_text())
    model = load_h5(MODEL_DIR / metadata["model_path"])
    return model, metadata


st.set_page_config(page_title="Brain MRI classification", page_icon="🧠", layout="centered")
st.title("Brain MRI image classification")
st.caption("Research demonstration using a four-class image model")
st.info("This model is not clinically validated and must not be used to diagnose or triage patients.")

try:
    model, metadata = load_classifier()
except (OSError, ValueError, KeyError) as exc:
    st.error("The trained model is missing or could not be loaded. Run the training pipeline first.")
    st.exception(exc)
    st.stop()

st.write(f"**Selected model:** {metadata['model'].replace('_', ' ').title()}")
upload = st.file_uploader("Upload one brain MRI image", type=["jpg", "jpeg", "png"])
if upload is not None:
    if upload.size > 10 * 1024 * 1024:
        st.error("Please upload an image smaller than 10 MB.")
        st.stop()
    try:
        image = Image.open(io.BytesIO(upload.getvalue())).convert("RGB")
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("This file could not be read as an image.")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)
    image_size = tuple(metadata["image_size"])
    resized = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
    batch = np.expand_dims(np.asarray(resized, dtype=np.float32), 0)
    probabilities = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))
    labels = metadata["classes"]
    st.subheader(f"Prediction: {labels[index].replace('_', ' ').title()}")
    st.write(f"Model score: **{probabilities[index]:.1%}**")
    st.caption("A model score is not a calibrated clinical probability.")
    st.bar_chart({label.replace("_", " ").title(): float(score) for label, score in zip(labels, probabilities)})
