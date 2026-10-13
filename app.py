"""Streamlit interface for the trained brain MRI classifier."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from src.inference import load_h5


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "reports" / "figures"
MODEL_NAMES = {
    "custom_cnn": "Custom CNN",
    "mobilenet_v2": "MobileNetV2",
    "efficientnet_b0": "EfficientNetB0",
    "mobilenet_v2_finetuned": "MobileNetV2 fine-tuned",
}
CHARTS = {
    "1 · Training class counts": "01_train_class_counts.png",
    "2 · Class mix across splits": "02_split_class_mix.png",
    "3 · MRI examples by class": "03_class_examples.png",
    "4 · Image file sizes": "04_file_kb.png",
    "5 · Brightness": "05_brightness.png",
    "6 · Contrast": "06_contrast.png",
    "7 · Entropy": "07_entropy.png",
    "8 · Edge strength": "08_edge_strength.png",
    "9 · Center brightness": "09_center_brightness.png",
    "10 · Color channel means": "10_channel_means.png",
    "11 · Brightness and contrast": "11_brightness_contrast.png",
    "12 · Brightness and edges": "12_brightness_edges.png",
    "13 · Feature correlation": "13_feature_correlation.png",
    "14 · Feature pairplot": "14_feature_pairplot.png",
    "15 · Source-image variants": "15_source_variants.png",
}


@st.cache_resource
def load_classifier() -> tuple[tf.keras.Model, dict]:
    metadata = json.loads((MODEL_DIR / "deployment.json").read_text())
    model = load_h5(MODEL_DIR / metadata["model_path"])
    return model, metadata


st.set_page_config(page_title="Brain MRI classification", page_icon="🧠", layout="centered")
st.title("Brain MRI image classification")
st.caption("Research demonstration using a four-class image model")
st.info("This model is not clinically validated and must not be used to diagnose or triage patients.")

predict_tab, results_tab, charts_tab = st.tabs(["Try the model", "Model results", "Explore the data"])

with predict_tab:
    try:
        model, metadata = load_classifier()
    except (OSError, ValueError, KeyError) as exc:
        st.error("The trained model is missing or could not be loaded. Run the training pipeline first.")
        st.exception(exc)
    else:
        st.write(f"**Selected model:** {MODEL_NAMES[metadata['model']]}")
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

with results_tab:
    results = json.loads((MODEL_DIR / "training_results.json").read_text())
    selected_name = results["selected_model"]
    selected = results[selected_name]["test"]
    st.subheader("Held-out test results")
    st.caption("The model was selected using validation macro F1 before the test set was scored.")
    accuracy_col, f1_col, recall_col = st.columns(3)
    accuracy_col.metric("Accuracy", f"{selected['accuracy']:.1%}")
    f1_col.metric("Macro F1", f"{selected['f1_macro']:.3f}")
    recall_col.metric("Meningioma recall", f"{selected['per_class']['meningioma']['recall']:.1%}")

    st.subheader("Model comparison")
    comparison = pd.DataFrame(
        [
            {
                "Model": MODEL_NAMES[name],
                "Validation accuracy": f"{results[name]['validation']['accuracy']:.1%}",
                "Validation macro F1": f"{results[name]['validation']['f1_macro']:.3f}",
                "Test accuracy": f"{results[name]['test']['accuracy']:.1%}",
                "Test macro F1": f"{results[name]['test']['f1_macro']:.3f}",
            }
            for name in MODEL_NAMES
        ]
    )
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    st.caption(f"Selected for deployment: {MODEL_NAMES[selected_name]}.")

    st.subheader("Training and validation trends")
    trend_name = st.selectbox("Choose a model", list(MODEL_NAMES), index=list(MODEL_NAMES).index(selected_name), format_func=MODEL_NAMES.get)
    st.image(FIGURE_DIR / f"{trend_name}_training.png", use_container_width=True)

    st.subheader("Selected model confusion matrix")
    st.image(FIGURE_DIR / "selected_model_confusion_matrix.png", use_container_width=True)
    st.caption("These image-level scores may overestimate performance on new patients or scanners. Patient IDs and external validation are unavailable.")

with charts_tab:
    st.subheader("Exploratory analysis")
    st.write("The notebook contains 15 charts of the training data, class examples, image features, and source-image variants.")
    chart_name = st.selectbox("Choose a chart", list(CHARTS), index=2)
    st.image(FIGURE_DIR / CHARTS[chart_name], caption=chart_name, use_container_width=True)
    st.caption("Known variants of the same source image were kept in one split to reduce evaluation leakage.")
    st.link_button("Open the complete notebook on GitHub", "https://github.com/vivekvx/brain-tumor-mri-classification/blob/main/Brain_Tumor_MRI_Project.ipynb")
