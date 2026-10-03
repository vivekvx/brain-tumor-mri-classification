"""Load legacy HDF5 Keras models with preprocessing state restored."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def load_h5(path: str | Path) -> tf.keras.Model:
    model = tf.keras.models.load_model(path, compile=False)
    # Keras 3.15 HDF5 loading restores the EfficientNet Normalization weights
    # but leaves its cached mean/variance at their defaults until finalized.
    # This is required for HDF5 inference to match the trained checkpoint.
    def visit(layer: tf.keras.layers.Layer) -> None:
        if isinstance(layer, tf.keras.layers.Normalization):
            layer.finalize_state()
        for child in getattr(layer, "layers", []):
            visit(child)
    visit(model)
    return model
