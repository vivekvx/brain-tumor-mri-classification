"""Small group-aware CV screen of learning rates for the three deep models.

Uses training images only. Source-image variants remain in the same fold.
Two short-epoch rates per model are compared; final full-length training remains
separate and uses the chosen rate if this report exists before training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from src.train import CLASSES, IMAGE_SIZE, SEED, custom_cnn, transfer_model


RATES = {
    "custom_cnn": [1e-3, 3e-4],
    "mobilenet_v2": [3e-4, 1e-4],
    "efficientnet_b0": [3e-4, 1e-4],
}


def dataset(images: np.ndarray, labels: np.ndarray, indices: np.ndarray, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((images[indices], labels[indices]))
    if training:
        ds = ds.shuffle(len(indices), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(16).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    manifest = pd.read_csv("data/cleaned/manifest.csv")
    train = manifest[manifest["new_split"] == "train"].reset_index(drop=True)
    images = np.stack([
        np.asarray(Image.open(Path("data/cleaned/train") / row["class"] / row["filename"]).convert("RGB").resize(IMAGE_SIZE), dtype=np.uint8)
        for row in train.to_dict("records")
    ])
    labels = train["class"].map({name: i for i, name in enumerate(CLASSES)}).to_numpy(dtype=np.int32)
    groups = (train["class"] + "/" + train["source_id"]).to_numpy()
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(images, labels, groups))
    for tr, va in folds:
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError("Source-image group leakage in CV")

    result = {"folds": 3, "epochs_per_fold": 2, "training_images": len(train), "models": {}}
    for name, rates in RATES.items():
        candidates = {}
        for rate in rates:
            fold_scores = []
            for fold_index, (tr, va) in enumerate(folds, 1):
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(SEED + fold_index)
                model = custom_cnn() if name == "custom_cnn" else transfer_model(name)
                model.compile(optimizer=tf.keras.optimizers.Adam(rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                model.fit(dataset(images, labels, tr, True), validation_data=dataset(images, labels, va, False), epochs=2, shuffle=False, verbose=0)
                pred = np.argmax(model.predict(dataset(images, labels, va, False), verbose=0), axis=1)
                score = float(f1_score(labels[va], pred, average="macro", zero_division=0))
                fold_scores.append(score)
                print(name, rate, "fold", fold_index, "macro F1", round(score, 3), flush=True)
                del model
            candidates[str(rate)] = {"fold_scores": fold_scores, "mean_f1": float(np.mean(fold_scores))}
            result["models"][name] = {"candidates": candidates}
            Path("reports/deep_cv.json").write_text(json.dumps(result, indent=2) + "\n")
        best = max(rates, key=lambda rate: candidates[str(rate)]["mean_f1"])
        result["models"][name]["selected_learning_rate"] = best
        Path("reports/deep_cv.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Saved reports/deep_cv.json", flush=True)


if __name__ == "__main__":
    main()
