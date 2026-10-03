"""Train and compare three CNN classifiers on the cleaned MRI split.

Run from the repository root:
    .venv/bin/python -m src.train
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from src.inference import load_h5


CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
IMAGE_SIZE = (160, 160)
SEED = 42


def dataset(root: Path, split: str, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    ds = tf.keras.utils.image_dataset_from_directory(
        root / split,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        interpolation="bilinear",
        shuffle=shuffle,
        seed=SEED if shuffle else None,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def augmentation() -> tf.keras.Sequential:
    # Horizontal flips and modest changes are train-only. Vertical flips can
    # reverse brain anatomy and are deliberately excluded.
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomRotation(0.04, seed=SEED),
            tf.keras.layers.RandomZoom(0.08, seed=SEED),
            tf.keras.layers.RandomTranslation(0.04, 0.04, seed=SEED),
            tf.keras.layers.RandomBrightness(0.08, value_range=(0, 255), seed=SEED),
        ],
        name="train_augmentation",
    )


def custom_cnn(augment: bool = True) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation()(inputs) if augment else inputs
    x = tf.keras.layers.Rescaling(1 / 255)(x)
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="custom_cnn")


def transfer_model(name: str, augment: bool = True) -> tf.keras.Model:
    if name == "mobilenet_v2":
        backbone = tf.keras.applications.MobileNetV2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")
        prep = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)
    elif name == "efficientnet_b0":
        backbone = tf.keras.applications.EfficientNetB0(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")
        prep = tf.keras.layers.Activation("linear")  # EfficientNet includes its own rescaling.
    else:
        raise ValueError(name)
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation()(inputs) if augment else inputs
    x = prep(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name=name)


def predictions(model: tf.keras.Model, ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.concatenate([y.numpy() for _, y in ds])
    prob = model.predict(ds, verbose=0)
    return y_true, np.argmax(prob, axis=1), prob


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(CLASSES))).tolist(),
    }


def plot_history(history: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("loss", "accuracy")):
        ax.plot(history[metric], label="training")
        ax.plot(history[f"val_{metric}"], label="validation")
        ax.set(title=metric.title(), xlabel="Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def export_h5(name: str, trained: tf.keras.Model, destination: Path) -> tf.keras.Model:
    # Keras 3's legacy HDF5 writer cannot serialize the random augmentation
    # layers on this TensorFlow version. Augmentation is inactive at inference,
    # so export the same architecture and weights without those layers.
    inference = custom_cnn(augment=False) if name == "custom_cnn" else transfer_model(name, augment=False)
    inference.set_weights(trained.get_weights())
    probe = tf.random.uniform((1, *IMAGE_SIZE, 3), maxval=255, seed=SEED)
    np.testing.assert_allclose(trained(probe, training=False).numpy(), inference(probe, training=False).numpy(), atol=1e-5)
    inference.save(destination, include_optimizer=False)
    reloaded = load_h5(destination)
    np.testing.assert_allclose(inference(probe, training=False).numpy(), reloaded(probe, training=False).numpy(), atol=1e-5)
    return reloaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/cleaned"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--custom-epochs", type=int, default=8)
    parser.add_argument("--transfer-epochs", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Reuse completed model results and HDF5 exports")
    args = parser.parse_args()
    if not (args.data / "manifest.csv").exists():
        parser.error("Cleaned dataset missing. Run scripts/prepare_dataset.py first.")
    args.models.mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(4)

    train_ds = dataset(args.data, "train", args.batch_size, True)
    valid_ds = dataset(args.data, "valid", args.batch_size, False)
    test_ds = dataset(args.data, "test", args.batch_size, False)
    result_path = args.models / "training_results.json"
    results: dict[str, dict] = json.loads(result_path.read_text()) if args.resume and result_path.exists() else {}
    cv_path = Path("reports/deep_cv.json")
    cv_models = json.loads(cv_path.read_text()).get("models", {}) if cv_path.exists() else {}
    for name in ("custom_cnn", "mobilenet_v2", "efficientnet_b0"):
        if name in results and (args.models / f"{name}.h5").exists():
            print(f"Reusing completed {name}", flush=True)
            continue
        print(f"\nTraining {name}", flush=True)
        tf.keras.backend.clear_session()
        model = custom_cnn() if name == "custom_cnn" else transfer_model(name)
        learning_rate = cv_models.get(name, {}).get("selected_learning_rate", 1e-3 if name == "custom_cnn" else 3e-4)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        checkpoint = args.models / f"{name}_best.keras"
        start = time.monotonic()
        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=args.custom_epochs if name == "custom_cnn" else args.transfer_epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True),
            ],
            shuffle=False,
            verbose=2,
        )
        elapsed = time.monotonic() - start
        model = tf.keras.models.load_model(checkpoint, compile=False)
        model = export_h5(name, model, args.models / f"{name}.h5")
        hist = {key: [float(x) for x in values] for key, values in history.history.items()}
        plot_history(hist, Path("reports/figures") / f"{name}_training.png")
        y_valid, pred_valid, _ = predictions(model, valid_ds)
        results[name] = {
            "validation": scores(y_valid, pred_valid),
            "history": hist,
            "training_seconds": round(elapsed, 1),
            "parameters": int(model.count_params()),
            "learning_rate": learning_rate,
        }
        result_path.write_text(json.dumps(results, indent=2) + "\n")
        print(name, "validation macro F1", results[name]["validation"]["f1_macro"], flush=True)

    # Selection uses validation only. The test set is read for scoring after
    # selection and does not influence model choice or hyperparameters.
    selected = max(results, key=lambda name: (results[name]["validation"]["f1_macro"], results[name]["validation"]["accuracy"]))
    for name in results:
        model = load_h5(args.models / f"{name}.h5")
        y_test, pred_test, _ = predictions(model, test_ds)
        results[name]["test"] = scores(y_test, pred_test)
    results["selected_model"] = selected
    result_path.write_text(json.dumps(results, indent=2) + "\n")
    metadata = {"model": selected, "model_path": f"{selected}.h5", "classes": CLASSES, "image_size": list(IMAGE_SIZE)}
    (args.models / "deployment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Selected:", selected, "test macro F1:", results[selected]["test"]["f1_macro"], flush=True)


if __name__ == "__main__":
    main()
