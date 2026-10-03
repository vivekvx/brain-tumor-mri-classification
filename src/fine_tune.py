"""Fine-tune the top MobileNetV2 layers and compare on validation only."""

from __future__ import annotations

import json
import time
from pathlib import Path

import tensorflow as tf

from src.train import CLASSES, SEED, dataset, export_h5, plot_history, predictions, scores


def main() -> None:
    models = Path("models")
    results_path = models / "training_results.json"
    results = json.loads(results_path.read_text())
    tf.keras.utils.set_random_seed(SEED)
    trained = tf.keras.models.load_model(models / "mobilenet_v2_best.keras", compile=False)
    backbone = next(layer for layer in trained.layers if layer.name.startswith("mobilenetv2"))
    backbone.trainable = True
    for layer in backbone.layers[:-20]:
        layer.trainable = False
    for layer in backbone.layers[-20:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    trained.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    training = dataset(Path("data/cleaned"), "train", 16, True)
    validation = dataset(Path("data/cleaned"), "valid", 16, False)
    checkpoint = models / "mobilenet_v2_finetuned_best.keras"
    start = time.monotonic()
    history = trained.fit(
        training,
        validation_data=validation,
        epochs=4,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True),
        ],
        shuffle=False,
        verbose=2,
    )
    elapsed = time.monotonic() - start
    best = tf.keras.models.load_model(checkpoint, compile=False)
    deployed = export_h5("mobilenet_v2", best, models / "mobilenet_v2_finetuned.h5")
    y_valid, pred_valid, _ = predictions(deployed, validation)
    hist = {key: [float(x) for x in values] for key, values in history.history.items()}
    plot_history(hist, Path("reports/figures/mobilenet_v2_finetuned_training.png"))
    results["mobilenet_v2_finetuned"] = {
        "validation": scores(y_valid, pred_valid),
        "history": hist,
        "parameters": int(deployed.count_params()),
        "learning_rate": 1e-5,
        "fine_tuned_layers": 20,
        "training_seconds": round(elapsed, 1),
    }
    candidates = ["custom_cnn", "mobilenet_v2", "efficientnet_b0", "mobilenet_v2_finetuned"]
    selected = max(candidates, key=lambda name: (results[name]["validation"]["f1_macro"], results[name]["validation"]["accuracy"]))
    test = dataset(Path("data/cleaned"), "test", 16, False)
    y_test, pred_test, _ = predictions(deployed, test)
    results["mobilenet_v2_finetuned"]["test"] = scores(y_test, pred_test)
    results["selected_model"] = selected
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    metadata = {"model": selected, "model_path": f"{selected}.h5", "classes": CLASSES, "image_size": [160, 160]}
    (models / "deployment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Fine-tuned validation macro F1:", results["mobilenet_v2_finetuned"]["validation"]["f1_macro"])
    print("Selected:", selected)


if __name__ == "__main__":
    main()
