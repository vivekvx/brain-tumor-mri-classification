"""Model-agnostic occlusion view of one held-out MRI image."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from src.inference import load_h5


def main() -> None:
    metadata = json.loads(Path("models/deployment.json").read_text())
    model = load_h5(Path("models") / metadata["model_path"])
    cls = "glioma"
    path = sorted((Path("data/cleaned/test") / cls).glob("*.jpg"))[0]
    image = Image.open(path).convert("RGB").resize(tuple(metadata["image_size"]))
    pixels = np.asarray(image, dtype=np.float32)
    base = model.predict(pixels[None, ...], verbose=0)[0]
    label = int(np.argmax(base))
    grid = 8
    step = pixels.shape[0] // grid
    variants = []
    for y in range(grid):
        for x in range(grid):
            altered = pixels.copy()
            altered[y * step:(y + 1) * step, x * step:(x + 1) * step] = 0
            variants.append(altered)
    scores = model.predict(np.asarray(variants), batch_size=16, verbose=0)[:, label]
    importance = (base[label] - scores).reshape(grid, grid)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(image)
    axes[0].set_title("Held-out input")
    axes[1].imshow(image)
    heat = axes[1].imshow(importance, cmap="magma", alpha=.55, extent=(0, pixels.shape[1], pixels.shape[0], 0))
    axes[1].set_title(f"Occlusion effect on {metadata['classes'][label]} score")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(heat, ax=axes[1], fraction=.046, pad=.04)
    fig.tight_layout()
    destination = Path("reports/figures/16_occlusion_explanation.png")
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    print(destination, "prediction", metadata["classes"][label], "score", round(float(base[label]), 3))


if __name__ == "__main__":
    main()
