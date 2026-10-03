"""Check data isolation, saved models, charts, and notebook structure."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import nbformat
import numpy as np
from PIL import Image

from src.inference import load_h5


def main() -> None:
    import csv
    with Path("data/cleaned/manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2443
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        groups[(row["class"], row["source_id"])].add(row["new_split"])
        assert (Path("data/cleaned") / row["new_split"] / row["class"] / row["filename"]).is_file()
    assert all(len(splits) == 1 for splits in groups.values())
    assert all(list(Path("reports/figures").glob(f"{index:02d}_*.png")) for index in range(1, 16))
    assert len(json.loads(Path("reports/hypothesis_tests.json").read_text())) == 3
    results = json.loads(Path("models/training_results.json").read_text())
    metadata = json.loads(Path("models/deployment.json").read_text())
    assert metadata["model"] == results["selected_model"]
    image_path = next(Path("data/cleaned/test/glioma").glob("*.jpg"))
    image = Image.open(image_path).convert("RGB").resize(tuple(metadata["image_size"]))
    batch = np.asarray(image, dtype=np.float32)[None, ...]
    for name in ("custom_cnn", "mobilenet_v2", "efficientnet_b0", "mobilenet_v2_finetuned"):
        model = load_h5(Path("models") / f"{name}.h5")
        prediction = model.predict(batch, verbose=0)[0]
        assert prediction.shape == (4,)
        assert np.isfinite(prediction).all()
        assert abs(float(prediction.sum()) - 1) < 1e-4
        print(name, "inference okay")
    notebook = nbformat.read("Brain_Tumor_MRI_Project.ipynb", as_version=4)
    nbformat.validate(notebook)
    print("Verified", len(rows), "images,", len(groups), "disjoint source groups, 15 charts, 4 models, and notebook schema")


if __name__ == "__main__":
    main()
