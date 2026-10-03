"""Generate 15 image-focused EDA charts and three source-level tests."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy.stats import kruskal


ROOT = Path("data/cleaned")
FIGURES = Path("reports/figures")
FEATURES = Path("reports/image_features.csv")
PALETTE = "Set2"
CLASS_ORDER = ["glioma", "meningioma", "no_tumor", "pituitary"]


def save(number: int, name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / f"{number:02d}_{name}.png", dpi=140, bbox_inches="tight")
    plt.close()


def extract_features(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in manifest.to_dict("records"):
        path = ROOT / row["new_split"] / row["class"] / row["filename"]
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
        gray = rgb.mean(axis=2)
        hist = np.histogram(gray, bins=32, range=(0, 1))[0].astype(float)
        prob = hist[hist > 0] / hist.sum()
        rows.append({
            "filename": row["filename"],
            "split": row["new_split"],
            "class": row["class"],
            "source_id": row["source_id"],
            "file_kb": path.stat().st_size / 1024,
            "brightness": float(gray.mean()),
            "contrast": float(gray.std()),
            "entropy": float(-(prob * np.log2(prob)).sum()),
            "edge_strength": float((np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()) / 2),
            "center_brightness": float(gray[16:48, 16:48].mean()),
            "border_brightness": float(np.concatenate((gray[:8].ravel(), gray[-8:].ravel(), gray[:, :8].ravel(), gray[:, -8:].ravel())).mean()),
            "red_mean": float(rgb[:, :, 0].mean()),
            "green_mean": float(rgb[:, :, 1].mean()),
            "blue_mean": float(rgb[:, :, 2].mean()),
        })
    return pd.DataFrame(rows)


def make_charts(df: pd.DataFrame, manifest: pd.DataFrame) -> None:
    train = df[df["split"] == "train"].copy()
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 4))
    sns.countplot(data=train, x="class", order=CLASS_ORDER, palette=PALETTE, hue="class", legend=False)
    plt.title("1. Training images per class"); save(1, "train_class_counts")

    ct = pd.crosstab(manifest["new_split"], manifest["class"]).reindex(["train", "valid", "test"])[CLASS_ORDER]
    ct.div(ct.sum(axis=1), axis=0).plot(kind="bar", stacked=True, figsize=(7, 4), colormap="Set2")
    plt.ylabel("Proportion"); plt.title("2. Class mix by split"); save(2, "split_class_mix")

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, cls in enumerate(CLASS_ORDER):
        examples = train[train["class"] == cls].sample(2, random_state=42)
        for j, row in enumerate(examples.itertuples()):
            image = Image.open(ROOT / "train" / cls / row.filename)
            axes[j, i].imshow(image)
            axes[j, i].set_title(cls)
            axes[j, i].axis("off")
    fig.suptitle("3. Sample MRI images by class"); save(3, "class_examples")

    for num, column, title in [
        (4, "file_kb", "JPEG file size by class"),
        (5, "brightness", "Mean pixel brightness by class"),
        (6, "contrast", "Pixel contrast by class"),
        (7, "entropy", "Pixel entropy by class"),
        (8, "edge_strength", "Edge strength by class"),
        (9, "center_brightness", "Center brightness by class"),
    ]:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=train, x="class", y=column, order=CLASS_ORDER, hue="class", palette=PALETTE, legend=False, showfliers=False)
        plt.title(f"{num}. {title}"); save(num, column)

    means = train.groupby("class")[["red_mean", "green_mean", "blue_mean"]].mean().reindex(CLASS_ORDER)
    means.plot(kind="bar", figsize=(8, 4), color=["#b95b5b", "#67a46b", "#5979b5"])
    plt.ylabel("Mean channel value (0–1)"); plt.title("10. RGB channel means"); save(10, "channel_means")

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=train, x="brightness", y="contrast", hue="class", hue_order=CLASS_ORDER, alpha=.5, s=20)
    plt.title("11. Brightness versus contrast"); save(11, "brightness_contrast")

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=train, x="brightness", y="edge_strength", hue="class", hue_order=CLASS_ORDER, alpha=.5, s=20)
    plt.title("12. Brightness versus edge strength"); save(12, "brightness_edges")

    plt.figure(figsize=(7, 5))
    sns.heatmap(train[["brightness", "contrast", "entropy", "edge_strength", "center_brightness", "border_brightness", "file_kb"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("13. Image-feature correlation"); save(13, "feature_correlation")

    sample = train.sample(min(400, len(train)), random_state=42)
    grid = sns.pairplot(sample, vars=["brightness", "contrast", "edge_strength"], hue="class", hue_order=CLASS_ORDER, corner=True, plot_kws={"s": 10, "alpha": .5})
    grid.fig.suptitle("14. Image-feature pair plot", y=1.02)
    grid.fig.savefig(FIGURES / "14_feature_pairplot.png", dpi=140, bbox_inches="tight")
    plt.close(grid.fig)

    family = train.groupby(["class", "source_id"]).size().rename("variants").reset_index()
    plt.figure(figsize=(8, 4))
    sns.countplot(data=family, x="class", hue="variants", order=CLASS_ORDER)
    plt.title("15. Number of variants per source image"); save(15, "source_variants")


def hypothesis_tests(df: pd.DataFrame) -> dict:
    # Collapse augmented variants so each named source image contributes once.
    train = df[df["split"] == "train"]
    source = train.groupby(["class", "source_id"], as_index=False)[["brightness", "edge_strength", "entropy"]].mean()
    results = {}
    for feature in ("brightness", "edge_strength", "entropy"):
        groups = [source[source["class"] == cls][feature].to_numpy() for cls in CLASS_ORDER]
        stat, pvalue = kruskal(*groups)
        results[feature] = {
            "null": f"The distribution of {feature} is the same across all four classes.",
            "alternative": f"At least one class has a different {feature} distribution.",
            "test": "Kruskal-Wallis on source-level means",
            "statistic": float(stat),
            "p_value": float(pvalue),
            "class_medians": {cls: float(np.median(values)) for cls, values in zip(CLASS_ORDER, groups)},
            "source_group_counts": {cls: len(values) for cls, values in zip(CLASS_ORDER, groups)},
        }
    # Holm adjustment controls family-wise error for the three planned tests.
    ordered = sorted(results, key=lambda name: results[name]["p_value"])
    adjusted = 0.0
    for rank, feature in enumerate(ordered):
        adjusted = max(adjusted, min(1.0, results[feature]["p_value"] * (len(ordered) - rank)))
        results[feature]["p_holm"] = adjusted
    return results


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(ROOT / "manifest.csv")
    df = extract_features(manifest)
    df.to_csv(FEATURES, index=False)
    make_charts(df, manifest)
    tests = hypothesis_tests(df)
    Path("reports/hypothesis_tests.json").write_text(json.dumps(tests, indent=2) + "\n")
    print(f"Saved {len(list(FIGURES.glob('[0-9][0-9]_*.png')))} EDA charts and three hypothesis tests.")
    print(json.dumps(tests, indent=2))


if __name__ == "__main__":
    main()
