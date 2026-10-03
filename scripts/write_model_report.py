"""Write a concise model comparison from measured validation/test results."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    results = json.loads(Path("models/training_results.json").read_text())
    selected = results["selected_model"]
    names = [name for name in ("custom_cnn", "mobilenet_v2", "efficientnet_b0", "mobilenet_v2_finetuned") if name in results]
    lines = [
        "# Model comparison",
        "",
        "Selection used validation macro F1. The test set was evaluated after selection.",
        "",
        "| Model | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 | Parameters |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        item = results[name]
        val, test = item["validation"], item["test"]
        lines.append(f"| {name} | {val['accuracy']:.3f} | {val['f1_macro']:.3f} | {test['accuracy']:.3f} | {test['f1_macro']:.3f} | {item['parameters']:,} |")
    lines += [
        "",
        f"**Selected for Streamlit: `{selected}`.**",
        "",
        "## Per-class test recall of selected model",
        "",
        "| Class | Recall | Support |",
        "|---|---:|---:|",
    ]
    for name in ("glioma", "meningioma", "no_tumor", "pituitary"):
        item = results[selected]["test"]["per_class"][name]
        lines.append(f"| {name} | {item['recall']:.3f} | {item['support']:.0f} |")
    lines += [
        "",
        "Macro F1 weights each class equally. Per-class recall shows where a high overall score hides missed examples. This is an image-level research evaluation, not a clinical accuracy estimate. Patient IDs and an external hospital dataset are unavailable.",
        "",
    ]
    Path("reports/model_comparison.md").write_text("\n".join(lines))
    print("reports/model_comparison.md")


if __name__ == "__main__":
    main()
