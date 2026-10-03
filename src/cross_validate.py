"""Group-aware cross-validation and hyperparameter tuning baseline.

Uses simple image measurements, never test images. This is a low-cost baseline
for the submission template's CV requirement, not a replacement for CNNs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["brightness", "contrast", "entropy", "edge_strength", "center_brightness", "border_brightness", "red_mean", "green_mean", "blue_mean"]


def main() -> None:
    df = pd.read_csv("reports/image_features.csv")
    train = df[df["split"] == "train"]
    valid = df[df["split"] == "valid"]
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("classify", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    search = GridSearchCV(
        pipeline,
        param_grid={"classify__C": [0.1, 1.0, 10.0]},
        scoring="f1_macro",
        cv=cv,
        refit=True,
        n_jobs=1,
        return_train_score=True,
    )
    groups = train["class"] + "/" + train["source_id"]
    search.fit(train[FEATURES], train["class"], groups=groups)
    prediction = search.predict(valid[FEATURES])
    result = {
        "model": "standardized image features + balanced logistic regression",
        "cv": "3-fold StratifiedGroupKFold on training source IDs",
        "search_values": [0.1, 1.0, 10.0],
        "best_C": search.best_params_["classify__C"],
        "cv_macro_f1": float(search.best_score_),
        "cv_by_C": {str(p["classify__C"]): float(score) for p, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])},
        "validation": {
            "accuracy": float(accuracy_score(valid["class"], prediction)),
            "precision_macro": float(precision_score(valid["class"], prediction, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(valid["class"], prediction, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(valid["class"], prediction, average="macro", zero_division=0)),
        },
    }
    Path("reports/cv_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
