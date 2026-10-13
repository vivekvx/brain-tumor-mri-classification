"""Build the executed-results-ready submission notebook from project outputs."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import pandas as pd

MODEL_LABELS = {
    "custom_cnn": "Custom CNN",
    "mobilenet_v2": "MobileNetV2",
    "efficientnet_b0": "EfficientNetB0",
    "mobilenet_v2_finetuned": "MobileNetV2 fine-tuned",
}


SUMMARY = """This project studies four-class classification of brain MRI images into glioma, meningioma, pituitary tumor, and no tumor. The supplied Roboflow export contains 2,443 JPEG images with predefined training, validation, and test directories. The project brief asks for a custom convolutional neural network, transfer learning, model comparison, and a Streamlit application that predicts a class from an uploaded image. This notebook connects those steps in a reproducible workflow and keeps the same class order from data loading through deployment.

Before modeling, we audited the archive and found a serious split problem: multiple image variants carrying the same source-image identifier appeared in different supplied splits, especially for glioma. A random image-level split would let a model see a variant of an evaluation image during training. We therefore reconstructed the split at the source-image level, preserving all images and approximately the original train, validation, and test proportions. The cleaned dataset has 1,696 training, 500 validation, and 247 test images. Every image is listed in a manifest with its original split, new split, source ID, and SHA-256 digest. No named source image crosses the new splits. Patient identifiers are unavailable, so this does not guarantee patient-level separation.

Exploratory analysis focuses on the training split. Fifteen charts examine class balance, examples, pixel brightness, contrast, entropy, edges, color channels, feature relationships, and augmentation multiplicity. Three planned Kruskal-Wallis tests compare source-level brightness, edge strength, and entropy across classes with Holm correction. These image statistics can reveal dataset or acquisition differences; a significant p-value does not prove a medically meaningful tumor feature. A simple balanced logistic-regression baseline uses nine image measurements and three-fold stratified group cross-validation to tune its regularization strength. It provides context for the deep models, but it cannot capture the spatial detail available to a CNN.

The main experiments train a custom CNN with convolution, batch normalization, pooling, and dropout, plus MobileNetV2 and EfficientNetB0 classifiers initialized with ImageNet weights. The transfer backbones are frozen for the initial experiments, followed by a small MobileNetV2 fine-tuning pass. Images are resized to 160 by 160 pixels. Each model includes the preprocessing appropriate to its backbone, while moderate random transformations are applied only during training. A short source-grouped cross-validation screen compares two learning rates per deep model. Early stopping monitors validation loss, and the best checkpoint is saved. The models are compared with validation accuracy, macro precision, macro recall, and macro F1. Macro scores give each class equal weight. The deployment choice is fixed using validation macro F1 before the held-out test set is evaluated.

The Streamlit application loads the selected HDF5 model, accepts a JPG or PNG MRI image, and displays the predicted class and four model scores. An occlusion view offers a limited explanation of one prediction by showing which image regions change its score when hidden. The application and deployment bundle are included, but their outputs are research demonstrations. The dataset has no patient IDs or external hospital cohort, the labels have not been independently verified, and the model scores are not calibrated clinical probabilities. Interpretation remains cautious because image labels alone cannot establish clinical validity. External validation, subgroup checks, and specialist review would be necessary before considering use in healthcare decisions."""


def main() -> None:
    assert 500 <= len(SUMMARY.split()) <= 600, len(SUMMARY.split())
    results = json.loads(Path("models/training_results.json").read_text())
    deep_cv_results = json.loads(Path("reports/deep_cv.json").read_text())
    tests = json.loads(Path("reports/hypothesis_tests.json").read_text())
    cv = json.loads(Path("reports/cv_results.json").read_text())
    df = pd.read_csv("reports/image_features.csv")
    train = df[df["split"] == "train"]
    med = train.groupby("class")[["brightness", "contrast", "entropy", "edge_strength", "center_brightness", "file_kb"]].median()
    corr = train[["brightness", "contrast", "edge_strength"]].corr()
    selected = results["selected_model"]
    cells = []
    def md(value: str) -> None:
        cells.append(nbf.v4.new_markdown_cell(value))
    def code(value: str) -> None:
        cells.append(nbf.v4.new_code_cell(value))

    md("# Brain Tumor MRI Image Classification\n\n**Project type:** Multiclass image classification  \n**Contribution:** Individual  \n**Domain:** Medical imaging research")
    md("## Project summary (500–600 words)\n\n" + SUMMARY)
    md("## Problem statement and GitHub\n\nBuild and compare a custom CNN and pretrained image classifiers for four MRI categories, then provide a Streamlit upload interface. This is an educational research project, not a diagnostic tool.\n\n**GitHub repository:** https://github.com/vivekvx/brain-tumor-mri-classification")
    md("**Potential use cases after clinical validation:** radiologist decision support, specialist triage, research cohort grouping, and remote second opinions. None is established by this image-level benchmark; false negatives, scanner shifts, and label quality must be assessed with patient-level external data first.")
    md("## 1. Know your data\n\nThe source is a Roboflow JPEG export with four folders and one-hot CSV labels. The following cell validates that the cleaned split is present, preparing it from `BRAIN_MRI_ZIP` if needed. Set that environment variable to the source ZIP path on a new machine.")
    code("""import os, sys, subprocess, json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Image

if not Path('data/cleaned/manifest.csv').exists():
    source_zip = os.environ.get('BRAIN_MRI_ZIP')
    if not source_zip:
        raise FileNotFoundError('Set BRAIN_MRI_ZIP to the supplied dataset ZIP before running this notebook.')
    subprocess.run([sys.executable, 'scripts/prepare_dataset.py', '--source', source_zip], check=True)
manifest = pd.read_csv('data/cleaned/manifest.csv')
display(pd.crosstab(manifest['class'], manifest['new_split']))
print('Rows:', len(manifest), 'classes:', sorted(manifest['class'].unique()))
print('Missing values:', int(manifest.isna().sum().sum()))
print('Duplicate output paths:', int(manifest.duplicated(['new_split', 'class', 'filename']).sum()))
print('Source groups crossing splits:', int((manifest.groupby(['class','source_id'])['new_split'].nunique() > 1).sum()))""")
    md("**Data understanding and wrangling.** Every image is 640 × 640 RGB before modeling; all files opened successfully during audit. The supplied CSV labels matched their folders. Duplicate augmented variants are kept together in the new split. Missing tabular values, text cleaning, categorical feature encoding, and outlier removal are not applicable to the raw MRI image input. The class label is encoded by the folder loader as an integer from 0 to 3. Source-image grouping reduces known leakage, although patient-level grouping is impossible without patient IDs.")
    md("## 2. Exploratory analysis: 15 charts\n\nThe charts below use the training images for image features. Split counts use the manifest. Each chart has a purpose, observation, and practical implication; observations about pixel statistics may reflect scanner or preparation differences.")
    code("""if not Path('reports/hypothesis_tests.json').exists():
    subprocess.run([sys.executable, '-m', 'src.eda'], check=True)
features = pd.read_csv('reports/image_features.csv')
display(features.head())""")

    chart_info = [
        (1, "train_class_counts", "A count chart makes training class balance visible.", "Glioma has 560 images and no tumor has 335; class sizes differ but none is absent.", "Macro metrics and per-class recall are needed so the largest class does not dominate evaluation."),
        (2, "split_class_mix", "Stacked proportions check the reconstructed split.", "The four class shares remain similar across train, validation, and test.", "Comparable split composition supports a fairer validation comparison; source-level leakage is checked separately."),
        (3, "class_examples", "A montage reveals visual diversity that summary statistics miss.", "Orientation and image appearance vary within classes.", "Augmentation and external validation matter when scans may come from different acquisition settings."),
        (4, "file_kb", "File size can expose compression or image-source differences.", f"Median training JPEG size ranges from {med['file_kb'].min():.1f} to {med['file_kb'].max():.1f} KB across classes.", "A classifier must not rely on source or compression artifacts as a shortcut."),
        (5, "brightness", "Boxplots compare the image brightness distributions.", f"Median brightness is {med.loc['glioma','brightness']:.3f} for glioma and {med.loc['no_tumor','brightness']:.3f} for no tumor.", "Large brightness differences may reflect acquisition or preprocessing; robustness checks are needed."),
        (6, "contrast", "Contrast summarizes variation inside each image.", f"Class median contrast spans {med['contrast'].min():.3f}–{med['contrast'].max():.3f}.", "Contrast variation can affect model generalization to other scanners."),
        (7, "entropy", "Entropy captures how spread out pixel intensities are.", f"Class median entropy spans {med['entropy'].min():.2f}–{med['entropy'].max():.2f} bits.", "An entropy difference is a dataset signal, not evidence of a clinical imaging marker."),
        (8, "edge_strength", "Edge strength measures local intensity transitions.", f"Class median edge strength spans {med['edge_strength'].min():.3f}–{med['edge_strength'].max():.3f}.", "Different edge profiles may reveal scanner or resolution effects that require validation."),
        (9, "center_brightness", "Center brightness checks where signal is concentrated.", f"Class median center brightness spans {med['center_brightness'].min():.3f}–{med['center_brightness'].max():.3f}.", "A shortcut based on framing would be unreliable outside this collection."),
        (10, "channel_means", "Channel means check whether these RGB files contain meaningful color.", "The three channels are similar because the source scans are visually close to grayscale.", "The app still accepts RGB files and uses the same three-channel model input."),
        (11, "brightness_contrast", "A scatterplot exposes joint distributions and outliers.", f"Brightness and contrast have correlation {corr.loc['brightness','contrast']:.2f}; classes overlap.", "A single image statistic cannot replace a spatial image model."),
        (12, "brightness_edges", "This view checks whether brighter images also have stronger edges.", f"Brightness and edge strength have correlation {corr.loc['brightness','edge_strength']:.2f}.", "Correlated low-level signals could enable shortcut learning; model validation should cover new sources."),
        (13, "feature_correlation", "A heatmap summarizes redundant image measurements.", "Several low-level features are correlated, so they are not independent sources of evidence.", "The feature baseline uses scaling and cross-validation rather than interpreting coefficients clinically."),
        (14, "feature_pairplot", "Pairwise plots show class overlap across three measurements.", "The classes overlap in this low-dimensional feature view.", "CNNs can use spatial detail that summary features discard, but still require held-out testing."),
        (15, "source_variants", "Source multiplicity reveals augmentation and split risks.", "Glioma contains repeated variants of named source images; the other classes mostly contain one variant each.", "Grouping by source ID prevents known variants from leaking into validation or test."),
    ]
    for number, slug, why, insight, impact in chart_info:
        md(f"### Chart {number}: {slug.replace('_', ' ').title()}\n\n- **Why this chart?** {why}\n- **Insight:** {insight}\n- **Potential impact or risk:** {impact}")
        code(f"display(Image(filename='reports/figures/{number:02d}_{slug}.png'))")

    md("## 3. Hypothesis testing\n\nThe three tests were specified for training data. Augmented variants are averaged within each named source image before testing. Kruskal–Wallis compares class distributions without assuming normality; Holm adjustment accounts for three planned tests. These are exploratory dataset comparisons, not clinical conclusions.")
    code("""tests = json.loads(Path('reports/hypothesis_tests.json').read_text())
display(pd.DataFrame([{'feature': key, 'H statistic': value['statistic'], 'p value': value['p_value'], 'Holm p': value['p_holm']} for key, value in tests.items()]))""")
    for idx, feature in enumerate(("brightness", "edge_strength", "entropy"), 1):
        test = tests[feature]
        conclusion = "Reject the null at 0.05" if test["p_holm"] < .05 else "Do not reject the null at 0.05"
        md(f"### Hypothesis {idx}: {feature.replace('_', ' ').title()}\n\n- **H₀:** {test['null']}\n- **H₁:** {test['alternative']}\n- **Test and reason:** Kruskal–Wallis on source-level means accommodates four independent groups without requiring normality.\n- **Result:** H={test['statistic']:.2f}, p={test['p_value']:.3g}, Holm-adjusted p={test['p_holm']:.3g}. {conclusion}. Class differences may reflect acquisition conditions and cannot establish tumor biology.")

    md("## 4. Preprocessing and feature engineering\n\nImages are resized to 160 × 160 and loaded as RGB. The custom CNN rescales pixels to 0–1. MobileNetV2 maps raw pixels to −1–1 using its expected ImageNet preprocessing. EfficientNetB0 includes its own rescaling. Training-only augmentation uses modest horizontal flips, rotation, zoom, shifts, and brightness changes. Vertical flips are excluded because they could reverse anatomical orientation. Batch normalization and dropout reduce training instability and overfitting. The largest training class has 560 images and the smallest has 335, so we report macro metrics without synthetic oversampling or deep-model class weighting. No tabular imputation, NLP preprocessing, manual feature selection, or dimensionality reduction is applied to the CNN inputs. The source-disjoint train/validation/test split is retained; the test set is never used for tuning.")
    md("## 5. Group cross-validation and hyperparameter tuning baseline\n\nA standardized image-feature logistic regression provides a lightweight baseline. Three-fold `StratifiedGroupKFold` keeps source IDs intact, and grid search compares C = 0.1, 1, and 10 using macro F1. This baseline is tuned only on training data and scored once on validation.")
    code("""if not Path('reports/cv_results.json').exists():
    subprocess.run([sys.executable, '-m', 'src.cross_validate'], check=True)
cv = json.loads(Path('reports/cv_results.json').read_text())
display(pd.DataFrame({'C': list(cv['cv_by_C']), 'CV macro F1': list(cv['cv_by_C'].values())}))
print('Selected C:', cv['best_C'], 'CV macro F1:', round(cv['cv_macro_f1'], 3), 'validation macro F1:', round(cv['validation']['f1_macro'], 3))""")
    md(f"The best C was **{cv['best_C']}**. The three-fold mean macro F1 was **{cv['cv_macro_f1']:.3f}** and validation macro F1 was **{cv['validation']['f1_macro']:.3f}**. The gain over the weakest C candidate is {cv['cv_macro_f1'] - min(cv['cv_by_C'].values()):.3f}. This baseline's strong score also warns that low-level acquisition cues may correlate with labels.")
    md("## 6. Deep-learning model implementation and comparison\n\nThe custom CNN uses three convolution blocks, batch normalization, max pooling, a dense layer, and dropout. MobileNetV2 and EfficientNetB0 use frozen ImageNet backbones with new dense classification heads. A three-fold source-grouped screen compares two learning rates per model using two epochs per fold. This is a coarse tuning screen, not a full convergence estimate. Final models then train longer with early stopping and best-checkpoint selection. Architecture comparison is performed on validation macro F1.")
    code("""if not Path('reports/deep_cv.json').exists():
    subprocess.run([sys.executable, '-m', 'src.deep_cv'], check=True)
deep_cv = json.loads(Path('reports/deep_cv.json').read_text())
deep_table = pd.DataFrame([{'model': name, 'learning rate': rate, '3-fold macro F1': stats['mean_f1']} for name, model in deep_cv['models'].items() for rate, stats in model['candidates'].items()])
display(deep_table)
sns.barplot(data=deep_table, x='model', y='3-fold macro F1', hue='learning rate')
plt.xticks(rotation=20); plt.tight_layout(); plt.show()
print('Chosen learning rates:', {name: model['selected_learning_rate'] for name, model in deep_cv['models'].items()})""")
    tuning_notes = []
    for name, model in deep_cv_results["models"].items():
        candidates = model["candidates"]
        chosen_rate = model["selected_learning_rate"]
        chosen_score = candidates[str(chosen_rate)]["mean_f1"]
        alternative_score = min(candidate["mean_f1"] for candidate in candidates.values())
        tuning_notes.append(
            f"- **{MODEL_LABELS[name]}:** learning rate {chosen_rate:g} "
            f"achieved grouped-CV macro F1 {chosen_score:.3f}, an improvement of "
            f"{chosen_score - alternative_score:.3f} over the weaker tested rate."
        )
    md("**Learning-rate tuning outcome.** The chart and table above compare both rates for each architecture. "
       "The two-epoch folds are a screening experiment, so these gains should not be read as final-model gains.\n\n"
       + "\n".join(tuning_notes))
    code("""if not Path('models/deployment.json').exists():
    subprocess.run([sys.executable, '-m', 'src.train'], check=True)
if not Path('models/mobilenet_v2_finetuned.h5').exists():
    subprocess.run([sys.executable, '-m', 'src.fine_tune'], check=True)
results = json.loads(Path('models/training_results.json').read_text())
names = ['custom_cnn','mobilenet_v2','efficientnet_b0','mobilenet_v2_finetuned']
comparison = pd.DataFrame([{'model': name, **{k: results[name]['validation'][k] for k in ['accuracy','precision_macro','recall_macro','f1_macro']}, 'parameters': results[name]['parameters'], 'training_seconds': results[name]['training_seconds']} for name in names])
display(comparison.sort_values('f1_macro', ascending=False))
comparison.set_index('model')[['accuracy','precision_macro','recall_macro','f1_macro']].plot(kind='bar', figsize=(10,4), ylim=(0,1), title='Validation metric comparison')
plt.ylabel('Score'); plt.xticks(rotation=20); plt.tight_layout(); plt.show()
print('Selected for deployment:', results['selected_model'])""")
    md("**Why these evaluation metrics matter.** Accuracy describes the overall share of correct images but can hide class-specific errors. "
       "Precision measures how often a predicted class is correct; low precision could cause unnecessary follow-up. "
       "Recall measures how many examples of a class are found; low tumor-class recall could mean missed cases. "
       "Macro F1 balances precision and recall while giving each class equal weight. These are research metrics, "
       "not evidence of clinical benefit or patient-level performance.")
    for name in ("custom_cnn", "mobilenet_v2", "efficientnet_b0", "mobilenet_v2_finetuned"):
        validation = results[name]["validation"]
        improvement = ""
        if name == "mobilenet_v2_finetuned":
            delta = validation["f1_macro"] - results["mobilenet_v2"]["validation"]["f1_macro"]
            improvement = f" Fine-tuning improved validation macro F1 by {delta:.3f} over frozen MobileNetV2."
        md(f"### {MODEL_LABELS[name]}\n\n"
           f"Validation accuracy was **{validation['accuracy']:.3f}** and macro F1 was **{validation['f1_macro']:.3f}**."
           f"{improvement} Training and validation accuracy/loss are shown below. The model's `.h5` file is saved under `models/`. "
           "The comparison chart above shows all four evaluation metrics; the confusion matrix below shows class-level errors.")
        code(f"display(Image(filename='reports/figures/{name}_training.png'))")
    md("### Held-out test evaluation\n\nSelection was fixed from validation macro F1. The following test metrics and confusion matrices are reported only after model selection. A false negative for a tumor class or a false positive for no tumor may have different consequences; neither can be judged clinically from this dataset alone.")
    code("""test_scores = pd.DataFrame([{'model': name, **{k: results[name]['test'][k] for k in ['accuracy','precision_macro','recall_macro','f1_macro']}} for name in names])
display(test_scores)
fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4))
for ax, name in zip(axes, names):
    sns.heatmap(results[name]['test']['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['glioma','meningioma','no_tumor','pituitary'], yticklabels=['glioma','meningioma','no_tumor','pituitary'])
    ax.set(title=name, xlabel='Predicted', ylabel='True')
plt.tight_layout(); plt.show()""")
    chosen = results[selected]["test"]
    md(f"The selected model is **{selected}**, chosen from validation macro F1. Its held-out test accuracy is **{chosen['accuracy']:.3f}**, macro precision **{chosen['precision_macro']:.3f}**, macro recall **{chosen['recall_macro']:.3f}**, and macro F1 **{chosen['f1_macro']:.3f}**. The comparison considers class-balanced performance as well as parameter count and training time. These results are image-level and require external patient-level validation before any clinical claim.")
    md("## 7. Model explanation\n\nOcclusion sensitivity hides small image patches and measures the change in the selected class score. The map shows which regions influenced one prediction; it is not a lesion segmentation or proof that the model used a medically appropriate feature.")
    code("""if not Path('reports/figures/16_occlusion_explanation.png').exists():
    subprocess.run([sys.executable, '-m', 'src.explain'], check=True)
display(Image(filename='reports/figures/16_occlusion_explanation.png'))""")
    md("## 8. Streamlit deployment\n\n`app.py` loads `models/deployment.json` and the selected `.h5` file. The **Try the model** tab accepts JPG/PNG uploads, displays the predicted class and four scores, and handles invalid files. **Model results** shows held-out metrics, model comparison, training curves, and the confusion matrix; **Explore the data** displays the 15 EDA charts. The app carries a research-use warning. The next cell builds a portable deployment ZIP. On a local machine, run `streamlit run app.py`; on Streamlit Community Cloud, publish the repository and set `app.py` as the entry point.")
    code("""subprocess.run([sys.executable, 'scripts/package_deployment.py'], check=True)
print('Local launch: streamlit run app.py')
print('Bundle: deployment/brain_mri_streamlit.zip')""")
    md("## Conclusion and future work\n\nThe project delivers source-disjoint splits, 15 EDA charts, three planned statistical tests, a grouped-CV baseline, a custom CNN, two ImageNet transfer models, validation-based model choice, final test metrics, an explanation view, and a Streamlit interface. The strongest limitation is the lack of patient IDs and external validation. Future work should obtain patient-level splits, independent hospital data, label review, scanner-subgroup analysis, probability calibration, and prospective clinical evaluation. The saved Keras `.h5` files replace the generic template's pickle/joblib suggestion for deployment.")

    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3 (brain MRI)", "language": "python", "name": "python3"}, "language_info": {"name": "python"}})
    destination = Path("Brain_Tumor_MRI_Project.ipynb")
    nbf.write(notebook, destination)
    print(destination, len(cells), "cells; summary words:", len(SUMMARY.split()))


if __name__ == "__main__":
    main()
