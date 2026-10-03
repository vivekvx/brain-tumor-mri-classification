# Brain Tumor MRI Image Classification

Repository: https://github.com/vivekvx/brain-tumor-mri-classification

Four-class MRI image classification project using a custom CNN, MobileNetV2,
EfficientNetB0, and a Streamlit interface. The classes are `glioma`,
`meningioma`, `no_tumor`, and `pituitary`.

**Research demonstration only.** This model is not clinically validated and
must not be used for diagnosis or patient triage. The dataset lacks patient
identifiers, so patient-level independence cannot be established.

## Dataset and split

The source is the [Labeled MRI Brain Tumor Dataset on Roboflow](https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset), supplied as a ZIP export. Its README states a CC BY 4.0 license. The supplied split had source-image variants crossing train, validation, and test. `scripts/prepare_dataset.py` creates a deterministic source-image-disjoint split and validates the class CSVs. See [the split report](reports/dataset_split.md).

The source ZIP and cleaned images are omitted from Git. Place the ZIP locally
and run:

```bash
python3 scripts/prepare_dataset.py --source /path/to/drive-download.zip
```

This creates `data/cleaned/{train,valid,test}/{class}/`, class CSVs, a full
SHA-256 manifest, and a JSON split report. The script will not overwrite an
existing output directory.

## Setup

Use Python 3.11 on macOS, Linux, or Windows:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.
On Apple Silicon, the conditional `tensorflow-metal` dependency enables GPU training.

## Reproduce the analysis and train models

```bash
python -m src.eda
python -m src.cross_validate
python -m src.deep_cv
python -m src.train
python -m src.fine_tune
```

EDA writes `reports/image_features.csv`, 15 numbered charts, and
`reports/hypothesis_tests.json`. Only training images are used for image-feature
exploration and hypothesis tests. The training script uses 160 × 160 RGB inputs,
moderate training-only augmentation, early stopping, and best-model checkpoints.
It saves all three models as `.h5` and selects one by validation macro F1. The
held-out test set is scored only after that selection. See
`models/training_results.json` and `models/deployment.json` for metrics and
selection.
`src.cross_validate` runs a three-fold, source-grouped hyperparameter search
for a simple image-feature baseline, providing a check against the CNN scores.
`src.deep_cv` runs a short, three-fold source-grouped learning-rate screen for
each deep architecture. Run it before full training so `src.train` picks those
rates. `src.fine_tune` unfreezes the top MobileNetV2 layers and compares that
candidate on validation before updating the deployment choice.

## Streamlit

Live app: https://tumor-mri-classification.streamlit.app/

```bash
streamlit run app.py
```

Upload a JPG or PNG MRI image. The app displays the predicted class, the
model score, and scores for all four classes. The score is not a calibrated
clinical probability. Deployment requires the selected `.h5` file and
`models/deployment.json` alongside `app.py`; no training data is needed at
inference time.
For Streamlit Community Cloud, use Python 3.11 and `app.py` as the entry
point; the root `requirements.txt` supplies dependencies. The packaged ZIP
contains a smaller deployment-only `requirements.txt` and the selected model.

## Notebook

`Brain_Tumor_MRI_Project.ipynb` is the submission notebook. It documents the
problem, dataset, 15 charts, three statistical tests, preprocessing, training,
model comparison, test evaluation, and how to launch the Streamlit app. It
calls the same scripts as the command-line workflow so the analysis and app
use the same model and label order.

## Evaluation limits

The source-image grouping uses the filename prefix before `_jpg.rf.`. This
prevents known augmented variants from crossing splits, but does not prove
that different source IDs belong to different patients. Image-level metrics
can overestimate performance on new patients or scanners. External validation
and clinician review are required before any clinical use.
