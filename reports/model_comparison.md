# Model comparison

Selection used validation macro F1. The test set was evaluated after selection.

| Model | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 | Parameters |
|---|---:|---:|---:|---:|---:|
| custom_cnn | 0.660 | 0.607 | 0.611 | 0.559 | 110,948 |
| mobilenet_v2 | 0.798 | 0.777 | 0.842 | 0.828 | 2,422,468 |
| efficientnet_b0 | 0.758 | 0.744 | 0.785 | 0.778 | 4,214,055 |
| mobilenet_v2_finetuned | 0.812 | 0.799 | 0.854 | 0.846 | 2,422,468 |

**Selected for Streamlit: `mobilenet_v2_finetuned`.**

## Per-class test recall of selected model

| Class | Recall | Support |
|---|---:|---:|
| glioma | 0.852 | 81 |
| meningioma | 0.636 | 55 |
| no_tumor | 0.918 | 49 |
| pituitary | 1.000 | 62 |

Macro F1 weights each class equally. Per-class recall shows where a high overall score hides missed examples. This is an image-level research evaluation, not a clinical accuracy estimate. Patient IDs and an external hospital dataset are unavailable.
