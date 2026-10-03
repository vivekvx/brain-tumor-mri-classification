# Brain MRI dataset split report

Source archive: `drive-download-20261003T062927Z-1-001.zip`. Random seed: `42`.
Grouping key: `class + filename before _jpg.rf.`. All 2443 images were retained in 2091 source-image groups.

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| glioma | 560 | 164 | 81 |
| meningioma | 378 | 112 | 55 |
| no_tumor | 335 | 99 | 49 |
| pituitary | 423 | 125 | 62 |
| **Total** | **1696** | **500** | **247** |

Images moved from the supplied split: **1165**.
Source-image groups crossing output splits: **0**.

The output is in `data/cleaned/`; `manifest.csv` records each image's original and new split plus SHA-256.
These names identify source images, not patients. Patient-level independence cannot be confirmed from this archive.
The test split should be reserved for final evaluation; model selection should use validation only.
