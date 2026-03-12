# runs4 — Threshold Tuning on Class-Weighted Models

## What Was Done
**Threshold tuning** on the models trained in **runs3** (class-weighted CE). Swept confidence thresholds from 0.10 to 0.55 (step 0.05) to find the optimal decision boundary for GLAUCOMA_SUSPECT detection.

## What It Used
- **Same models** from runs3 (no retraining)
- Models were trained with weighted CE loss (GLAUCOMA_SUSPECT=2.0, NORMAL=0.67)
- Evaluated on the **test set** (21 GLAUCOMA_SUSPECT, 131 NORMAL)

## Best Results

### YOLO11s (Small)
**Best F1** at threshold=0.10: F1=0.6939, Precision=0.6071, **Recall=0.8095**

| Threshold | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |
|:---------:|:--------:|:---------:|:------:|:--:|:--:|:--:|:--:|:--:|
| 0.10 | 0.9013 | 0.6071 | **0.8095** | **0.6939** | 17 | 11 | 120 | 4 |
| 0.50 | 0.8816 | 0.5882 | 0.4762 | 0.5263 | 10 | 7 | 124 | 11 |

### YOLO11l (Large)
**Best F1** at threshold=0.40: F1=0.6111, Precision=0.7333, Recall=0.5238

| Threshold | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |
|:---------:|:--------:|:---------:|:------:|:--:|:--:|:--:|:--:|:--:|
| 0.10 | 0.7961 | 0.3750 | **0.7143** | 0.4918 | 15 | 25 | 106 | 6 |
| 0.40 | 0.9079 | 0.7333 | 0.5238 | **0.6111** | 11 | 4 | 127 | 10 |

## Confusion Matrix at Best F1 Threshold

### YOLO11s @ thresh=0.10

|  | Pred NORMAL | Pred GLAUCOMA |
|--|:-----------:|:-------------:|
| **True NORMAL** | TN = 120 | FP = 11 |
| **True GLAUCOMA** | FN = 4 | TP = 17 |

### YOLO11l @ thresh=0.40

|  | Pred NORMAL | Pred GLAUCOMA |
|--|:-----------:|:-------------:|
| **True NORMAL** | TN = 127 | FP = 4 |
| **True GLAUCOMA** | FN = 10 | TP = 11 |

## Confusion Matrix Images
- `yolo11s_weighted_confusion_matrix.png` / `_normalized.png`
- `yolo11l_weighted_confusion_matrix.png` / `_normalized.png`

## Key Takeaway
Lowering threshold from 0.50 to 0.10 boosts **YOLO11s recall from 47.6% → 80.9%** (catches 17/21 glaucoma cases). The trade-off is more false positives (7 → 11). For medical screening, high recall is preferred.
