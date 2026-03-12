# Threshold Tuning Report — Augmented Data Models

**Models**: YOLO11s-cls (small) and YOLO11l-cls (large) trained on augmented data
**Test Set**: Untouched test split from data_augment pipeline
**Thresholds Evaluated**: 0.10 to 0.55 (step 0.05)
**Positive Class**: GLAUCOMA_SUSPECT
**Confusion Matrices**: Generated on TEST data at the best-F1 threshold for each model

---

## yolo11s_augmented

| Threshold | Accuracy | Precision | Recall | F1 Score | TP | FP | TN | FN |
|:---------:|:--------:|:---------:|:------:|:--------:|:--:|:--:|:--:|:--:|
| 0.10 | 0.8618 | 0.5000 | 0.5714 | 0.5333 | 12 | 12 | 119 | 9 |
| 0.15 | 0.8684 | 0.5217 | 0.5714 | 0.5455 | 12 | 11 | 120 | 9 |
| 0.20 | 0.8750 | 0.5455 | 0.5714 | 0.5581 | 12 | 10 | 121 | 9 |
| 0.25 | 0.8947 | 0.6316 | 0.5714 | 0.6000 | 12 | 7 | 124 | 9 |
| 0.30 | 0.8882 | 0.6250 | 0.4762 | 0.5405 | 10 | 6 | 125 | 11 |
| 0.35 | 0.8882 | 0.6250 | 0.4762 | 0.5405 | 10 | 6 | 125 | 11 |
| 0.40 | 0.8947 | 0.6667 | 0.4762 | 0.5556 | 10 | 5 | 126 | 11 |
| 0.45 | 0.8947 | 0.6667 | 0.4762 | 0.5556 | 10 | 5 | 126 | 11 |
| 0.50 | 0.9013 | 0.7143 | 0.4762 | 0.5714 | 10 | 4 | 127 | 11 |
| 0.55 | 0.9013 | 0.7143 | 0.4762 | 0.5714 | 10 | 4 | 127 | 11 |

**Best F1**: threshold=0.25, F1=0.6000, Precision=0.6316, Recall=0.5714

**Best Recall**: threshold=0.25, Recall=0.5714, Precision=0.6316, F1=0.6000

**Confusion Matrix (test set)**: `yolo11s_augmented_confusion_matrix.png`
**Normalized Confusion Matrix (test set)**: `yolo11s_augmented_confusion_matrix_normalized.png`

---

## yolo11l_augmented

| Threshold | Accuracy | Precision | Recall | F1 Score | TP | FP | TN | FN |
|:---------:|:--------:|:---------:|:------:|:--------:|:--:|:--:|:--:|:--:|
| 0.10 | 0.8618 | 0.5000 | 0.7619 | 0.6038 | 16 | 16 | 115 | 5 |
| 0.15 | 0.8684 | 0.5161 | 0.7619 | 0.6154 | 16 | 15 | 116 | 5 |
| 0.20 | 0.8947 | 0.5926 | 0.7619 | 0.6667 | 16 | 11 | 120 | 5 |
| 0.25 | 0.8947 | 0.5926 | 0.7619 | 0.6667 | 16 | 11 | 120 | 5 |
| 0.30 | 0.9013 | 0.6250 | 0.7143 | 0.6667 | 15 | 9 | 122 | 6 |
| 0.35 | 0.9013 | 0.6500 | 0.6190 | 0.6341 | 13 | 7 | 124 | 8 |
| 0.40 | 0.8947 | 0.6316 | 0.5714 | 0.6000 | 12 | 7 | 124 | 9 |
| 0.45 | 0.9013 | 0.6667 | 0.5714 | 0.6154 | 12 | 6 | 125 | 9 |
| 0.50 | 0.9013 | 0.6667 | 0.5714 | 0.6154 | 12 | 6 | 125 | 9 |
| 0.55 | 0.9079 | 0.7059 | 0.5714 | 0.6316 | 12 | 5 | 126 | 9 |

**Best F1**: threshold=0.20, F1=0.6667, Precision=0.5926, Recall=0.7619

**Best Recall**: threshold=0.20, Recall=0.7619, Precision=0.5926, F1=0.6667

**Confusion Matrix (test set)**: `yolo11l_augmented_confusion_matrix.png`
**Normalized Confusion Matrix (test set)**: `yolo11l_augmented_confusion_matrix_normalized.png`

---

