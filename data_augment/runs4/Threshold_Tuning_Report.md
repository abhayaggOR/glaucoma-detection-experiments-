# Threshold Tuning Report — Class-Weighted Models

**Models**: YOLO11s-cls (small) and YOLO11l-cls (large) trained with class-weighted CE on augmented data
**Test Set**: Untouched test split from data_augment pipeline
**Thresholds Evaluated**: 0.10 to 0.55 (step 0.05)
**Positive Class**: GLAUCOMA_SUSPECT
**Confusion Matrices**: Generated on TEST data at the best-F1 threshold for each model

---

## yolo11s_weighted

| Threshold | Accuracy | Precision | Recall | F1 Score | TP | FP | TN | FN |
|:---------:|:--------:|:---------:|:------:|:--------:|:--:|:--:|:--:|:--:|
| 0.10 | 0.9013 | 0.6071 | 0.8095 | 0.6939 | 17 | 11 | 120 | 4 |
| 0.15 | 0.8947 | 0.6000 | 0.7143 | 0.6522 | 15 | 10 | 121 | 6 |
| 0.20 | 0.9013 | 0.6250 | 0.7143 | 0.6667 | 15 | 9 | 122 | 6 |
| 0.25 | 0.8882 | 0.5909 | 0.6190 | 0.6047 | 13 | 9 | 122 | 8 |
| 0.30 | 0.8882 | 0.5909 | 0.6190 | 0.6047 | 13 | 9 | 122 | 8 |
| 0.35 | 0.8947 | 0.6190 | 0.6190 | 0.6190 | 13 | 8 | 123 | 8 |
| 0.40 | 0.8947 | 0.6190 | 0.6190 | 0.6190 | 13 | 8 | 123 | 8 |
| 0.45 | 0.8947 | 0.6316 | 0.5714 | 0.6000 | 12 | 7 | 124 | 9 |
| 0.50 | 0.8816 | 0.5882 | 0.4762 | 0.5263 | 10 | 7 | 124 | 11 |
| 0.55 | 0.8882 | 0.6250 | 0.4762 | 0.5405 | 10 | 6 | 125 | 11 |

**Best F1**: threshold=0.10, F1=0.6939, Precision=0.6071, Recall=0.8095

**Best Recall**: threshold=0.10, Recall=0.8095, Precision=0.6071, F1=0.6939

**Confusion Matrix (test set)**: `yolo11s_weighted_confusion_matrix.png`
**Normalized Confusion Matrix (test set)**: `yolo11s_weighted_confusion_matrix_normalized.png`

---

## yolo11l_weighted

| Threshold | Accuracy | Precision | Recall | F1 Score | TP | FP | TN | FN |
|:---------:|:--------:|:---------:|:------:|:--------:|:--:|:--:|:--:|:--:|
| 0.10 | 0.7961 | 0.3750 | 0.7143 | 0.4918 | 15 | 25 | 106 | 6 |
| 0.15 | 0.8421 | 0.4516 | 0.6667 | 0.5385 | 14 | 17 | 114 | 7 |
| 0.20 | 0.8684 | 0.5200 | 0.6190 | 0.5652 | 13 | 12 | 119 | 8 |
| 0.25 | 0.8618 | 0.5000 | 0.5714 | 0.5333 | 12 | 12 | 119 | 9 |
| 0.30 | 0.8947 | 0.6471 | 0.5238 | 0.5789 | 11 | 6 | 125 | 10 |
| 0.35 | 0.9013 | 0.6875 | 0.5238 | 0.5946 | 11 | 5 | 126 | 10 |
| 0.40 | 0.9079 | 0.7333 | 0.5238 | 0.6111 | 11 | 4 | 127 | 10 |
| 0.45 | 0.9079 | 0.7692 | 0.4762 | 0.5882 | 10 | 3 | 128 | 11 |
| 0.50 | 0.9145 | 0.8333 | 0.4762 | 0.6061 | 10 | 2 | 129 | 11 |
| 0.55 | 0.9145 | 0.8333 | 0.4762 | 0.6061 | 10 | 2 | 129 | 11 |

**Best F1**: threshold=0.40, F1=0.6111, Precision=0.7333, Recall=0.5238

**Best Recall**: threshold=0.10, Recall=0.7143, Precision=0.3750, F1=0.4918

**Confusion Matrix (test set)**: `yolo11l_weighted_confusion_matrix.png`
**Normalized Confusion Matrix (test set)**: `yolo11l_weighted_confusion_matrix_normalized.png`

---

