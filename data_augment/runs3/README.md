# runs3 — Class-Weighted Cross Entropy Training

## What Was Done
Trained **YOLO11s-cls** and **YOLO11l-cls** with **class-weighted cross entropy loss** on augmented data to address class imbalance (GLAUCOMA_SUSPECT : NORMAL ≈ 1:6.6).

## Loss Function
**Weighted CrossEntropyLoss** with inverse-frequency class weights:
- GLAUCOMA_SUSPECT (class 0): **2.0000**
- NORMAL (class 1): **0.6667**

## Training Configuration

| Parameter | YOLO11s (Small) | YOLO11l (Large) |
|-----------|:---------------:|:---------------:|
| Batch Size | 16 | 32 |
| Epochs (max) | 300 | 300 |
| Early Stopping | patience=100 | patience=75 |
| Image Size | 512 | 512 |
| Data | Augmented (data_augment) | Augmented (data_augment) |

## Training Summary

| | YOLO11s | YOLO11l |
|---|:---:|:---:|
| Epochs Completed | 220 | 108 |
| Best Epoch | 120 | 33 |
| Training Time | 3.67 hours | 2.35 hours |

## Test Set Metrics (threshold = 0.50)

| Metric | YOLO11s | YOLO11l |
|--------|:-------:|:-------:|
| **Accuracy** | 0.8816 | 0.9145 |
| **Precision** | 0.5882 | 0.8333 |
| **Recall** | 0.4762 | 0.4762 |
| **F1 Score** | 0.5263 | 0.6061 |

## Confusion Matrix on Test Data

### YOLO11s (Small)

|  | Pred NORMAL | Pred GLAUCOMA |
|--|:-----------:|:-------------:|
| **True NORMAL** | TN = 124 | FP = 7 |
| **True GLAUCOMA** | FN = 11 | TP = 10 |

### YOLO11l (Large)

|  | Pred NORMAL | Pred GLAUCOMA |
|--|:-----------:|:-------------:|
| **True NORMAL** | TN = 129 | FP = 2 |
| **True GLAUCOMA** | FN = 11 | TP = 10 |

## Confusion Matrix Images
- `yolo11s_weighted/confusion_matrix.png`
- `yolo11s_weighted/confusion_matrix_normalized.png`
- `yolo11l_weighted/confusion_matrix.png`
- `yolo11l_weighted/confusion_matrix_normalized.png`

## Key Takeaway
Both models detect only **10/21 (47.6%)** glaucoma cases at threshold=0.50. The large model has much higher precision (0.83 vs 0.59). Threshold tuning (runs4) shows recall improves significantly at lower thresholds.
