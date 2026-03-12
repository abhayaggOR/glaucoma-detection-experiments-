# Runs3 Report — Class-Weighted Cross Entropy Training

**Experiment**: Training YOLO v11 models with **class-weighted cross entropy loss** on augmented data  
**Date**: March 2026  
**Data**: Augmented dataset (3:1 ratio)  
**Loss Function**: Weighted CrossEntropyLoss with inverse-frequency weights

---

## Motivation

The training data has significant class imbalance:

| Split | GLAUCOMA_SUSPECT | NORMAL | Total | Ratio |
|-------|:----------------:|:------:|:-----:|:-----:|
| Train | 203 | 609 | 812 | 1 : 3.0 |
| Val   | 20 | 131 | 151 | 1 : 6.6 |
| Test  | 21 | 131 | 152 | 1 : 6.2 |

Standard cross entropy treats all classes equally, which biases the model toward the majority class (NORMAL). Class-weighted CE addresses this by assigning higher loss weight to the minority class.

**Weights used** (inverse frequency):
- GLAUCOMA_SUSPECT (class 0): **2.0000**
- NORMAL (class 1): **0.6667**

---

## Training Configuration

| Parameter | YOLO11s-cls (Small) | YOLO11l-cls (Large) |
|-----------|:-------------------:|:-------------------:|
| Image Size | 512 | 512 |
| Batch Size | 16 | 32 |
| Max Epochs | 300 | 300 |
| Early Stop (patience) | 100 | 75 |
| Optimizer | AdamW (lr=0.001667) | AdamW (lr=0.001667) |
| Device | GPU (NVIDIA RTX 2000 Ada) | GPU (NVIDIA RTX 2000 Ada) |

---

## Training Summary

| Metric | YOLO11s (Small) | YOLO11l (Large) |
|--------|:---------------:|:---------------:|
| Epochs Completed | 220 / 300 | 108 / 300 |
| Best Epoch | 120 | 33 |
| Early Stopped | Yes (patience=100) | Yes (patience=75) |
| Training Time | 3.665 hours | 2.347 hours |

---

## Test Set Metrics (threshold = 0.50)

| Metric | YOLO11s (Small) | YOLO11l (Large) |
|--------|:---------------:|:---------------:|
| **Accuracy**  | 0.8816 | 0.9145 |
| **Precision** | 0.5882 | 0.8333 |
| **Recall**    | 0.4762 | 0.4762 |
| **F1 Score**  | 0.5263 | 0.6061 |

### Confusion Matrix — YOLO11s (Small)

|  | Predicted NORMAL | Predicted GLAUCOMA |
|--|:----------------:|:------------------:|
| **Actual NORMAL** | TN = 124 | FP = 7 |
| **Actual GLAUCOMA** | FN = 11 | TP = 10 |

- Out of 21 glaucoma cases: detected **10**, missed **11**
- Out of 131 normal cases: correctly classified **124**, false alarm on **7**

### Confusion Matrix — YOLO11l (Large)

|  | Predicted NORMAL | Predicted GLAUCOMA |
|--|:----------------:|:------------------:|
| **Actual NORMAL** | TN = 129 | FP = 2 |
| **Actual GLAUCOMA** | FN = 11 | TP = 10 |

- Out of 21 glaucoma cases: detected **10**, missed **11**
- Out of 131 normal cases: correctly classified **129**, false alarm on **2**

---

## Confusion Matrix Images (Test Set)

### YOLO11s-cls (Small)
- `yolo11s_weighted/confusion_matrix.png`
- `yolo11s_weighted/confusion_matrix_normalized.png`

### YOLO11l-cls (Large)
- `yolo11l_weighted/confusion_matrix.png`
- `yolo11l_weighted/confusion_matrix_normalized.png`

---

## Key Observations

1. **Both models detect only ~48% of glaucoma cases** (10 out of 21) at the default 0.50 threshold — recall is low.
2. **Large model has much higher precision** (0.8333 vs 0.5882) — fewer false positives.
3. **Large model converged faster** (best at epoch 33 vs 120) but overall recall is identical.
4. **Threshold tuning** (see runs4) shows that lowering the threshold to 0.10 significantly improves recall:
   - Small model at thresh=0.10: Recall=0.8095, F1=0.6939
   - Large model at thresh=0.10: Recall=0.7143, F1=0.4918
5. **The class-weighted loss helped** shift the decision boundary slightly toward detecting more glaucoma cases, but the effect is modest at the default threshold. Threshold tuning is essential to unlock the full potential of the weighted models.
