# Runs15 — Differential Learning Rate Fine-Tuning

**Experiment**: Differential Learning Rates applied to YOLO11s-cls and YOLO11l-cls using **PyTorch backward gradient hooks** to scale gradients per-block while preserving the Ultralytics scheduler.

**Objective**: Test whether applying lower learning rates to early/stable feature extraction layers and a higher LR to the final classification head enables better fine-tuning on the imbalanced glaucoma dataset.

---

## 1. Training Configuration

| Parameter | Value |
|-----------|-------|
| **Loss Function** | Focal Loss (γ=2.0, α=[0.85, 0.15]) |
| **Optimizer** | AdamW |
| **Epochs** | 200 |
| **Image Size** | 512 |
| **Batch Size** | 16 |
| **Early Stopping** | 75 (patience) |

### Differential LR Strategy (via gradient hooks)

| Block Range | Modules | Effective LR |
|:-----------:|:-------:|:------------:|
| Early layers | 0 – 3 | **1e-5** (scale ×0.1) |
| Middle layers | 4 – 7 | **5e-5** (scale ×0.5) |
| Final layers + head | 8 – 10 | **1e-4** (scale ×1.0) |

**Implementation**: A `register_hook(lambda grad: grad * scale)` backward hook is registered on every trainable parameter before training begins.

---

## 2. Test Set Evaluation Metrics

> ⚠️ **Note on Ultralytics CM axis**: The confusion matrix is stored as `matrix[PREDICTED][TRUE]`.
> - TP = cm[0][0] (Predicted Glaucoma, True Glaucoma)
> - FP = cm[0][1] (Predicted Glaucoma, True Normal)
> - FN = cm[1][0] (Predicted Normal, True Glaucoma)
> - TN = cm[1][1] (Predicted Normal, True Normal)

| Metric | YOLO11s (Small) | YOLO11l (Large) |
|--------|:---------------:|:---------------:|
| **Recall (Glaucoma)** | **0.4762** | **0.4286** |
| **Precision (Glaucoma)** | 0.5882 | 0.5625 |
| **F1-Score (Glaucoma)** | 0.5263 | 0.4865 |
| **Accuracy (Overall)** | 0.8816 | 0.8750 |

### Confusion Matrix Raw Data

**YOLO11s-cls:**
`TP: 10, FP: 7, FN: 11, TN: 124`
(10 glaucoma caught, 11 glaucoma missed, 7 False Alarms, 124 Normal correct)

**YOLO11l-cls:**
`TP: 9, FP: 7, FN: 12, TN: 124`
(9 glaucoma caught, 12 glaucoma missed, 7 False Alarms, 124 Normal correct)

---

## 3. Key Takeaways

1. **Moderate Recall**: Both models catch ~43–48% of real glaucoma cases on the test set. This is better than plain 3-layer or 5-layer freezing (Runs13/14), but well below the ~90% achieved by full Focal Loss training (Runs6).
2. **Higher Precision than Recall**: The differential LR strategy leads models to be more conservative — when they do flag glaucoma, they are right ~56–59% of the time, but they miss about half the real cases.
3. **Gradient hooks add stability**: Training with different effective LRs per block prevents runaway catastrophic forgetting in the early layers while still allowing the classification head to adapt.
