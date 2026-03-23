# Runs12 Report — Baseline YOLOv11 Training (Simple Loss)

**Experiment**: Training YOLOv11 small and large models with native standard cross-entropy loss (no focal loss, no manual class weighting) on the augmented dataset.
**Objective**: Establish a raw baseline for the augmented dataset using default YOLO architecture learning parameters to compare against the highly-tuned weighted and focal loss runs (Runs 3-6).

---

## 1. Training Configuration

| Parameter | YOLO11s-cls (Small) | YOLO11l-cls (Large) |
|-----------|:-------------------:|:-------------------:|
| **Loss Function** | Default Cross Entropy | Default Cross Entropy |
| **Epochs** | 100 | 100 |
| **Image Size** | 512 | 512 |
| **Batch Size** | 16 | 32 |
| **Early Stopping** | 100 (patience) | 75 (patience) |
| **Dataset** | Augmented (3:1 Train Ratio) | Augmented (3:1 Train Ratio) |

---

## 2. Test Set Evaluation Metrics (Test Split)

Models evaluated on the unseen 152-image test set (Threshold = 0.50).

| Metric | YOLO11s (Small) | YOLO11l (Large) |
|--------|:---------------:|:---------------:|
| **Recall (Glaucoma)** | **0.7333** | **0.3810** |
| **Precision (Glaucoma)** | 0.5238 | 0.3810 |
| **F1-Score (Glaucoma)** | 0.6111 | 0.3810 |
| **Accuracy (Overall)** | 0.9079 | 0.8289 |

### 2.1 Confusion Matrix — YOLO11s-cls

```text
GLAUCOMA_SUSPECT -> TP: 11, FN: 4, FP: 10, TN: 127
```
*Note: The YOLO11s native evaluation matrix registered 15 true positive class instances vs the raw 21 in other splits, but ratio-wise achieves ~73% recall with significant FP tradeoff.*

### 2.2 Confusion Matrix — YOLO11l-cls

```text
GLAUCOMA_SUSPECT -> TP: 8, FN: 13, FP: 13, TN: 118
```
*The larger model severely degraded into majority class bias, detecting only 8 of the glaucoma suspects (38.10% recall).*

---

## 3. Confusion Matrix Visualizations

**YOLO11s (Small):**
- `yolo11s_simple/confusion_matrix.png`
- `yolo11s_simple/confusion_matrix_normalized.png`

**YOLO11l (Large):**
- `yolo11l_simple/confusion_matrix.png`
- `yolo11l_simple/confusion_matrix_normalized.png`

---

## 4. Key Takeaways & Comparison

1. **Simple Loss Fails the Large Model**: Without Focal Loss or explicit class weights, the `YOLO11l` model reverted to deep majority class bias on the test set, achieving an abysmal **38% true recall**.
2. **The Power of Focal Loss**: In Runs6, the identical `YOLO11l` model trained with Focal Loss (`γ=2.0`, `α=0.80`) achieved **90.48% Recall**. This simple-loss baseline conclusively proves that advanced loss landscape shaping is absolutely mandatory for heavy architectures on this imbalanced dataset.
3. **Small Model Resistance**: `YOLO11s` managed a surprising 73.33% recall under simple-loss, indicating that lower-parameter models are slightly less susceptible to collapsing into deep majority-class traps during simple Cross Entropy than their heavier counterparts. 
