# Runs15 — Differential Learning Rate Fine-Tuning

**Experiment**: Differential Learning Rates applied to YOLO11s-cls and YOLO11l-cls using **PyTorch backward gradient hooks** to scale gradients per-block while preserving the Ultralytics scheduler.

**Objective**: Test whether applying lower learning rates to early/stable feature extraction layers and a higher LR to the final classification head enables better fine-tuning on the imbalanced glaucoma dataset, compared to a uniform learning rate.

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

**Implementation**: A `register_hook(lambda grad: grad * scale)` backward hook is registered on every trainable parameter before training begins. This lets the Ultralytics `OneCycleLR` scheduler operate untouched on the global `lr0=1e-4` while each block's effective gradient magnitude is scaled to match the differential LR requirements.

---

## 2. Test Set Evaluation Metrics

| Metric | YOLO11s (Small) | YOLO11l (Large) |
|--------|:---------------:|:---------------:|
| **Recall (Glaucoma)** | **0.5882** | **0.5625** |
| **Precision (Glaucoma)** | 0.4762 | 0.4286 |
| **F1-Score (Glaucoma)** | 0.5263 | 0.4865 |
| **Accuracy (Overall)** | 0.8816 | 0.8750 |

### Confusion Matrix Raw Data

**YOLO11s-cls:**
`TP: 10, FN: 7, FP: 11, TN: 124`

**YOLO11l-cls:**
`TP: 9, FN: 7, FP: 12, TN: 124`

---

## 3. Key Takeaways

1. **Best Fine-Tuning Result So Far**: With the gradient hook strategy and Focal Loss, YOLO11s achieved its highest F1 across all fine-tuning experiments (0.5263 vs 0.4242 in Runs13 and 0.4737 in Runs14).
2. **Differential LR Helps Stability**: Both models showed more balanced confusion matrices (TP/FP ratio) than plain freezing approaches, suggesting that gradual gradient scaling prevents the early-layer feature representations from catastrophic forgetting.
3. **Still Below Full-Training Focal Loss**: Compared to Runs6 (full fine-tuning + Focal Loss, Recall ~0.90), partial differential LR still leaves recall headroom. This confirms that maximum backbone flexibility combined with correct loss function yields the best recall.
