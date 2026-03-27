# Runs16 — Progressive Unfreezing Fine-Tuning

**Experiment**: Progressive unfreezing of YOLO11s-cls and YOLO11l-cls across three chained training stages with decreasing freeze depth and decreasing learning rate, all using Focal Loss.

**Objective**: Allow the model to first specialise the classification head in isolation, then progressively adapt deeper feature extraction layers to the new dataset, reducing the risk of catastrophic forgetting of ImageNet pre-trained representations.

---

## 1. Training Configuration

| Parameter | Value |
|-----------|-------|
| **Loss Function** | Focal Loss (γ=2.0, α=[0.85, 0.15]) |
| **Optimizer** | AdamW |
| **Total Epochs** | 140 (20 + 40 + 80) |
| **Image Size** | 512 |
| **Batch Size** | 16 |
| **Early Stopping** | 75 patience (per stage) |

### Progressive Unfreezing Stages

| Stage | Freeze | Epochs | Learning Rate | What's Being Trained |
|:-----:|:------:|:------:|:-------------:|---------------------|
| 1 | 10 | 20 | 1e-3 | Classification Head only |
| 2 | 5 | 40 | 5e-4 | Last 6 modules (5–10) |
| 3 | 0 | 80 | 1e-4 | All 11 modules |

Each stage loads the `best.pt` from the previous stage to warm-start the next phase.

---

## 2. Test Set Evaluation Metrics (Stage 3 Best Weights)

| Metric | YOLO11s (Small) | YOLO11l (Large) |
|--------|:---------------:|:---------------:|
| **Recall (Glaucoma)** | **0.5625** | **0.2857** |
| **Precision (Glaucoma)** | 0.4286 | 0.6667 |
| **F1-Score (Glaucoma)** | 0.4865 | 0.4000 |
| **Accuracy (Overall)** | 0.8750 | 0.8816 |

### Confusion Matrix Raw Data

**YOLO11s-cls (Stage 3):**
`TP: 9, FN: 7, FP: 12, TN: 124`

**YOLO11l-cls (Stage 3):**
`TP: 6, FP: 3, FN: 15, TN: 128`

> ⚠️ Note: Ultralytics confusion matrix stores `matrix[PREDICTED][TRUE]`. TP=predicted Glaucoma & truly Glaucoma, FN=predicted Normal & truly Glaucoma, FP=predicted Glaucoma & truly Normal.

---

## 3. Key Takeaways

1. **High Precision for YOLO11l**: Progressive unfreezing led YOLO11l to learn conservative but accurate predictions — Precision=0.6667 (correct when it flags glaucoma), but Recall=0.2857 (misses many actual cases).
2. **YOLO11s balanced better**: The small model achieved a more useful Recall=0.5625 with Precision=0.4286 — catching more glaucoma at the expense of some false positives.
3. **Multi-Stage Chaining is Stable**: The progressive approach prevents catastrophic forgetting, but doesn't automatically maximise recall unless the full backbone is free to adapt from the start.
4. **Best strategy confirmed**: Full-model training with Focal Loss (Runs6, ~90% Recall) remains decisively superior for this class-imbalanced screening task.
