# Runs16 — Progressive Unfreezing Fine-Tuning

**Experiment**: Progressive unfreezing of YOLO11s-cls and YOLO11l-cls across three chained training stages with decreasing freeze depth and decreasing learning rate, all using Focal Loss.

**Objective**: Allow the model to first specialise the classification head in isolation, then progressively adapt deeper feature extraction layers, reducing the risk of catastrophic forgetting.

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

Each stage loads `best.pt` from the previous stage to warm-start the next phase.

---

## 2. Test Set Evaluation Metrics (Stage 3 Best Weights)

> ⚠️ **Note on Ultralytics CM axis**: The confusion matrix is stored as `matrix[PREDICTED][TRUE]`.
> - TP = cm[0][0] (Predicted Glaucoma, True Glaucoma)
> - FP = cm[0][1] (Predicted Glaucoma, True Normal)
> - FN = cm[1][0] (Predicted Normal, True Glaucoma)
> - TN = cm[1][1] (Predicted Normal, True Normal)

| Metric | YOLO11s (Small) | YOLO11l (Large) |
|--------|:---------------:|:---------------:|
| **Recall (Glaucoma)** | **0.4286** | **0.2857** |
| **Precision (Glaucoma)** | 0.5625 | 0.6667 |
| **F1-Score (Glaucoma)** | 0.4865 | 0.4000 |
| **Accuracy (Overall)** | 0.8750 | 0.8816 |

### Confusion Matrix Raw Data

**YOLO11s-cls (Stage 3):**
`TP: 9, FP: 7, FN: 12, TN: 124`
(9 glaucoma caught, 12 glaucoma missed, 7 False Alarms, 124 Normal correct)

**YOLO11l-cls (Stage 3):**
`TP: 6, FP: 3, FN: 15, TN: 128`
(6 glaucoma caught, 15 glaucoma missed, 3 False Alarms, 128 Normal correct)

---

## 3. Key Takeaways

1. **High Precision, Low Recall**: The progressive unfreezing strategy leads to conservative, high-confidence predictions. When the model flags Glaucoma it is correct 56–67% of the time, but recall is low — it misses most actual glaucoma cases.
2. **YOLO11l: Very selective**  With Recall=0.2857 but Precision=0.6667, the large model catches only ~3 in 9 glaucoma cases but when it does predict Glaucoma it is highly reliable.
3. **Multi-stage chaining is stable but not optimal for recall**: Starting head-first helps prevent catastrophic forgetting, but does not push recall as high as full-model Focal Loss training.
4. **Best strategy confirmed**: Full-model training with Focal Loss (Runs6, ~90% Recall) remains decisively superior for this class-imbalanced clinical screening task.
