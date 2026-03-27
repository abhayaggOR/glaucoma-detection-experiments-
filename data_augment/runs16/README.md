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
| **Recall (Glaucoma)** | **0.5625** | **0.6667** |
| **Precision (Glaucoma)** | 0.4286 | 0.2857 |
| **F1-Score (Glaucoma)** | 0.4865 | 0.4000 |
| **Accuracy (Overall)** | 0.8750 | 0.8816 |

### Confusion Matrix Raw Data

**YOLO11s-cls (Stage 3):**
`TP: 9, FN: 7, FP: 12, TN: 124`

**YOLO11l-cls (Stage 3):**
`TP: 6, FN: 3, FP: 15, TN: 128`

---

## 3. Key Takeaways

1. **Best Recall for YOLO11l Across All Fine-Tuning Experiments**: Progressive unfreezing pushed YOLO11l recall to **66.67%** — the highest it has reached in all Runs12–16 freezing/fine-tuning variants.
2. **Precision–Recall Trade-off**: The large model sacrifices precision (28.57%) for higher recall in Stage 3. This is actually desirable for a clinical glaucoma screening tool where missing a positive case (FN) is far more costly than a false alarm (FP).
3. **Multi-Stage Chaining Works**: Starting head-first, then progressively relaxing the backbone, leads to better adaptation for the large model than any one-shot fine-tuning variant tested in Runs13 and Runs14.
4. **Next Step**: To match or exceed Runs6's ~90% recall, the full backbone must be trained end-to-end with Focal Loss from the start — confirming that the best strategy is standard full-model Focal Loss training rather than any freezing strategy.
