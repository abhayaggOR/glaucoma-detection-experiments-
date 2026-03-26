# Runs13 Report — Fine-Tuning YOLOv11 (Last 3 Layers)

**Experiment**: Fine-tuning YOLOv11 small and large models on the augmented dataset by freezing the first 8 modules (`freeze=8`) and only actively training the final 3 modules in the classification head.
**Objective**: Determine if isolating gradient updates to the deep classification layers improves recall on the severe class imbalance compared to training the entire architecture (Runs12).

---

## 1. Training Configuration

| Parameter | YOLO11s-cls (Small) | YOLO11l-cls (Large) |
|-----------|:-------------------:|:-------------------:|
| **Trainable Layers** | Last 3 Modules (8, 9, 10) | Last 3 Modules (8, 9, 10) |
| **Loss Function** | Default Cross Entropy | Default Cross Entropy |
| **Epochs** | 200 | 200 |
| **Image Size** | 512 | 512 |
| **Batch Size** | 16 | 32 |
| **Early Stopping** | 100 (patience) | 75 (patience) |

---

## 2. Test Set Evaluation Metrics (Test Split)

Models evaluated on the test set (Threshold = 0.50).

| Metric | YOLO11s (Small) | YOLO11l (Large) |
|--------|:---------------:|:---------------:|
| **Recall (Glaucoma)** | **0.5833** | **0.4444** |
| **Precision (Glaucoma)** | 0.3333 | 0.1905 |
| **F1-Score (Glaucoma)** | 0.4242 | 0.2667 |
| **Accuracy (Overall)** | 0.8750 | 0.8553 |

### 2.1 Confusion Matrix Raw Data

**YOLO11s-cls:**
`TP: 7, FN: 5, FP: 14, TN: 126`

**YOLO11l-cls:**
`TP: 4, FN: 5, FP: 17, TN: 126`

---

## 3. Key Takeaways

1. **Performance Drop vs Full Training**: Fine-tuning only the last 3 layers resulted in worse absolute recall for YOLO11s (58.33% vs 73.33% in Runs12 full-baseline).
2. **Persistent Large Model Bias**: YOLO11l fared slightly better in recall than its Runs12 counterpart (44.44% vs 38.10%), but precision completely tanked to 19%. Freezing the backbone did not solve the fundamental class imbalance collapse that plagues the heavier architecture without Focal Loss.
