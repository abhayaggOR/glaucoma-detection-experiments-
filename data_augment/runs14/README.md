# Runs14 Report — Fine-Tuning YOLOv11 (Last 5 Layers)

**Experiment**: Fine-tuning YOLOv11 small and large models on the augmented dataset by freezing the first 6 modules (`freeze=6`) and training the last 5 modules.
**Objective**: Following Runs13 (where only the last 3 layers were trained), this expands the trainable parameter space slightly deeper into the architecture to see if mid-level feature adaptations prevent the majority-class bias.

---

## 1. Training Configuration

| Parameter | YOLO11s-cls (Small) | YOLO11l-cls (Large) |
|-----------|:-------------------:|:-------------------:|
| **Trainable Layers** | Last 5 Modules (6, 7, 8, 9, 10)| Last 5 Modules (6, 7, 8, 9, 10)|
| **Loss Function** | Default Cross Entropy | Default Cross Entropy |
| **Epochs** | 200 | 200 |
| **Image Size** | 512 | 512 |
| **Batch Size** | 16 | **16 (Reduced due to OOM)** |
| **Early Stopping** | 100 (patience) | 75 (patience) |

*(Note: YOLO11l batch size had to be reduced to 16, as opening 5 layers to backpropagation caused the 16GB VRAM to fragment and OOM during validation Dataloader pinning at batch 32).*

---

## 2. Test Set Evaluation Metrics (Test Split)

Models evaluated on the exact same test dataset.

| Metric | YOLO11s (Small) | YOLO11l (Large) |
|--------|:---------------:|:---------------:|
| **Recall (Glaucoma)** | **0.5294** | **0.4211** |
| **Precision (Glaucoma)** | 0.4286 | 0.3810 |
| **F1-Score (Glaucoma)** | 0.4737 | 0.4000 |
| **Accuracy (Overall)** | 0.8684 | 0.8421 |

### 2.1 Confusion Matrix Raw Data

**YOLO11s-cls:**
`TP: 9, FN: 8, FP: 12, TN: 123`

**YOLO11l-cls:**
`TP: 8, FN: 11, FP: 13, TN: 120`

---

## 3. Conclusion & Takeaways

1. **Balanced, but Still Poor**: Training the last 5 layers yielded marginally better F1 harmonizations than the 3-layer freeze (Runs13), pulling YOLO11l out of its precision floor (up to 38% from 19%). 
2. **Full Training is Superior**: Comparing Runs12, Runs13, and Runs14 proves that when using a standard Cross-Entropy loss on this dataset, **full architecture training (Runs12) outperforms all manual freezing configurations**. 
3. **The Absolute Necessity of Loss Tweaking**: Ultimately, none of the freezing attempts independently broke 60% recall on the heavyweight model. We must return to Focal Loss (Runs6) or Class Weights to force the large parameters off the background-class local minima.
