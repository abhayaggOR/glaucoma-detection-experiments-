# Runs21 — Cross-Scale Attention (Softmax Weighted Sum) + SupCon

**Experiment**: Per-image **dynamic scale weighting** using softmax attention across P3/P4/P5.

---

## Architecture

### Cross-Scale Attention Module
1. Project each scale to **common 512-d space**:
   - P3 (256) → Linear(256, 512)
   - P4 (256) → Linear(256, 512)
   - P5 (512) → Linear(512, 512)
2. Compute softmax attention over 3 **learned** scalar scores:
```
α = softmax([a3, a4, a5])
h = α3·P3' + α4·P4' + α5·P5'  ∈ R^512
```

### SupCon Pretraining
- **Projection Head**: Linear(512→256) → ReLU → Linear(256→128) → L2 Norm
- **Epochs**: 150, AdamW LR=1e-3, τ=0.07
- Logs α3/α4/α5 every 10 epochs to track scale importance evolution

### Classifier Fine-Tuning
- **Classifier**: Linear(512→2)
- 4 ablations on test set only

---

## Test Set Evaluation Metrics

| Experiment | Accuracy | Precision | Recall | F1-Score |
|-----------|:--------:|:---------:|:------:|:--------:|
| **Baseline** | TBD | TBD | TBD | TBD |
| **SupCon Frozen** | TBD | TBD | TBD | TBD |
| **SupCon Partial** | TBD | TBD | TBD | TBD |
| **SupCon Full** | TBD | TBD | TBD | TBD |

---

## Output Files
- `training.log` (includes α3/α4/α5 evolution), `tsne_pre.png`, `tsne_post.png`, `*_cm.png`
