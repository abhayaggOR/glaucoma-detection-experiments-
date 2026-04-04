# Runs20 — Channel Attention (SE Block) + Cross-Space Fusion + SupCon

**Experiment**: Multi-scale feature fusion with a **Squeeze-and-Excitation (SE) channel attention** block applied on the concatenated 1792-d feature vector.

---

## Architecture

### Cross-Space Projection + SE Block
1. Extract P3/P4/P5 via GAP → project to 256/512/1024 → concat h ∈ R^1792
2. Apply SE block:
```
w = σ(W2 · ReLU(W1 · h))   where W1: 1792→448, W2: 448→1792
h' = w ⊙ h
```

### SupCon Pretraining
- Operates on attention-gated features **h'**
- **Projection Head**: Linear(1792→512) → ReLU → Linear(512→128) → L2 Norm
- **Epochs**: 150, AdamW LR=1e-3, τ=0.07

### Classifier Fine-Tuning
- **Classifier**: Linear(1792→2)
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
- `training.log`, `tsne_pre.png`, `tsne_post.png`, `*_cm.png`, `*_best.pth`
