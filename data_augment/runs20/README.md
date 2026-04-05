# Runs20 — SE Channel Attention + Cross-Space Fusion + SupCon

**Experiment**: Multi-scale feature fusion with a **Squeeze-and-Excitation (SE) channel attention** block applied on the concatenated 1792-d feature vector.

---

## Architecture

### Cross-Space Projection + SE Channel Attention
1. Extract P3 (256-d), P4 (256-d), P5 (512-d) via GAP
2. Cross-space project: P3→256, P4→512, P5→1024 → concat **h ∈ R^1792**
3. Apply SE block (reduction ratio r=4, squeeze dim=448):
```
w = σ(W2 · ReLU(W1 · h))     W1: 1792→448, W2: 448→1792
h' = w ⊙ h                    channel-recalibrated features
```

### SupCon Pretraining
- **Projection Head**: Linear(1792→512) → ReLU → Linear(512→128) → L2 Norm
- **Loss**: SupConLoss (τ=0.07), AdamW LR=1e-3, CosineAnnealing
- **Epochs**: 150

### Classifier Fine-Tuning
- **Classifier**: Linear(1792→2), CrossEntropy + Label Smoothing 0.1
- **LR**: backbone=1e-5, classifier=1e-3, max epochs=200, patience=100
- All 4 ablations evaluated on **test set only** (152 images)

---

## Test Set Evaluation Results

> ✅ Metrics computed using `scikit-learn` with `y_true=1` = GLAUCOMA (inverted from native ImageFolder mapping).
> ✅ Confusion matrices built exclusively from the 152-image test split.

| Experiment | Accuracy | Precision | Recall | F1-Score | TP | FN | FP | TN |
|-----------|:--------:|:---------:|:------:|:--------:|:--:|:--:|:--:|:--:|
| Baseline (no SupCon) | 0.9211 | 0.8462 | 0.5238 | 0.6471 | 11 | 10 | 2 | 129 |
| SupCon + Frozen | 0.9145 | 0.8333 | 0.4762 | 0.6061 | 10 | 11 | 2 | 129 |
| **SupCon + Partial** | **0.9145** | **0.7500** | **0.5714** | **0.6486** | **12** | **9** | **4** | **127** |
| **SupCon + Full** | **0.9211** | **0.8000** | **0.5714** | **0.6667** | **12** | **9** | **3** | **128** |

**🏆 Best: SupCon + Full Finetuning** — **57.14% Recall** (12/21), highest F1=66.67% for SE-based model.

---

## Output Files
- `training.log` — full epoch-by-epoch metrics
- `tsne_pre.png` / `tsne_post.png` — embedding visualizations
- `baseline_cm.png`, `supcon_frozen_cm.png`, `supcon_partial_cm.png`, `supcon_full_cm.png`
- `*_best.pth` — best model checkpoints

---

## Key Takeaways
1. **SE attention improves full-finetuning**: SupCon+Full reached 57.14% recall (12/21) — matching Runs19 Frozen but with a lower FP count (3 vs 3+).
2. **Channel recalibration prevents early collapse**: The SE block gating helped the frozen backbone experiments maintain slightly better F1 than raw frozen in Runs17/18.
3. **SE vs Learnable Weights**: The SE block (Runs20) provides 57.14% best recall vs Runs19's 61.90% best recall — learnable scalar weights outperform SE-style channel gating on this small dataset.
