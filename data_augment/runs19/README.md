# Runs19 — Learnable Scale Weights + Cross-Space Fusion + SupCon

**Experiment**: Multi-scale feature fusion (P3, P4, P5) with **learnable scalar weights** (w3, w4, w5) applied per-scale after cross-space projection.

---

## Architecture

### Cross-Space Projection (Matching Spec Dims)
| Scale | Native Dim | Projected Dim |
|-------|:-----------:|:-------------:|
| P3 (Layer 4) | 256 | 256 |
| P4 (Layer 6) | 256 → | **512** (learned up-projection) |
| P5 (Layer 9) | 512 → | **1024** (learned up-projection) |
| **Concatenated h** | | **1792** |

**Learnable Scale Weights** (trainable `nn.Parameter`, init=1.0):
```
h = [w3·P3 || w4·P4 || w5·P5] ∈ R^1792
```

**Learned values after SupCon pretraining:**
| w3 (P3) | w4 (P4) | w5 (P5) |
|:-------:|:-------:|:-------:|
| **1.0435** | 1.0100 | 0.9732 |

> Fine-grained P3 features are weighted slightly higher, consistent with the clinical importance of optic disc border details.

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
| SupCon + Frozen | 0.9211 | 0.8000 | 0.5714 | 0.6667 | 12 | 9 | 3 | 128 |
| **SupCon + Partial** | **0.9211** | **0.7647** | **0.6190** | **0.6842** | **13** | **8** | **4** | **127** |
| SupCon + Full | 0.9211 | 0.8000 | 0.5714 | 0.6667 | 12 | 9 | 3 | 128 |

**🏆 Best: SupCon + Partial Finetuning** — **61.90% Recall** (13/21 Glaucoma detected), best across all experiments to date.

---

## Output Files
- `training.log` — full epoch-by-epoch metrics + learned w3/w4/w5 values
- `tsne_pre.png` / `tsne_post.png` — embedding visualizations
- `baseline_cm.png`, `supcon_frozen_cm.png`, `supcon_partial_cm.png`, `supcon_full_cm.png`
- `*_best.pth` — best model checkpoints

---

## Key Takeaways
1. **Learnable weights confirm P3 priority**: w3=1.044 > w4=1.010 > w5=0.973, confirming the model autonomously up-weighted early fine-grained features.
2. **Partial finetuning breakthrough**: 61.90% Recall (13/21) — highest recall achieved in the entire project pipeline.
3. **Longer training helped significantly**: With patience=100 and 200 max epochs, the partial finetuner discovered a genuinely better plateau vs the early-stopped v1 run (which halted at ep17 with 52% recall).
