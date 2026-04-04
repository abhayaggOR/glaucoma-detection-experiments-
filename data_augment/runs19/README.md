# Runs19 — Learnable Scale Weights + Cross-Space Fusion + SupCon

**Experiment**: Multi-scale feature fusion (P3, P4, P5) with **learnable scalar weights** (w3, w4, w5) per scale applied after cross-space projection.

---

## Architecture

### Cross-Space Projection (Matching Spec Dims)
| Scale | Native Dim | Projected Dim |
|-------|:-----------:|:-------------:|
| P3 (Layer 4) | 256 | 256 |
| P4 (Layer 6) | 256 → | **512** |
| P5 (Layer 9) | 512 → | **1024** |
| **Concatenated h** | | **1792** |

**Learnable Scale Weights** are trainable `nn.Parameter` scalars initialized to 1.0:
```
h = [w3·P3 || w4·P4 || w5·P5] ∈ R^1792
```

### SupCon Pretraining
- **Projection Head**: Linear(1792→512) → ReLU → Linear(512→128) → L2 Norm
- **Loss**: SupConLoss (temperature τ=0.07)
- **Epochs**: 150, AdamW LR=1e-3, CosineAnnealing

### Classifier Fine-Tuning
4 ablations evaluated on **test set only**:
- Baseline (no SupCon)
- SupCon + Frozen Backbone
- SupCon + Partial Finetune (last 3 YOLO blocks + projections)
- SupCon + Full Finetune

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
- `training.log` — full epoch-by-epoch metrics + learned w3/w4/w5 values
- `tsne_pre.png` / `tsne_post.png` — embedding visualizations
- `*_cm.png` — confusion matrices (all on test data)
- `*_best.pth` — best model checkpoints (per experiment)
