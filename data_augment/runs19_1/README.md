# Runs19.1 — Learnable Scale Weights + Cross-Space Fusion + SupCon @ 1024×1024

**Experiment**: Identical to Runs19, but with **input resolution increased from 512×512 to 1024×1024** to test whether higher resolution improves Glaucoma recall by preserving finer optic disc details.

---

## What Changed vs Runs19
| Parameter | Runs19 | Runs19.1 |
|-----------|:------:|:--------:|
| Image Size | 512×512 | **1024×1024** |
| All other params | — | **identical** |

---

## Architecture (same as Runs19)

| Component | Detail |
|-----------|--------|
| Backbone | YOLOv11s (Runs6, γ=2.0, α=0.80) |
| Feature Extraction | P3 (Layer4), P4 (Layer6), P5 (Layer9) via GAP |
| Cross-Space Projection | P3: 256→256, P4: 256→512, P5: 512→1024 |
| Learnable Scale Weights | w3, w4, w5 (init=1.0) |
| Feature Dim | **1792** |
| Projection Head | Linear(1792→512) → ReLU → Linear(512→128) → L2 Norm |
| Classifier | Linear(1792→2) |

## Hyperparameters

| Stage | Parameter | Value |
|-------|-----------|-------|
| **SupCon** | Epochs | 150 |
| | LR | 1e-3 |
| | Optimizer | AdamW (wd=1e-4) |
| | Temperature τ | 0.07 |
| | Scheduler | CosineAnnealingLR |
| **Fine-tune** | Max Epochs | 200 |
| | Patience | 100 |
| | LR (backbone) | 1e-5 |
| | LR (classifier) | 1e-3 |
| | Loss | CrossEntropy + LabelSmoothing=0.1 |

---

## Test Set Evaluation Results

> ✅ All metrics on **test set only** (152 images), scikit-learn with y=1 → Glaucoma.

| Experiment | Accuracy | Precision | Recall | F1-Score | TP | FN | FP | TN |
|-----------|:--------:|:---------:|:------:|:--------:|:--:|:--:|:--:|:--:|
| Baseline | TBD | TBD | TBD | TBD | - | - | - | - |
| SupCon + Frozen | TBD | TBD | TBD | TBD | - | - | - | - |
| SupCon + Partial | TBD | TBD | TBD | TBD | - | - | - | - |
| SupCon + Full | TBD | TBD | TBD | TBD | - | - | - | - |

**Comparison target (Runs19 best):** SupCon+Partial → Recall=**61.90%**, F1=**68.42%**

---

## Output Files
- `training.log` — epoch metrics + learned w3/w4/w5 values
- `tsne_pre.png` / `tsne_post.png`
- `*_cm.png` — confusion matrices
- `*_best.pth` — checkpoints
EOF
