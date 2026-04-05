# Runs21 — Cross-Scale Softmax Attention + SupCon

**Experiment**: Per-image **dynamic scale weighting** using learned softmax attention across projected P3/P4/P5 scales combined into a 512-d representation.

---

## Architecture

### Cross-Scale Attention Module
1. Extract P3 (256-d), P4 (256-d), P5 (512-d) via GAP
2. Project to common **512-d** space:
   - P3 → Linear(256, 512)
   - P4 → Linear(256, 512)
   - P5 → Linear(512, 512)
3. Compute softmax attention over 3 learned scalar scores:
```
α = softmax([a3, a4, a5])
h = α3·P3' + α4·P4' + α5·P5'  ∈ R^512
```

**Learned attention weights (after 150-epoch SupCon pretraining):**
| α3 (P3) | α4 (P4) | α5 (P5) |
|:-------:|:-------:|:-------:|
| **0.3378** | 0.3342 | 0.3280 |

> Near-uniform distribution: the model found all three scales roughly equally informative. Slight preference for P3 (fine-grained) consistent with Runs19 and 21 findings.

### SupCon Pretraining
- **Projection Head**: Linear(512→256) → ReLU → Linear(256→128) → L2 Norm
- **Loss**: SupConLoss (τ=0.07), AdamW LR=1e-3, CosineAnnealing
- **Epochs**: 150

### Classifier Fine-Tuning
- **Classifier**: Linear(512→2), CrossEntropy + Label Smoothing 0.1
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
| SupCon + Full | 0.9145 | 0.8333 | 0.4762 | 0.6061 | 10 | 11 | 2 | 129 |

**Best: SupCon + Partial Finetuning** — 57.14% Recall (12/21 Glaucoma detected), F1=64.86%.

Post-finetuning (full) attention weights: **α3=0.3376, α4=0.3341, α5=0.3283** — weights stabilized near-uniformly throughout all training stages.

---

## Output Files
- `training.log` — epoch metrics + α3/α4/α5 evolution every 10 epochs
- `tsne_pre.png` / `tsne_post.png` — embedding visualizations
- `baseline_cm.png`, `supcon_frozen_cm.png`, `supcon_partial_cm.png`, `supcon_full_cm.png`
- `*_best.pth` — best model checkpoints

---

## Key Takeaways
1. **Uniform attention convergence**: Softmax attention weights converged to near-uniform values (0.338/0.334/0.328), suggesting that for glaucoma classification on this dataset, no single scale dominates — all are equally important.
2. **512-d compression disadvantage**: Collapsing to 512-d via weighted sum loses the full 1792-d diversity available in Runs19/20, resulting in slightly lower recall vs Runs19 (57% vs 62%).
3. **Cross-scale vs scalar weights**: Runs19's simple scalar weights (w3, w4, w5 on the full 1792-d concatenation) outperform Runs21's more complex softmax attention mechanism, suggesting that feature diversity (keeping all 1792 dims) matters more than softer weighting.
