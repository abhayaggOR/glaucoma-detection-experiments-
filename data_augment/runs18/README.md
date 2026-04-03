# Runs18 Report — Multi-Scale Supervised Contrastive Learning

**Experiment**: Taking our backbone analysis to its apex by aggregating hierarchical contextual structures (P3+P4+P5) across YOLOv11s before mapping representations in Supervised Contrastive Learning.
**Objective**: Force the embedding space to separate Glaucoma vs Normal not just by deep global semantic logic (P5), but by utilizing fine-grained texture gradients like Optic Disc rims, natively contained only in early layers (P3, P4).

---

## 1. Pipeline Architecture

### Stage 1: Multi-Scale Feature Extraction
- **Backbone**: `YOLO11s-cls` initialized with `runs6` weights.
- **Modification**: Passed inputs through internal sequence blocks. Intercepted parallel tensors exactly at:
  - `P3` (Block 4): $256$ dimensions post-GAP.
  - `P4` (Block 6): $256$ dimensions post-GAP.
  - `P5` (Block 9 - C2PSA): $512$ dimensions post-GAP.
- **Concatenation**: $[h_3 || h_4 || h_5] \rightarrow h \in \mathbb{R}^{1024}$.

### Stage 2: SupCon Pretraining
- **Augmentation (Strong)**: Heavy multi-cropping and intensity blurs perfectly replicating `Runs17` standards.
- **Projection Head**: Linear mapping `1024 -> 512 -> ReLU -> 128 (L2 Normalized)`.
- **Loss**: SupCon Loss (Temperature $\tau = 0.07$).
- **Hyperparameters**: Batch 64, AdamW, lr=1e-3, 150 Epochs.

### Stage 3: Fine-Tuning 
Replaced projection layer with standard `Linear(1024, 2)` classifier to predict outputs directly from Multi-Scale embeddings across multiple ablation targets.

---

## 2. Test Set Evaluation Metrics (Unseen Splitting)

| Model State | Accuracy | Precision | Recall | F1 Score |
|-------------|:--------:|:---------:|:------:|:--------:|
| **Baseline Multi-Scale** | TBH | TBH | TBH | TBH |
| **SupCon (Frozen Backbone)** | TBH | TBH | TBH | TBH |
| **SupCon (Partial Finetune)** | TBH | TBH | TBH | TBH |
| **SupCon (Full Finetune)** | TBH | TBH | TBH | TBH |

---

## 3. Visualizations (Multi-Scale 1024 $\rightarrow$ 2D)

1. **Before SupCon**: `tsne_pre_supcon.png`
2. **After SupCon**: `tsne_post_supcon.png`

---

## 4. Key Takeaways

1. *To be calculated post training*
2. *To be calculated post training*
