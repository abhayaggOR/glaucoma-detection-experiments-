# Runs17 Report — Supervised Contrastive Learning (SupCon) Pipeline

**Experiment**: Enhancing a pre-trained YOLOv11s backbone via Supervised Contrastive Learning.
**Objective**: To drastically improve glaucoma versus normal class separation (and thus recall) by generating heavy dual-crop views of each image and forcing representations in the projection space to cluster tightly based on class identity, independent of the background. 

---

## 1. Pipeline Architecture

### Stage 1: Feature Extraction
- **Backbone**: `YOLO11s-cls` initialized with `runs6` weights (best existing recall).
- **Modification**: Stripped classification head. Extracted features from `model[:8]` -> `AdaptiveAvgPool2d` -> `Flatten` -> `1024-dim` embedding.

### Stage 2: SupCon Pretraining
- **Augmentation (Strong)**: Generated two extremely augmented copies of each image using Random Resized Crops, Horizontal Flips, Color Jittering, Gaussian Blur, and Grayscale probability (20%).
- **Projection Head**: `Linear(1024->512) -> ReLU -> Linear(512->128) -> L2 Normalization`.
- **Loss**: PyTorch natively ported `SupConLoss` (Temperature \(\tau = 0.07\)).
- **Hyperparameters**: Batch 64, AdamW, lr=1e-3, 150 Epochs.

### Stage 3: Fine-Tuning 
After burning the SupCon topology into the backbone, the projection head is discarded, and a standard `Linear(1024, 2)` classifier is trained with CrossEntropy and Label Smoothing = 0.1 across four simultaneous ablation experiments.

---

## 2. Test Set Evaluation Metrics (Unseen Splitting)

| Model State | Accuracy | Precision | Recall | F1 Score |
|-------------|:--------:|:---------:|:------:|:--------:|
| **Baseline (Runs6 + New Head)** | TBH | TBH | TBH | TBH |
| **SupCon (Frozen Backbone)** | TBH | TBH | TBH | TBH |
| **SupCon (Partial Finetune)** | TBH | TBH | TBH | TBH |
| **SupCon (Full Finetune)** | TBH | TBH | TBH | TBH |

---

## 3. Visualizations

The experiment auto-generated extreme dimensionality reductions representing the 1024-dimension YOLOv11s feature space mathematically flattened to 2D using t-SNE.

1. **Before SupCon**: `tsne_pre_supcon.png`
2. **After SupCon**: `tsne_post_supcon.png`

---

## 4. Key Takeaways

1. *To be calculated post training*
2. *To be calculated post training*
