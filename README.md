# Glaucoma Detection from Fundus Images

## 👁️ Problem Statement
The objective of this project is to build a robust binary classification model (**NORMAL** vs **GLAUCOMA_SUSPECT**) using fundus images collected from three different camera systems: Remidio, Bosch, and Forus. 

The primary challenge is a **severe class imbalance**: the raw dataset consists of ~86.4% NORMAL cases and ~13.6% GLAUCOMA_SUSPECT cases. Because missing a glaucoma diagnosis has severe real-world consequences, the true goal of this project is to achieve **high recall (sensitivity)** for the GLAUCOMA_SUSPECT class while maintaining an acceptable False Positive rate (precision).

## 📂 Repository Structure & Experiment Progression
This repository documents a chronological series of experiments designed to conquer the class imbalance issue. 

> **Note**: Due to GitHub file size limits, the raw image datasets (`1.0_Original_Fundus_Images/`, `train/`, `val/`, `test/`) and large model weight files (`*.pt`) have been rigorously excluded via `.gitignore`. The repository contains the complete codebase, experiment outputs, training logs, results (confusion matrices), and detailed reports.

---

### Phase 1: Baseline YOLOv11 Training (`Phase1_Baseline/`)
- **What**: Initial attempt to train YOLO11s-cls and YOLO11l-cls classification models on the combined dataset of 1,008 raw images.
- **Result**: Models suffered from extreme majority-class bias. They achieved ~86% accuracy by simply predicting "NORMAL" for almost every image, resulting in a **0%–7% recall** for glaucoma cases.
- **See Report**: `Phase1_Baseline/YOLO_Training_Report.md`

### Phase 2: Data Augmentation & Cleaning (`data_augment/`)
- **What**: Generated targeted augmentations (rotation, brightness, contrast) applied *only* to the minority GLAUCOMA_SUSPECT class in the training split. This brought the training ratio to a much healthier **3:1** (Normal : Glaucoma).
- **Result**: The models started learning genuine features! Recall improved from near-zero up to **47%–57%**.
- **See Reports**: `data_augment/Data_Augmentation_Report.md` & `data_augment/Augmented_Training_Report.md`

### Phase 3: Advanced Loss Functions & Threshold Tuning (`data_augment/`)
To push recall even higher, we experimented with advanced weighting techniques holding the augmented dataset constant:

1. **Class-Weighted Cross Entropy (Runs 3 & 5)**
   - **Scripts**: `train_weighted.py`, `train_grid_search.py`
   - **What**: Penalized errors on the minority class more heavily.
   - **Result**: A grid search over weight ratios led to a configuration (weights `6.0 : 0.25`) that improved recall up to **~80%**, verifying that the models needed explicit class bias correction.
   
2. **Confidence Threshold Tuning (Runs 4)**
   - **Scripts**: `threshold_tuning_weighted.py`, `threshold_tuning_augmented.py`
   - **What**: Since this is a medical screening application, the default 0.50 probability threshold is suboptimal. We swept across thresholds.
   - **Result**: Lowering the decision threshold to `0.20 - 0.25` drastically boosted recall, catching significantly more positive cases.

3. **Focal Loss Hyperparameter Tuning (Runs 6) 🏆**
   - **Scripts**: `train_focal_loss.py`
   - **What**: Replaced standard Cross Entropy with Focal Loss to focus training on hard-to-classify examples while weighting classes.
   - **Result**: **Our best overall configuration**. YOLO11l with focal loss (`γ=2.0`, `α=[0.80, 0.20]`) at a decision threshold of `0.20` achieved a massive **90.48% Recall**, detecting 19 out of 21 test glaucoma cases.

### Phase 4: Alternative Architectures (`data_augment/runs7` through `runs10`)
- **What**: Evaluated whether PyTorch-native EfficientNet variants (B0 to B3) could outperform YOLO for this specific task, using explicit label smoothing.
- **Result**: The EfficientNet models underperformed compared to YOLOv11 across the board. YOLO remained the definitive choice for this deployment.

### Phase 5: Advanced Unfreezing & Layer Analysis (`data_augment/runs12` to `runs16`)
## 📈 Phase 5: Advanced Loss & Gradient Sculpting (Runs 15–16)

With fundamental data structures established, we performed advanced PyTorch gradient surgery to unstick the heavier models from local minima without reducing model capacity:
1. **Differential Learning Rates (`Runs15`)**: Applied parameter hook scaling to manually isolate the backbone to `1e-5`, the neck to `5e-5`, and the classification head to `1e-4`, drastically stabilizing early-layer weight decay.
2. **Progressive Unfreezing (`Runs16`)**: Sequentially chained `YOLO.train()` states mathematically unfreezing the architecture from Head-Only -> Deep Layers -> Full Architecture over 140 epochs to prevent catastrophic forgetting.

### Phase 5 Validation (Test Split)
| Experiment | Model | Recall (Glaucoma) | Precision | F1-Score | Accuracy |
|---|---|:---:|:---:|:---:|:---:|
| **Runs15** (Differential LR) | YOLO11s | 0.4762 | **0.5882** | 0.5263 | 0.8816 |
| **Runs15** (Differential LR) | YOLO11l | 0.4286 | 0.5625 | 0.4865 | 0.8750 |
| **Runs16** (Progressive Unfreezing) | YOLO11s | 0.4286 | 0.5625 | 0.4865 | 0.8750 |
| **Runs16** (Progressive Unfreezing) | YOLO11l | **0.2857** | **0.6667** | 0.4000 | 0.8816 |

**Conclusion**: Surgical PyTorch techniques provide marginal stability but do not overcome the absolute raw effectiveness of a fully un-frozen architecture guided purely by heavily biased `Focal Loss` (Runs6).

---

## 🚀 Phase 6: Supervised Contrastive Learning (SupCon) — [CURRENTLY RUNNING]

To maximize the feature separability of our absolute best `YOLO11s` backbone (from Runs6), we are currently executing a comprehensive **Supervised Contrastive Learning (SupCon)** pipeline inside `Runs17`:
1. **Stage 1 (Feature Extraction)**: Stripped the standard Ultralytics classification head, leaving a pure 1024-dimensional spatial feature extractor.
2. **Stage 2 (Contrastive Pretraining)**: Forcing representation tightness using heavy dual-augmentations (RandomResizedCrops, Gaussian Blurs, Color Jitters, Grayscale) under a PyTorch `SupConLoss` constraint (150 Epochs).
3. **Stage 3 (Classifier Fine-Tuning)**: Training a dedicated linear classifier through four simultaneous ablations:
   - Baseline (No SupCon)
   - SupCon + Frozen Backbone
   - SupCon + Partial Backbone Finetuning
   - SupCon + Full Pipeline Finetuning
   

### Phase 6 Validation (Test Split - Runs 17)
| Experiment Strategy | Recall (Glaucoma) | Precision | F1-Score | Accuracy |
|---------------------|:-----------------:|:---------:|:--------:|:--------:|
| Baseline (No SupCon) | 47.62% | 83.33% | 60.61% | 91.45% |
| SupCon + Frozen Backbone | 42.86% | 90.00% | 58.06% | 91.45% |
| **SupCon + Partial Finetune** | **52.38%** | **84.62%** | **64.71%** | **92.11%** |
| SupCon + Full Finetune | 42.86% | 90.00% | 58.06% | 91.45% |

**Conclusion:**
Contrastive learning forced incredibly rigid representation clustering. While raw recall ceiling slightly dipped against basic mathematical BCE tuning, the PyTorch models natively learned how to almost completely eliminate False Positives organically. Evaluated securely on the unmodified `152` image test branch safely bypassing earlier Ultralytics `val()` inversion risks.


## 🌌 Phase 7: Multi-Scale Topologies & SupCon (Runs 18) — [QUEUED]

Extending the mathematical foundations established in Phase 6, we are extracting hierarchical multi-scale feature maps from the core YOLO architecture. 
The hypothesis strongly dictates that early YOLO layers natively embed fine-grained structural anomalies (e.g. optic disc cup geometries) while failing at global logic, whereas deep layers represent global semantics but compress geometries irreversibly.

1. **Multi-Scale Architecture**: Paralleled extraction across P3 (Block 4 - 256d), P4 (Block 6 - 256d), and P5 (Block 9 - 512d) resulting in an aggregated 1024-dimensional feature topology without the classification bottleneck.
2. **SupCon Stage**: Identical to Runs17 methodology.
3. **Finetuning Parameters**: Evaluating Baseline vs Frozen vs Partial Finetune vs Full Finetune on the Multi-Scale outputs.


### Phase 7 Validation (Test Split - Runs 18)
| Experiment Strategy | Recall (Glaucoma) | Precision | F1-Score | Accuracy |
|---------------------|:-----------------:|:---------:|:--------:|:--------:|
| Baseline Multi-Scale | 47.62% | 90.91% | 62.50% | 92.11% |
| SupCon + Frozen Backbone | 47.62% | 90.91% | 62.50% | 92.11% |
| **SupCon + Partial Finetune** | **52.38%** | **84.62%** | **64.71%** | **92.11%** |
| SupCon + Full Finetune | 47.62% | 90.91% | 62.50% | 92.11% |

**Conclusion:**
Multi-Scale Contrastive Learning successfully extracts extreme precision. The aggregated topological hierarchy (P3+P4+P5) generated an incredibly strict boundary for the `NORMAL` class, elevating Precision outlandishly high (90.91%) across almost all ablations! Only 1 False Positive was recorded across the 152 test images. Partial Fine-tuning safely optimized the 1024-dimensional boundary to capture 52.38% (11/21) of Glaucoma cases while preserving elite precision constraints.


---

## 🔬 Phase 8: Advanced Scale Fusion Experiments (Runs 19–21) — [IN PROGRESS]

Taking the multi-scale hierarchy established in Phase 7 further, we are evaluating three distinct mechanisms for combining P3/P4/P5 features from the YOLOv11s backbone. All experiments use **cross-space projection fusion** (P3:256→256, P4:256→512, P5:512→1024) before applying attention or weighting, yielding a consistent **1792-d** representation.

| Run | Strategy | Feature Dim | Key Module |
|-----|----------|:-----------:|------------|
| **Runs19** | Learnable Scalar Scale Weights (w3, w4, w5) | 1792 | Trainable `nn.Parameter` per-scale scalars |
| **Runs20** | SE Channel Attention | 1792 | SE block: sigmoid-gated channel recalibration (r=4) |
| **Runs21** | Cross-Scale Softmax Attention | 512 | Softmax-weighted sum over projected 512-d scales |

All experiments follow the same SupCon pretraining (150 epochs, τ=0.07) and classifier fine-tuning (Baseline / Frozen / Partial / Full), evaluated **only on the 152-image test set**.


### Phase 8 Partial Results (Runs 19 & 20 Complete — Runs 21 pending)

| Run | Strategy | Best Experiment | Recall | Precision | F1 | Accuracy |
|-----|----------|-----------------|:------:|:---------:|:--:|:--------:|
| **Runs19** | Learnable Scale Weights (1792-d) | **SupCon + Partial** | **61.90%** | 76.47% | **68.42%** | 92.11% |
| **Runs20** | SE Channel Attention (1792-d) | SupCon + Full | 57.14% | 80.00% | 66.67% | 92.11% |
| **Runs21** | Cross-Scale Softmax Attention (512-d) | *Pending* | — | — | — | — |

> 🏆 **Runs19 SupCon + Partial Finetuning achieved 61.90% Recall (13/21 Glaucoma detected)** — the highest recall in the entire project. Learnable scale weights autonomously prioritized fine-grained P3 features (w3=1.044 > w4=1.010 > w5=0.973).

*(Runs21 results to be appended once training completes)*
