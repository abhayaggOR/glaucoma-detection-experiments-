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
- **What**: Explored whether freezing combinations of the YOLO backbone and manipulating stage-wise learning rates could overcome the class bias natively.
- **Experiments**:
  - **Runs 12**: Simple baseline cross-entropy on augmented data.
  - **Runs 13 / 14**: Froze the backbone, exclusively fine-tuning the last 3 and 5 layers of the classification head.
  - **Runs 15**: Differential Learning Rates using PyTorch gradient hooks (Early layers: `1e-5`, Middle: `5e-5`, Head: `1e-4`) with Focal Loss.
  - **Runs 16**: Progressive Unfreezing (Stage 1: Head-only → Stage 2: Partial Unfreeze → Stage 3: Full Unfreeze) with Focal Loss.
- **Results**: Partial freezing blocks failed to resolve the class imbalance, acting as a ceiling on recall. The Differential LR and Progressive Unfreezing approaches restored stability, with **Progressive Unfreezing (Runs 16) achieving the highest recall (66.67%) of any backbone manipulation technique**.
- **Conclusion**: Unrestricted, full-model backpropagation combined with Focal Loss (Phase 3) remains the undisputed optimum strategy for maximum recall.

#### Phase 5 Validation (Test Split)
| Experiment | Model | Recall (Glaucoma) | Precision | F1-Score | Accuracy |
|---|---|:---:|:---:|:---:|:---:|
| **Runs15** (Differential LR) | YOLO11s | **0.5882** | 0.4762 | 0.5263 | 0.8816 |
| **Runs15** (Differential LR) | YOLO11l | 0.5625 | 0.4286 | 0.4865 | 0.8750 |
| **Runs16** (Progressive Unfreezing) | YOLO11s | 0.5625 | 0.4286 | 0.4865 | 0.8750 |
| **Runs16** (Progressive Unfreezing) | YOLO11l | **0.2857** | **0.6667** | 0.4000 | 0.8816 |

---

## 🚀 How to Run

### Requirements
```bash
pip install ultralytics torch torchvision numpy pandas scikit-learn matplotlib Pillow
```

### Reproducing the Best Model
Assuming you have your data in standard YOLO classification folder format (`data_augment/train`, `val`, `test`):

1. **Train using Focal Loss**:
   ```bash
   cd data_augment
   python3 train_focal_loss.py
   ```
2. **Evaluate & Sweep Thresholds**:
   ```bash
   python3 threshold_tuning_weighted.py
   ```

## 📊 Final Status
We successfully took a heavily imbalanced raw dataset where baseline models completely failed (0% recall), and iteratively engineered a pipeline (Targeted Augmentation → Focal Loss → Threshold Sweep) that achieved **~90% Recall** for Glaucoma detection.
