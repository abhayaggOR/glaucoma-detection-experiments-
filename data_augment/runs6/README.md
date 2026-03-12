# runs6 — Focal Loss Hyperparameter Tuning

## What Was Done
Trained YOLO11s and YOLO11l with **Focal Loss** instead of standard cross entropy. Tuned focal loss hyperparameters (γ and α) across **4 configurations × 2 models = 8 training runs**. Each model was also threshold-tuned (0.10–0.55).

**Focal Loss**: FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
- **γ (gamma)**: focusing parameter — higher = more focus on hard examples
- **α (alpha)**: class weight — higher for minority class

## Configurations Tested

| Config | γ (gamma) | α_Glaucoma | α_Normal | Description |
|:------:|:---------:|:----------:|:--------:|:-----------:|
| 1 | 1.0 | 0.75 | 0.25 | Mild focusing |
| 2 | 2.0 | 0.75 | 0.25 | Standard focal |
| 3 | 2.0 | 0.80 | 0.20 | Stronger minority bias |
| 4 | 3.0 | 0.75 | 0.25 | Aggressive focusing |

## Training Configuration
- **Epochs**: 300 (early stopping: patience=100 for small, patience=75 for large)
- **Batch Size**: 16 (small), 32 (large)
- **Image Size**: 512
- **Data**: Augmented (data_augment)

## Best Results (Best F1 per Configuration)

| Model | γ | α_Glaucoma | Best Thresh | F1 | Precision | Recall | Accuracy |
|-------|:-:|:----------:|:-----------:|:--:|:---------:|:------:|:--------:|
| yolo11s | 1.0 | 0.75 | 0.25 | 0.7556 | 0.7083 | 0.8095 | 0.9276 |
| yolo11l | 1.0 | 0.75 | 0.25 | 0.6667 | 0.6250 | 0.7143 | 0.9013 |
| yolo11s | 2.0 | 0.75 | 0.35 | 0.6667 | 0.5926 | 0.7619 | 0.8947 |
| yolo11l | 2.0 | 0.75 | 0.50 | 0.7234 | 0.6538 | 0.8095 | 0.9145 |
| **yolo11s** | **2.0** | **0.80** | **0.45** | **0.7692** | **0.8333** | **0.7143** | **0.9408** |
| yolo11l | 2.0 | 0.80 | 0.20 | 0.6909 | 0.5588 | **0.9048** | 0.8882 |
| yolo11s | 3.0 | 0.75 | 0.35 | 0.6087 | 0.5600 | 0.6667 | 0.8816 |
| yolo11l | 3.0 | 0.75 | 0.30 | 0.5714 | 0.4286 | 0.8571 | 0.8224 |

## Overall Winners
- **Best YOLO11s**: γ=2.0, α=0.80 — **F1=0.7692**, Precision=0.8333, Recall=0.7143
- **Best YOLO11l**: γ=2.0, α=0.75 — **F1=0.7234**, Recall=0.8095
- **Highest Recall**: YOLO11l γ=2.0 α=0.80 — **Recall=0.9048** (19/21 detected!)

## Confusion Matrix at Best Configs

### Best YOLO11s (γ=2.0, α=0.80, thresh=0.45)

|  | Pred NORMAL | Pred GLAUCOMA |
|--|:-----------:|:-------------:|
| **True NORMAL** | TN = 128 | FP = 3 |
| **True GLAUCOMA** | FN = 6 | TP = 15 |

### Highest Recall — YOLO11l (γ=2.0, α=0.80, thresh=0.20)

|  | Pred NORMAL | Pred GLAUCOMA |
|--|:-----------:|:-------------:|
| **True NORMAL** | TN = 116 | FP = 15 |
| **True GLAUCOMA** | FN = 2 | TP = 19 |

## Confusion Matrix Images
Per-config confusion matrices: `{model}_{config}_confusion_matrix.png`

## Key Takeaway
- **Focal loss with γ=2.0, α=0.80** gave the best overall YOLO11s result: **F1=0.7692** with only 3 false positives.
- **YOLO11l with γ=2.0, α=0.80** achieved **90.48% recall** — detected 19 out of 21 glaucoma cases (only missed 2).
- Focal loss outperformed both standard and grid-search weighted CE across most configurations.
