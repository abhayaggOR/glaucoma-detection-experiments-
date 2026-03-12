# runs5 — Grid Search over Class Weights

## What Was Done
**Grid search** over class weight ratios for the weighted cross entropy loss. Tested 4 different weight ratios × 2 models (YOLO11s + YOLO11l) = **8 training runs** total. Each model was also threshold-tuned (0.10–0.55).

## Weight Ratios Tested

| Config | GLAUCOMA_SUSPECT Weight | NORMAL Weight |
|:------:|:-----------------------:|:-------------:|
| 1 | 3.0 | 0.50 |
| 2 | 4.0 | 0.40 |
| 3 | 5.0 | 0.30 |
| 4 | 6.0 | 0.25 |

## Training Configuration
- **Epochs**: 300 (early stopping: patience=100 for small, patience=75 for large)
- **Batch Size**: 16 (small), 32 (large)
- **Image Size**: 512
- **Data**: Augmented (data_augment)

## Best Results (Best F1 per Configuration)

| Model | W_Glaucoma | W_Normal | Best Thresh | F1 | Precision | Recall | Accuracy |
|-------|:---------:|:-------:|:-----------:|:--:|:---------:|:------:|:--------:|
| yolo11s | 3.0 | 0.50 | 0.20 | 0.6400 | 0.5517 | 0.7619 | 0.8816 |
| yolo11l | 3.0 | 0.50 | 0.15 | 0.6939 | 0.6071 | 0.8095 | 0.9013 |
| yolo11s | 4.0 | 0.40 | 0.15 | 0.6818 | 0.6522 | 0.7143 | 0.9079 |
| yolo11l | 4.0 | 0.40 | 0.25 | 0.4490 | 0.3929 | 0.5238 | 0.8224 |
| yolo11s | 5.0 | 0.30 | 0.15 | 0.6939 | 0.6071 | 0.8095 | 0.9013 |
| yolo11l | 5.0 | 0.30 | 0.30 | 0.4783 | 0.4400 | 0.5238 | 0.8421 |
| **yolo11s** | **6.0** | **0.25** | **0.15** | **0.6977** | **0.6818** | **0.7143** | **0.9145** |
| **yolo11l** | **6.0** | **0.25** | **0.20** | **0.7234** | **0.6538** | **0.8095** | **0.9145** |

## Overall Winners
- **Best YOLO11s**: weights=[6.0, 0.25], F1=0.6977, Recall=0.7143
- **Best YOLO11l**: weights=[6.0, 0.25], **F1=0.7234, Recall=0.8095**

## Confusion Matrix at Best F1 Threshold

### Best YOLO11l (weights=6.0:0.25, thresh=0.20)

|  | Pred NORMAL | Pred GLAUCOMA |
|--|:-----------:|:-------------:|
| **True NORMAL** | TN = 122 | FP = 9 |
| **True GLAUCOMA** | FN = 4 | TP = 17 |

## Confusion Matrix Images
Per-config confusion matrices: `{model}_{weight_config}_confusion_matrix.png`

## Key Takeaway
Higher weights (6.0:0.25) performed best. **YOLO11l with weights=[6.0, 0.25]** achieved the highest F1 (0.7234) with **80.95% recall** — detects 17/21 glaucoma cases.
