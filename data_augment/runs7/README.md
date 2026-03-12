# runs7 — EfficientNet-B0 with Label Smoothing

## What Was Done
Trained **EfficientNet-B0** (torchvision, pretrained on ImageNet1K_V1) on augmented data with **label smoothing (0.1)**.

## Training Configuration

| Parameter | Value |
|-----------|:-----:|
| Model | EfficientNet-B0 |
| Batch Size | 16 |
| Epochs (max) | 300 |
| Early Stopping | patience=100 |
| Image Size | 512 |
| Label Smoothing | 0.1 |
| Pretrained | ImageNet1K_V1 |
| Data | Augmented (data_augment) |
| Optimizer | AdamW (auto lr=0.001667) |

## Training Summary

| Metric | Value |
|--------|:-----:|
| Epochs Completed | 186 |
| Best Epoch | 86 |
| Training Time | 4.254 hours |

## Test Set Metrics (threshold = 0.50)

| Metric | Value |
|--------|:-----:|
| **Accuracy** | 0.2039 |
| **Precision** | 0.1324 |
| **Recall** | 0.8571 |
| **F1 Score** | 0.2293 |

## Confusion Matrix on Test Data (threshold = 0.50)

|  | Pred NORMAL | Pred GLAUCOMA |
|--|:-----------:|:-------------:|
| **True NORMAL** | TN = 13 | FP = 118 |
| **True GLAUCOMA** | FN = 3 | TP = 18 |

## Confusion Matrix Images (test data)
- `efficientnet_b0_confusion_matrix_test.png`
- `efficientnet_b0_confusion_matrix_test_normalized.png`

## Key Observations
- High recall (85.7%) — detects 18/21 glaucoma cases
- Very low precision (13.2%) — too many false positives (118)
- The model heavily over-predicts GLAUCOMA_SUSPECT at threshold=0.50
- Threshold tuning towards higher thresholds could improve precision significantly
