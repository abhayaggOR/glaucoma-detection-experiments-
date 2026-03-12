# runs8 — EfficientNet-B1 with Label Smoothing

## What Was Done
Trained **EfficientNet-B1** (torchvision, pretrained on ImageNet1K_V1) on augmented data with **label smoothing (0.1)**.

## Training Configuration

| Parameter | Value |
|-----------|:-----:|
| Model | EfficientNet-B1 |
| Batch Size | 16 |
| Epochs (max) | 300 |
| Early Stopping | patience=100 |
| Image Size | 512 |
| Label Smoothing | 0.1 |
| Pretrained | ImageNet1K_V1 |
| Data | Augmented (data_augment) |
| Optimizer | AdamW (auto) |

## Training Summary

| Metric | Value |
|--------|:-----:|
| Epochs Completed | 121 |
| Best Epoch | 21 |
| Training Time | 2.894 hours |

## Test Set Metrics (threshold = 0.50)

| Metric | Value |
|--------|:-----:|
| **Accuracy** | 0.1579 |
| **Precision** | 0.1206 |
| **Recall** | 0.8095 |
| **F1 Score** | 0.2099 |

## Confusion Matrix on Test Data (threshold = 0.50)

|  | Pred NORMAL | Pred GLAUCOMA |
|--|:-----------:|:-------------:|
| **True NORMAL** | TN = 7 | FP = 124 |
| **True GLAUCOMA** | FN = 4 | TP = 17 |

## Confusion Matrix Images (test data)
- `efficientnet_b1_confusion_matrix_test.png`
- `efficientnet_b1_confusion_matrix_test_normalized.png`

## Key Observations
- Best epoch at 21 suggests the model converged early and then degraded
- High recall (80.9%) but extremely low precision (12.1%)
- Almost all images predicted as GLAUCOMA_SUSPECT (124 false positives)
- Worst overall performance among the 4 EfficientNet variants
