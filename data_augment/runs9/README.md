# runs9 — EfficientNet-B2 with Label Smoothing

## What Was Done
Trained **EfficientNet-B2** (torchvision, pretrained on ImageNet1K_V1) on augmented data with **label smoothing (0.1)**.

## Training Configuration

| Parameter | Value |
|-----------|:-----:|
| Model | EfficientNet-B2 |
| Batch Size | 32 |
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
| Epochs Completed | 189 |
| Best Epoch | 89 |
| Training Time | 4.532 hours |

## Test Set Metrics (threshold = 0.50)

| Metric | Value |
|--------|:-----:|
| **Accuracy** | 0.5658 |
| **Precision** | 0.1538 |
| **Recall** | 0.4762 |
| **F1 Score** | 0.2326 |

## Confusion Matrix on Test Data (threshold = 0.50)

|  | Pred NORMAL | Pred GLAUCOMA |
|--|:-----------:|:-------------:|
| **True NORMAL** | TN = 76 | FP = 55 |
| **True GLAUCOMA** | FN = 11 | TP = 10 |

## Confusion Matrix Images (test data)
- `efficientnet_b2_confusion_matrix_test.png`
- `efficientnet_b2_confusion_matrix_test_normalized.png`

## Key Observations
- Better than B0/B1 at threshold=0.50 in terms of false positives (55 vs 118/124)
- Still poor precision (15.4%) and moderate recall (47.6%)
- Trained the longest (189 epochs) — model kept learning but struggled to generalize
