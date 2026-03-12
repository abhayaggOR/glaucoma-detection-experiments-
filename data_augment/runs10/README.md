# runs10 — EfficientNet-B3 with Label Smoothing

## What Was Done
Trained **EfficientNet-B3** (torchvision, pretrained on ImageNet1K_V1) on augmented data with **label smoothing (0.1)**.

## Training Configuration

| Parameter | Value |
|-----------|:-----:|
| Model | EfficientNet-B3 |
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
| Epochs Completed | 125 |
| Best Epoch | 25 |
| Training Time | 3.156 hours |

## Test Set Metrics (threshold = 0.50)

| Metric | Value |
|--------|:-----:|
| **Accuracy** | 0.8355 |
| **Precision** | 0.2500 |
| **Recall** | 0.0952 |
| **F1 Score** | 0.1379 |

## Confusion Matrix on Test Data (threshold = 0.50)

|  | Pred NORMAL | Pred GLAUCOMA |
|--|:-----------:|:-------------:|
| **True NORMAL** | TN = 125 | FP = 6 |
| **True GLAUCOMA** | FN = 19 | TP = 2 |

## Confusion Matrix Images (test data)
- `efficientnet_b3_confusion_matrix_test.png`
- `efficientnet_b3_confusion_matrix_test_normalized.png`

## Key Observations
- Highest accuracy (83.5%) but almost entirely by predicting NORMAL
- Very low recall (9.5%) — only detects 2/21 glaucoma cases
- Best epoch at 25 suggests the larger model struggled to learn useful features
- Fewest false positives (6) but most false negatives (19)

---

## Overall EfficientNet Comparison (runs7–10)

| Model | Runs | Epochs | Best | Accuracy | Precision | Recall | F1 |
|-------|:----:|:------:|:----:|:--------:|:---------:|:------:|:--:|
| B0 | runs7 | 186 | 86 | 0.2039 | 0.1324 | **0.8571** | 0.2293 |
| B1 | runs8 | 121 | 21 | 0.1579 | 0.1206 | 0.8095 | 0.2099 |
| B2 | runs9 | 189 | 89 | 0.5658 | 0.1538 | 0.4762 | 0.2326 |
| B3 | runs10 | 125 | 25 | 0.8355 | 0.2500 | 0.0952 | 0.1379 |

**Note**: All EfficientNet models performed significantly worse than the YOLO models from runs3-6. The test set evaluation uses standard ImageNet preprocessing (Resize + Normalize), which may differ from the ultralytics training pipeline preprocessing — this likely contributed to the poor results.
