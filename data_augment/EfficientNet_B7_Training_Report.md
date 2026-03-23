# EfficientNet B7 Training Report — Augmented Data with Label Smoothing

**Dataset**: Augmented data (data_augment)
**Label Smoothing**: 0.1
**Image Size**: 512
**Epochs**: 300
**Pretrained**: ImageNet1K_V1

---

## Training Configuration

| Model | Runs Dir | Batch Size | Patience |
|-------|:--------:|:----------:|:--------:|
| EfficientNet-B7 | runs11 | 8 | 100 |

---

## Test Set Metrics (threshold = 0.50)

| Model | Accuracy | Precision | Recall | F1 Score | TP | FP | TN | FN |
|-------|:--------:|:---------:|:------:|:--------:|:--:|:--:|:--:|:--:|
| efficientnet_b7 | 0.8158 | 0.1111 | 0.0476 | 0.0667 | 1 | 8 | 123 | 20 |

---

## Confusion Matrices

### efficientnet_b7 (runs11)
- `efficientnet_b7_confusion_matrix_test.png`
- `efficientnet_b7_confusion_matrix_test_normalized.png`

