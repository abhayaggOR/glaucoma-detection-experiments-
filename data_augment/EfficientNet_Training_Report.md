# EfficientNet B0–B3 Training Report — Augmented Data with Label Smoothing

**Dataset**: Augmented data (data_augment)
**Label Smoothing**: 0.1
**Image Size**: 512
**Epochs**: 300
**Pretrained**: ImageNet1K_V1

---

## Training Configuration

| Model | Runs Dir | Batch Size | Patience |
|-------|:--------:|:----------:|:--------:|
| EfficientNet-B0 | runs7  | 16 | 50 |
| EfficientNet-B1 | runs8  | 16 | 50 |
| EfficientNet-B2 | runs9  | 32 | 50 |
| EfficientNet-B3 | runs10 | 32 | 50 |

---

## Test Set Metrics (threshold = 0.50)

| Model | Accuracy | Precision | Recall | F1 Score | TP | FP | TN | FN |
|-------|:--------:|:---------:|:------:|:--------:|:--:|:--:|:--:|:--:|
| efficientnet_b0 | 0.2039 | 0.1324 | 0.8571 | 0.2293 | 18 | 118 | 13 | 3 |
| efficientnet_b1 | 0.1579 | 0.1206 | 0.8095 | 0.2099 | 17 | 124 | 7 | 4 |
| efficientnet_b2 | 0.5658 | 0.1538 | 0.4762 | 0.2326 | 10 | 55 | 76 | 11 |
| efficientnet_b3 | 0.8355 | 0.2500 | 0.0952 | 0.1379 | 2 | 6 | 125 | 19 |

---

## Confusion Matrices

### efficientnet_b0 (runs7)
- `efficientnet_b0_confusion_matrix_test.png`
- `efficientnet_b0_confusion_matrix_test_normalized.png`

### efficientnet_b1 (runs8)
- `efficientnet_b1_confusion_matrix_test.png`
- `efficientnet_b1_confusion_matrix_test_normalized.png`

### efficientnet_b2 (runs9)
- `efficientnet_b2_confusion_matrix_test.png`
- `efficientnet_b2_confusion_matrix_test_normalized.png`

### efficientnet_b3 (runs10)
- `efficientnet_b3_confusion_matrix_test.png`
- `efficientnet_b3_confusion_matrix_test_normalized.png`

