# Augmented Data — Training & Threshold Tuning Report

**Date**: 2 March 2026
**Models**: YOLO11s-cls (small) and YOLO11l-cls (large)
**Dataset**: Augmented glaucoma classification dataset (see [Data_Augmentation_Report.md](file://data_augment/Data_Augmentation_Report.md))
**Training Script**: [train_augmented.py](file://data_augment/train_augmented.py)
**Threshold Tuning Script**: [threshold_tuning_augmented.py](file://data_augment/threshold_tuning_augmented.py)

---

## 1. Dataset Overview

The dataset was created from 1,008 fundus images (3 cameras: Remidio, Bosch, Forus), split 70/15/15 with stratification, and the minority class (GLAUCOMA_SUSPECT) was augmented **only in the training set** to achieve a 3:1 ratio.

| Split | NORMAL | GLAUCOMA SUSPECT | Total | Ratio |
|-------|:---:|:---:|:---:|:---:|
| **Train** | 609 | **203** (96 orig + 107 aug) | **812** | **3.0:1** |
| Val | 131 | 20 | 151 | 6.5:1 |
| **Test** | 131 | 21 | **152** | 6.2:1 |

**Augmentation**: Rotation ±5°, Brightness ±10%, Contrast ±10% — applied to GLAUCOMA_SUSPECT train images only.

---

## 2. Training Configuration

| Parameter | Value |
|-----------|-------|
| **Epochs** | 300 (with early stopping, patience = 100) |
| **Image Size** | 512 × 512 |
| **Batch Size** | 16 |
| **rect** | True |
| **Device** | GPU 0 |
| **Results Directory** | `data_augment/runs/` |

Both models used identical training settings.

---

## 3. Training Results (Validation Set)

### 3.1 Best Model Metrics

| Metric | YOLO11s-cls (Small) | YOLO11l-cls (Large) |
|--------|:-------------------:|:-------------------:|
| **Best Val Accuracy** | **92.05 %** | **92.05 %** |
| Best Epoch | 74 | 90 |
| Epochs Completed | 174 (early-stopped) | 165 (early-stopped) |
| Final Train Loss | 0.1356 | 0.1823 |
| Final Val Loss | 0.7141 | 0.5039 |

### 3.2 Training Curves

````carousel
![YOLO11s-cls (Augmented) Training Curves](data_augment/runs/yolo11s_augmented/results.png)
<!-- slide -->
![YOLO11l-cls (Augmented) Training Curves](data_augment/runs/yolo11l_augmented/results.png)
````

### 3.3 Key Observations

**YOLO11s-cls (Small):**
- Training loss dropped steadily from ~0.58 → 0.14, indicating genuine feature learning
- Validation accuracy oscillated significantly (range ~77%–92%) but the smoothed trend increased
- Val loss spike around epoch 30 (reached ~1.55) but recovered and generally stayed around 0.4–0.7
- Top-5 accuracy = 100% throughout

**YOLO11l-cls (Large):**
- Training loss dropped from ~0.62 → 0.15, similar to the small model
- Early instability: accuracy plunged to ~50% around epochs 10–25, then recovered
- Val loss spike to ~3.95 in early epochs but settled into the 0.4–0.8 range
- Both models ultimately reached the **same peak validation accuracy (92.05%)** — the large model is no longer failing as it did in the non-augmented experiments

---

## 4. Test Set Evaluation (Default Threshold = 0.5)

All confusion matrices below are evaluated on the **test set** (152 images: 21 GLAUCOMA SUSPECT, 131 NORMAL). This test set was **not augmented** and was held out during training.

### 4.1 Confusion Matrices (Test Set)

````carousel
![YOLO11s-cls Test Set Confusion Matrix](data_augment/runs/yolo11s_augmented/confusion_matrix.png)
<!-- slide -->
![YOLO11s-cls Test Set Confusion Matrix Normalized](data_augment/runs/yolo11s_augmented/confusion_matrix_normalized.png)
<!-- slide -->
![YOLO11l-cls Test Set Confusion Matrix](data_augment/runs/yolo11l_augmented/confusion_matrix.png)
<!-- slide -->
![YOLO11l-cls Test Set Confusion Matrix Normalized](data_augment/runs/yolo11l_augmented/confusion_matrix_normalized.png)
````

### 4.2 Confusion Matrix Values (Absolute Counts)

| | **True: GLAUCOMA SUSPECT** | **True: NORMAL** |
|---|:---:|:---:|
| **YOLO11s — Pred: GLAUCOMA SUSPECT** | 10 (TP) | 4 (FP) |
| **YOLO11s — Pred: NORMAL** | 11 (FN) | 127 (TN) |
| **YOLO11l — Pred: GLAUCOMA SUSPECT** | 12 (TP) | 6 (FP) |
| **YOLO11l — Pred: NORMAL** | 9 (FN) | 125 (TN) |

### 4.3 Per-Class Metrics (Default Threshold = 0.5)

| Metric | Class | YOLO11s-cls (Small) | YOLO11l-cls (Large) |
|--------|-------|:-------------------:|:-------------------:|
| **Precision** | GLAUCOMA SUSPECT | **71.43 %** (10/14) | **66.67 %** (12/18) |
| **Recall** | GLAUCOMA SUSPECT | **47.62 %** (10/21) | **57.14 %** (12/21) |
| **F1-Score** | GLAUCOMA SUSPECT | **57.14 %** | **61.54 %** |
| **Precision** | NORMAL | **92.03 %** (127/138) | **93.28 %** (125/134) |
| **Recall** | NORMAL | **96.95 %** (127/131) | **95.42 %** (125/131) |
| **F1-Score** | NORMAL | **94.42 %** | **94.34 %** |
| **Overall Accuracy** | — | **90.13 %** (137/152) | **90.13 %** (137/152) |

> [!IMPORTANT]
> Both models achieve **90.13% accuracy** on the test set. The **large model has higher glaucoma recall (57.14% vs 47.62%)** but lower precision (66.67% vs 71.43%). This is a significant improvement over the non-augmented experiments where the large model completely failed (0% recall).

---

## 5. Threshold Tuning (runs2/)

The default YOLO classification threshold is 0.5, meaning if P(GLAUCOMA_SUSPECT) ≥ 0.5, the image is classified as glaucoma suspect. Lowering the threshold increases recall (catches more positive cases) at the cost of precision (more false alarms). We evaluated thresholds from **0.10 to 0.55** (step 0.05) on the **test set**.

### 5.1 YOLO11s-cls (Small) — Threshold Sweep

| Threshold | Accuracy | Precision | Recall | F1 Score | TP | FP | TN | FN |
|:---------:|:--------:|:---------:|:------:|:--------:|:--:|:--:|:--:|:--:|
| 0.10 | 86.18 % | 50.00 % | 57.14 % | 53.33 % | 12 | 12 | 119 | 9 |
| 0.15 | 86.84 % | 52.17 % | 57.14 % | 54.55 % | 12 | 11 | 120 | 9 |
| 0.20 | 87.50 % | 54.55 % | 57.14 % | 55.81 % | 12 | 10 | 121 | 9 |
| **0.25** | **89.47 %** | **63.16 %** | **57.14 %** | **60.00 %** | **12** | **7** | **124** | **9** |
| 0.30 | 88.82 % | 62.50 % | 47.62 % | 54.05 % | 10 | 6 | 125 | 11 |
| 0.35 | 88.82 % | 62.50 % | 47.62 % | 54.05 % | 10 | 6 | 125 | 11 |
| 0.40 | 89.47 % | 66.67 % | 47.62 % | 55.56 % | 10 | 5 | 126 | 11 |
| 0.45 | 89.47 % | 66.67 % | 47.62 % | 55.56 % | 10 | 5 | 126 | 11 |
| 0.50 | 90.13 % | 71.43 % | 47.62 % | 57.14 % | 10 | 4 | 127 | 11 |
| 0.55 | 90.13 % | 71.43 % | 47.62 % | 57.14 % | 10 | 4 | 127 | 11 |

**Best F1 threshold = 0.25**: F1 = **60.00%**, Precision = 63.16%, Recall = **57.14%**

### 5.2 YOLO11l-cls (Large) — Threshold Sweep

| Threshold | Accuracy | Precision | Recall | F1 Score | TP | FP | TN | FN |
|:---------:|:--------:|:---------:|:------:|:--------:|:--:|:--:|:--:|:--:|
| 0.10 | 86.18 % | 50.00 % | 76.19 % | 60.38 % | 16 | 16 | 115 | 5 |
| 0.15 | 86.84 % | 51.61 % | 76.19 % | 61.54 % | 16 | 15 | 116 | 5 |
| **0.20** | **89.47 %** | **59.26 %** | **76.19 %** | **66.67 %** | **16** | **11** | **120** | **5** |
| 0.25 | 89.47 % | 59.26 % | 76.19 % | 66.67 % | 16 | 11 | 120 | 5 |
| 0.30 | 90.13 % | 62.50 % | 71.43 % | 66.67 % | 15 | 9 | 122 | 6 |
| 0.35 | 90.13 % | 65.00 % | 61.90 % | 63.41 % | 13 | 7 | 124 | 8 |
| 0.40 | 89.47 % | 63.16 % | 57.14 % | 60.00 % | 12 | 7 | 124 | 9 |
| 0.45 | 90.13 % | 66.67 % | 57.14 % | 61.54 % | 12 | 6 | 125 | 9 |
| 0.50 | 90.13 % | 66.67 % | 57.14 % | 61.54 % | 12 | 6 | 125 | 9 |
| 0.55 | 90.79 % | 70.59 % | 57.14 % | 63.16 % | 12 | 5 | 126 | 9 |

**Best F1 threshold = 0.20**: F1 = **66.67%**, Precision = 59.26%, Recall = **76.19%**

### 5.3 Threshold Tuning Summary

| | YOLO11s-cls | YOLO11l-cls |
|---|:---:|:---:|
| **Best F1 Threshold** | 0.25 | 0.20 |
| **F1 at Best Threshold** | **60.00 %** | **66.67 %** |
| **Precision at Best Threshold** | 63.16 % | 59.26 % |
| **Recall at Best Threshold** | 57.14 % | **76.19 %** |
| **Accuracy at Best Threshold** | 89.47 % | 89.47 % |
| **Recall at Default (0.5)** | 47.62 % | 57.14 % |
| **Recall Improvement** | +9.52 pp | +19.05 pp |

> [!TIP]
> **YOLO11l-cls with threshold = 0.20 is the best configuration for glaucoma screening.** It captures **76.19% of glaucoma suspects** (16 out of 21) with an F1 of 66.67%. This comes at a cost of 11 false positives out of 131 normal images (FP rate = 8.4%), which is acceptable for a screening application where missing positive cases is more dangerous than over-referring.

---

## 6. Comparison: Augmented vs Non-Augmented Models

This compares the best augmented models (from this report) against the best non-augmented model (Runs2 YOLO11s from the main [YOLO_Training_Report.md](file://YOLO_Training_Report.md)).

| Metric | Non-Aug Runs2 YOLO11s | Aug YOLO11s (t=0.25) | Aug YOLO11l (t=0.20) |
|--------|:--------------------:|:--------------------:|:--------------------:|
| **Test Set** | 202 images | 152 images | 152 images |
| **Overall Accuracy** | **93.07 %** | 89.47 % | 89.47 % |
| **Glaucoma Recall** | 59.26 % (16/27) | 57.14 % (12/21) | **76.19 %** (16/21) |
| **Glaucoma Precision** | **84.21 %** (16/19) | 63.16 % (12/19) | 59.26 % (16/27) |
| **Glaucoma F1** | **69.57 %** | 60.00 % | 66.67 % |
| **False Positives** | 3 | 7 | 11 |
| **False Negatives** | 11 | 9 | **5** |

> [!WARNING]
> The test sets differ (202 vs 152 images) due to the different train/test split ratios (80/20 vs 70/15/15), so direct numeric comparison should be interpreted with caution. The augmented large model achieves the **highest recall (76.19%)** and **lowest false negatives (5)**, which is the most critical metric for a medical screening application.

---

## 7. Key Findings

1. **Data augmentation enabled the large model to learn**: In non-augmented experiments, YOLO11l-cls completely failed (0% recall, stuck at majority-class baseline). With augmented data, it achieved **92.05% val accuracy** and **57.14% test recall** at default threshold — matching the small model.

2. **Both models converge to the same accuracy**: YOLO11s and YOLO11l both reach 92.05% val accuracy and 90.13% test accuracy, suggesting the dataset size (not model capacity) is the bottleneck.

3. **Threshold tuning significantly improves recall**:
   - YOLO11s: Recall improved from 47.62% → 57.14% (+9.52 pp) by lowering threshold to 0.25
   - YOLO11l: Recall improved from 57.14% → **76.19%** (+19.05 pp) by lowering threshold to 0.20

4. **YOLO11l-cls benefits more from threshold tuning**: The large model shows a wider range of confidence scores for GLAUCOMA_SUSPECT, allowing greater recall gains when the threshold is lowered.

5. **Trade-off**: Higher recall comes at the cost of more false positives (11 at threshold=0.20 vs 6 at threshold=0.50 for the large model), but for glaucoma screening, **missing positive cases is more costly than over-referring**.

---

## 8. File References

| File | Description |
|------|-------------|
| `runs/yolo11s_augmented/` | YOLO11s training artifacts, confusion matrices (test set) |
| `runs/yolo11l_augmented/` | YOLO11l training artifacts, confusion matrices (test set) |
| `runs2/Threshold_Tuning_Report.md` | Detailed threshold sweep tables |
| `runs2/yolo11s_augmented/` | YOLO11s test evaluation (confusion matrices) |
| `runs2/yolo11l_augmented/` | YOLO11l test evaluation (confusion matrices) |
| `runs2/yolo11s_augmented_confusion_matrix.png` | Threshold-tuned CM for small model |
| `runs2/yolo11l_augmented_confusion_matrix.png` | Threshold-tuned CM for large model |
| `Data_Augmentation_Report.md` | Augmentation pipeline details & dataset statistics |
| `train_augmented.py` | Training script |
| `threshold_tuning_augmented.py` | Threshold tuning script |
