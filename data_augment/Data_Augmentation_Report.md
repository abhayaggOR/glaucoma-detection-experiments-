# Data Augmentation Report — Glaucoma Classification

**Generated**: Script `data_augment_pipeline.py`
**Seed**: 42
**Target Train Ratio**: Normal:Glaucoma = 3:1

---

## 1. Source Dataset

| Camera | Images in CSV | Matched on Disk | NORMAL | GLAUCOMA SUSPECT |
|--------|:---:|:---:|:---:|:---:|
| **Bosch** | 104 | 103 | 84 | 19 |
| **Forus** | 95 | 95 | 79 | 16 |
| **Remidio** | 810 | 810 | 708 | 102 |
| **Total** | — | **1008** | **871** | **137** |

Original ratio: **6.4:1** (Normal:Glaucoma)

---

## 2. Train / Val / Test Split

| Parameter | Value |
|-----------|-------|
| Split Ratio | **70 / 15 / 15** |
| Stratification | Yes — by Majority Decision |
| Random Seed | 42 |

### Before Augmentation

| Split | NORMAL | GLAUCOMA SUSPECT | Total | Ratio |
|-------|:---:|:---:|:---:|:---:|
| Train | 609 | 96 | 705 | 6.3:1 |
| Val | 131 | 20 | 151 | 6.5:1 |
| Test | 131 | 21 | 152 | 6.2:1 |

### After Augmentation (train only)

| Split | NORMAL | GLAUCOMA SUSPECT | Total | Ratio | Notes |
|-------|:---:|:---:|:---:|:---:|-------|
| **Train** | 609 | **203** (96 orig + 107 aug) | **812** | **3.0:1** ✅ | Augmented |
| Val | 131 | 20 | 151 | 6.5:1 | Untouched |
| Test | 131 | 21 | 152 | 6.2:1 | Untouched |

---

## 3. Augmentation Details

### Transforms Applied

| Transform | Range | Method |
|-----------|-------|--------|
| **Rotation** | ±5° | `PIL.Image.rotate()` with bilinear interpolation |
| **Brightness** | ±10% (factor 0.9–1.1) | `PIL.ImageEnhance.Brightness` |
| **Contrast** | ±10% (factor 0.9–1.1) | `PIL.ImageEnhance.Contrast` |

All three transforms are applied to every augmented image with independently randomized parameters.

### Parameter Tuning Decision

The augmentation parameters were initially set to more aggressive values:

| Transform | Previous (v1) | Updated (v2) |
|-----------|:---:|:---:|
| Rotation | ±30° | **±5°** |
| Brightness | ±20% (0.8–1.2) | **±10% (0.9–1.1)** |
| Contrast | ±20% (0.8–1.2) | **±10% (0.9–1.1)** |

These original parameters were found to be **too aggressive** for medical fundus images — large rotations and strong brightness/contrast shifts risked distorting clinically relevant features (e.g., optic disc appearance, cup-to-disc ratio cues). The parameters were therefore updated to more conservative values to ensure that augmented images remain realistic and diagnostically faithful while still providing meaningful variability for training.

### Image Quality

| Aspect | Detail |
|--------|--------|
| **Original copies** | `shutil.copy2()` — byte-for-byte identical, metadata preserved |
| **Augmented format** | JPEG, quality=95 (near-lossless) |
| **Resolution** | Same as original — no resizing |
| **Augmented naming** | `aug_<index>_<original_name>.jpg` |

### Augmentation Distribution

- Total glaucoma images in train: **96**
- Each image received **1** augmented copy(ies)
- **11** randomly selected images received 1 extra copy
- Total augmented images created: **107**

---

## 4. Output Directory Structure

```
data_augment/
├── train/
│   ├── NORMAL/              (609 originals)
│   └── GLAUCOMA_SUSPECT/    (203 = 96 orig + 107 aug)
├── val/
│   ├── NORMAL/              (131 originals)
│   └── GLAUCOMA_SUSPECT/    (20 originals)
├── test/
│   ├── NORMAL/              (131 originals)
│   └── GLAUCOMA_SUSPECT/    (21 originals)
├── train_labels.csv         (812 rows)
├── val_labels.csv           (151 rows)
├── test_labels.csv          (152 rows)
└── Data_Augmentation_Report.md
```

---

## 5. CSV Label Files

Each CSV contains the following columns:

| Column | Description |
|--------|-------------|
| `filename` | Image filename (matches file in the corresponding split folder) |
| `label` | `NORMAL` or `GLAUCOMA SUSPECT` |
| `source` | `original` or `augmented` |
| `camera` | Source camera: `Remidio`, `Bosch`, or `Forus` |
| `augmentation_params` | For augmented images: rotation, brightness, contrast values used |

---

## 6. Important Notes

1. **Original images are never modified** — all originals are byte-for-byte copies via `shutil.copy2()`
2. **Val and test sets are untouched** — they retain the original natural class distribution
3. **Augmentation is applied only to GLAUCOMA SUSPECT images in the train set**
4. **Augmented images use JPEG quality=95** — visually indistinguishable from originals
5. **Stratified splitting** ensures proportional representation of both classes across all splits
6. **Seed=42** for full reproducibility of splits and augmentation selection
