# Exploratory Data Analysis (EDA) Report — Chaksu Glaucoma Dataset

## 1. Overview

This report summarises the exploratory data analysis performed on the **Chaksu Glaucoma Dataset (ID: 20123135)**, which contains retinal fundus images captured by **three different fundus cameras** — Remidio, Bosch, and Forus. Each image was independently labelled by **5 expert ophthalmologists**, and a **Majority Decision** (≥ 3 out of 5 agree) determines the ground-truth label: **NORMAL** or **GLAUCOMA SUSPECT**.

The EDA was conducted in three Jupyter notebooks:

| Notebook | Camera | Location |
|----------|--------|----------|
| [eda_remidio_image.ipynb](file://6.0_Glaucoma_Decision/eda/eda_remidio_image.ipynb) | Remidio | Most extensive — covers CSV/label analysis, image-level EDA, and cross-dataset preparation |
| [eda_bosch_image.ipynb](file://6.0_Glaucoma_Decision/eda/eda_bosch_image.ipynb) | Bosch | Image-level EDA (resolution, brightness, contrast) |
| [eda_forus_image.ipynb](file://6.0_Glaucoma_Decision/eda/eda_forus_image.ipynb) | Forus | Image-level EDA (resolution, brightness, contrast) |

---

## 2. Dataset Summary

### 2.1 Image Counts

| Camera | Images on Disk | Images in CSV | Format | Notes |
|--------|:--------------:|:-------------:|--------|-------|
| **Remidio** | **810** | 810 | `.jpg` / `.JPG` | Largest subset; CSV originally had `.tif` filenames that needed transformation |
| **Bosch** | **104** | 145 | `.JPG` | **41 images in CSV have no matching file on disk** — significant mismatch |
| **Forus** | **95** | 95 | `.png` | Full match between CSV and disk |
| **Total** | **1,009** | **1,050** | — | 1,008 usable after filtering (1 Bosch image missing in later pipeline) |

> [!IMPORTANT]
> The Bosch dataset has a **28% mismatch** between CSV entries (145) and actual images on disk (104). This means 41 labelled samples have no corresponding image. This was handled by filtering to only images that physically exist.

### 2.2 Label Distribution (Majority Decision)

| Camera | NORMAL | GLAUCOMA SUSPECT | Total | Glaucoma % |
|--------|:------:|:----------------:|:-----:|:----------:|
| **Remidio** | 708 | 102 | 810 | 12.6% |
| **Bosch** | 125 | 20 | 145 (CSV) | 13.8% |
| **Forus** | 79 | 16 | 95 | 16.8% |
| **Combined** | **912** | **138** | **1,050** | **~13.1%** |

> [!WARNING]
> **Severe class imbalance** across all three cameras — GLAUCOMA SUSPECT comprises only **12.6–16.8%** of each dataset. This is a critical consideration for model training, as naive classifiers can achieve ~86% accuracy simply by predicting NORMAL for everything.

---

## 3. Expert Label Analysis (Remidio)

The Remidio notebook performed the most detailed label analysis, examining individual expert annotations.

### 3.1 Per-Expert Label Counts

| Label | Expert 1 | Expert 2 | Expert 3 | Expert 4 | Expert 5 |
|-------|:--------:|:--------:|:--------:|:--------:|:--------:|
| **NORMAL** | 665 | 535 | 694 | 678 | 736 |
| **GLAUCOMA SUSPECT** | 145 | 275 | 116 | 132 | 74 |

**Key observations:**
- **Expert 2 is the most aggressive** — flagged **275 images (34%)** as glaucoma suspect, nearly **4× more** than Expert 5
- **Expert 5 is the most conservative** — flagged only **74 images (9.1%)**
- There is also a **labelling typo** in Expert 5's annotations: `"GLAUCOMA  SUSUPECT"` (misspelling with extra space) — this was handled during EDA as the code searched for the substring `"GLAUCOMA"` to count votes

> [!NOTE]
> The wide disagreement between experts (9% to 34% glaucoma rate) highlights the **subjective nature of glaucoma screening** and validates the use of majority voting as the ground truth.

### 3.2 Expert Agreement Distribution

The number of experts (out of 5) who voted "GLAUCOMA SUSPECT" per image:

| Glaucoma Votes | Image Count | Interpretation |
|:--------------:|:-----------:|----------------|
| **0** | 440 | All experts agree: NORMAL (54.3%) |
| **1** | 185 | Mild disagreement — 1 expert flagged (22.8%) |
| **2** | 83 | Two experts flagged — still majority NORMAL (10.2%) |
| **3** | 41 | Borderline — majority says GLAUCOMA SUSPECT (5.1%) |
| **4** | 37 | Strong agreement on GLAUCOMA SUSPECT (4.6%) |
| **5** | 24 | All experts agree: GLAUCOMA SUSPECT (3.0%) |

**Key takeaway:** Only **24 images (3%)** have unanimous glaucoma agreement. The majority of glaucoma-suspect labels are **borderline cases** (3 out of 5 votes), meaning the model needs to learn subtle features.

---

## 4. Image-Level Analysis

### 4.1 Resolution Comparison

Each camera produces images at a **fixed, uniform resolution** — there is **zero variance** within each camera:

| Camera | Resolution (H × W) | Aspect Ratio | Megapixels | Orientation |
|--------|:-------------------:|:------------:|:----------:|:-----------:|
| **Remidio** | **3264 × 2448** | 0.75 (3:4 portrait) | 7.99 MP | Portrait |
| **Bosch** | **1440 × 1920** | 1.33 (4:3 landscape) | 2.76 MP | Landscape |
| **Forus** | **1536 × 2048** | 1.33 (4:3 landscape) | 3.15 MP | Landscape |

> [!IMPORTANT]
> **Remidio images are ~2.5–3× larger** than Bosch/Forus images in pixel count. When training models, all images are resized to a common size (512×512 or 1024×1024), meaning Remidio images lose more detail from downscaling while Bosch/Forus images are upscaled. Additionally, **Remidio images are portrait-oriented** while the other two are landscape, which affects how aspect-ratio-preserving modes (like `rect=True` in YOLO) handle padding.

### 4.2 Brightness Distribution

Mean pixel intensity (grayscale, 0–255) was computed for every image across all three cameras:

| Statistic | Remidio | Bosch | Forus |
|-----------|:-------:|:-----:|:-----:|
| **Darkest image** | 11.8 | 22.5 | 34.1 |
| **Brightest image** | 65.0 | 73.7 | 73.6 |
| **Typical range** | 15–40 | 22–74 | 34–74 |
| **Peak density** | 20–25 | — | — |

**Remidio brightness buckets** (detailed bucketing was performed):

| Brightness Range | Image Count |
|:----------------:|:-----------:|
| 10–15 | 12 |
| 15–20 | 135 |
| **20–25** | **257** (peak) |
| 25–30 | 180 |
| 30–35 | 115 |
| 35–40 | 74 |
| 40–45 | 25 |
| 45–50 | 6 |
| 50–65 | 6 |

**Key observations:**
- **All datasets are relatively dark** — fundus images have low overall brightness (mean intensities mostly below 75 on a 0–255 scale)
- **Remidio is the darkest** — its peak is at 20–25 intensity, with 12 images below 15 (extremely dark, potentially problematic)
- **Bosch and Forus are brighter** on average — their darkest images (~22 and ~34) are still brighter than many Remidio images
- Very dark images (< 15 intensity) could be **quality outliers** that hurt model training

**Darkest images identified:**

| Camera | Filename | Brightness | Contrast |
|--------|----------|:----------:|:--------:|
| Remidio | IMG_2702.JPG | 11.8 | 17.1 |
| Remidio | IMG_2701.JPG | 12.6 | 17.4 |
| Remidio | IMG_3212.JPG | 13.3 | 19.3 |
| Bosch | Image141.JPG | 22.5 | 23.3 |
| Bosch | Image176.JPG | 22.6 | 24.8 |
| Forus | 75.png | 34.1 | 35.1 |

### 4.3 Contrast Distribution

Contrast was measured as the standard deviation of pixel intensities (grayscale):

| Statistic | Remidio | Bosch | Forus |
|-----------|:-------:|:-----:|:-----:|
| **Lowest contrast** | ~17 | ~23 | ~24 |
| **Highest contrast** | ~66 | ~81 | ~58 |
| **Typical range** | 20–50 | 23–81 | 24–58 |

**Remidio contrast buckets:**

| Contrast Range | Image Count |
|:--------------:|:-----------:|
| 10–15 | 0 |
| 15–20 | 9 |
| 20–25 | 106 |
| **25–30** | **221** (peak) |
| **30–35** | **220** |
| 35–40 | 123 |
| 40–45 | 79 |
| 45–50 | 31 |
| 50–55 | 15 |
| 55–70 | 6 |

**Key observations:**
- **Bosch images have the highest contrast range** (23–81), suggesting more variation in image quality
- **Remidio contrast peaks at 25–35** — most images have moderate contrast
- **Very low contrast images** (< 20) exist in Remidio (9 images) — these may appear "washed out" and could be difficult for models to classify

**Brightest / highest-contrast images identified:**

| Camera | Filename | Brightness | Contrast |
|--------|----------|:----------:|:--------:|
| Bosch | Image119.JPG | 73.7 | 81.2 |
| Bosch | Image106.JPG | 73.1 | 81.2 |
| Forus | 78.png | 73.6 | 47.5 |
| Remidio | IMG_2566.JPG | 65.0 | 65.6 |

---

## 5. Data Preprocessing Steps Performed During EDA

### 5.1 Remidio CSV Transformation

The original Remidio CSV (`remidio.csv`) had image filenames in the format:
```
17521.tif-17521-1.tif
```
But the actual image files on disk were named:
```
17521.jpg
```

The EDA notebook:
1. Extracted the base numeric ID from the original `.tif` compound filenames
2. Created a new `image_name_jpg` column with `.jpg` extensions
3. Replaced the `Images` column and saved as `remidio_images_updated.csv`

> [!NOTE]
> This CSV transformation was a **critical preprocessing step** — without it, the training pipeline would fail to match CSV labels to actual image files.

### 5.2 Balanced Dataset Creation

To address the severe class imbalance, the EDA created **balanced CSV files** for each camera by **undersampling the majority class** (NORMAL) to match the minority class (GLAUCOMA SUSPECT):

| Camera | Original NORMAL | Original GLAUCOMA | Balanced (each class) | Total Balanced |
|--------|:---------------:|:-----------------:|:---------------------:|:--------------:|
| **Remidio** | 708 | 102 | **102** | **204** |
| **Bosch** | 125 | 20 | **20** | **40** |
| **Forus** | 79 | 16 | **16** | **32** |
| **Combined** | — | — | **138** | **276** |

A final **combined balanced dataset** was created by merging all three balanced CSVs:
- **276 total images** (138 NORMAL + 138 GLAUCOMA SUSPECT)
- Saved as `combined_balanced_all_cameras.csv` with a `camera` column to track source

> [!WARNING]
> Undersampling discards **~73% of NORMAL images** (from ~912 to 138). While this achieves perfect balance, it drastically reduces the training data. Alternative approaches (oversampling, SMOTE, weighted loss) might preserve more information.

---

## 6. Cross-Camera Comparison Summary

| Property | Remidio | Bosch | Forus |
|----------|:-------:|:-----:|:-----:|
| **Image count** | 810 | 104 | 95 |
| **Share of total** | 80.3% | 10.3% | 9.4% |
| **Resolution** | 3264×2448 | 1440×1920 | 1536×2048 |
| **Format** | JPG | JPG | PNG |
| **Orientation** | Portrait | Landscape | Landscape |
| **Brightness range** | 12–65 | 22–74 | 34–74 |
| **Contrast range** | 17–66 | 23–81 | 24–58 |
| **Glaucoma rate** | 12.6% | 13.8% | 16.8% |
| **CSV-disk mismatch** | 0 | 41 missing | 0 |

---

## 7. Key Takeaways & Implications for Model Training

| # | Finding | Implication |
|---|---------|-------------|
| 1 | **Severe class imbalance** (~86% NORMAL) across all cameras | Models may default to predicting majority class; need class weighting, oversampling, or focal loss |
| 2 | **Remidio dominates** the dataset (80.3% of images) | Model may overfit to Remidio's imaging characteristics; Bosch/Forus underrepresented |
| 3 | **Different resolutions & orientations** across cameras | Resizing to a common input size affects cameras differently; Remidio loses the most detail |
| 4 | **Remidio images are significantly darker** than Bosch/Forus | Domain shift between cameras; brightness normalization or augmentation could help |
| 5 | **Expert disagreement is high** — labelling rates range from 9% to 34% | The ground truth itself is noisy; many borderline cases (3/5 votes) make classification inherently difficult |
| 6 | **Only 24 images (3%) have unanimous glaucoma agreement** | True "easy" glaucoma cases are rare; the model must learn subtle optic disc features |
| 7 | **41 Bosch images in CSV have no matching file** | Data integrity issue; reduces usable Bosch data from 145 to 104 |
| 8 | **Remidio CSV required filename transformation** (`.tif` → `.jpg`) | Original CSV filenames did not match disk filenames; preprocessing was needed |
| 9 | **Balanced dataset created via undersampling** (276 images) | Achieves class balance but discards 73% of data; might consider augmentation instead |
| 10 | **All cameras have zero resolution variance** within themselves | No within-camera resolution preprocessing required |

---

## 8. Source Files

| File | Description |
|------|-------------|
| [eda_remidio_image.ipynb](file://6.0_Glaucoma_Decision/eda/eda_remidio_image.ipynb) | Full Remidio EDA — labels, experts, images, CSV transformation, balanced dataset creation |
| [eda_bosch_image.ipynb](file://6.0_Glaucoma_Decision/eda/eda_bosch_image.ipynb) | Bosch image-level EDA — resolution, brightness, contrast |
| [eda_forus_image.ipynb](file://6.0_Glaucoma_Decision/eda/eda_forus_image.ipynb) | Forus image-level EDA — resolution, brightness, contrast |
| [remidio_images_updated.csv](file://6.0_Glaucoma_Decision/remidio_images_updated.csv) | Remidio CSV with fixed `.jpg` filenames |
| [combined_balanced_all_cameras.csv](file://6.0_Glaucoma_Decision/combined_balanced_all_cameras.csv) | Final balanced dataset (276 images, 138 per class) |
