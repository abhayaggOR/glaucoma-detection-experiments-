"""
Data Augmentation Pipeline for Glaucoma Classification
=======================================================
Rebalances the training set from ~6.3:1 to 3:1 (Normal:Glaucoma)
by augmenting GLAUCOMA SUSPECT images in the training split only.

Output: data_augment/ folder with train/val/test splits, CSV labels, and report.
"""

import os
import shutil
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image, ImageEnhance

# ==============================================================
# CONFIG
# ==============================================================
BASE_DIR = "/home/abhay/chaksu/20123135/Train"
OUTPUT_DIR = os.path.join(BASE_DIR, "data_augment")
RANDOM_SEED = 42
TARGET_RATIO = 3  # Normal:Glaucoma = 3:1

DATASETS = [
    {
        "csv": os.path.join(BASE_DIR, "6.0_Glaucoma_Decision/bosch_data.csv"),
        "img_dir": os.path.join(BASE_DIR, "1.0_Original_Fundus_Images/Bosch"),
        "camera": "Bosch",
    },
    {
        "csv": os.path.join(BASE_DIR, "6.0_Glaucoma_Decision/forus_data.csv"),
        "img_dir": os.path.join(BASE_DIR, "1.0_Original_Fundus_Images/Forus"),
        "camera": "Forus",
    },
    {
        "csv": os.path.join(BASE_DIR, "6.0_Glaucoma_Decision/remidio_images_updated.csv"),
        "img_dir": os.path.join(BASE_DIR, "1.0_Original_Fundus_Images/Remidio"),
        "camera": "Remidio",
    },
]

# Augmentation intensity ranges
ROTATION_RANGE = (-5, 5)         # degrees (±5°)
BRIGHTNESS_RANGE = (0.9, 1.1)   # ±10%
CONTRAST_RANGE = (0.9, 1.1)     # ±10%
JPEG_QUALITY = 95                # near-lossless


# ==============================================================
# AUGMENTATION FUNCTION
# ==============================================================
def augment_image(img_path, save_path):
    """
    Apply random rotation, brightness, and contrast adjustments.
    All three transforms are applied with random parameters.
    Saves as JPEG quality=95 to preserve image quality.
    """
    random.seed(None)  # ensure truly random augmentations
    img = Image.open(img_path).convert("RGB")

    # 1. Random rotation
    angle = random.uniform(*ROTATION_RANGE)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))

    # 2. Random brightness
    brightness_factor = random.uniform(*BRIGHTNESS_RANGE)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # 3. Random contrast
    contrast_factor = random.uniform(*CONTRAST_RANGE)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # Save with high quality
    img.save(save_path, "JPEG", quality=JPEG_QUALITY)
    return {
        "rotation": round(angle, 1),
        "brightness": round(brightness_factor, 2),
        "contrast": round(contrast_factor, 2),
    }


# ==============================================================
# STEP 1: LOAD & MERGE DATA
# ==============================================================
print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

dfs = []
camera_stats = []
for d in DATASETS:
    df = pd.read_csv(d["csv"])
    df["image_path"] = df["Images"].apply(lambda x: os.path.join(d["img_dir"], x))
    df["camera"] = d["camera"]
    before = len(df)
    df = df[df["image_path"].apply(os.path.exists)]
    after = len(df)
    n_normal = (df["Majority Decision"] == "NORMAL").sum()
    n_glaucoma = (df["Majority Decision"] == "GLAUCOMA SUSPECT").sum()
    print(f"  {d['camera']}: {before} in CSV, {after} on disk (N={n_normal}, G={n_glaucoma})")
    camera_stats.append({
        "camera": d["camera"],
        "csv_count": before,
        "disk_count": after,
        "normal": n_normal,
        "glaucoma": n_glaucoma,
    })
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
total_n = (data["Majority Decision"] == "NORMAL").sum()
total_g = (data["Majority Decision"] == "GLAUCOMA SUSPECT").sum()
print(f"\nTotal: {len(data)} images (NORMAL={total_n}, GLAUCOMA={total_g}, ratio={total_n/total_g:.1f}:1)")


# ==============================================================
# STEP 2: STRATIFIED 70/15/15 SPLIT
# ==============================================================
print("\n" + "=" * 60)
print("STEP 2: STRATIFIED 70/15/15 SPLIT")
print("=" * 60)

random.seed(RANDOM_SEED)

train_df, temp_df = train_test_split(
    data, test_size=0.30,
    stratify=data["Majority Decision"],
    random_state=RANDOM_SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50,
    stratify=temp_df["Majority Decision"],
    random_state=RANDOM_SEED
)

split_info = {}
for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    nn = (df["Majority Decision"] == "NORMAL").sum()
    gg = (df["Majority Decision"] == "GLAUCOMA SUSPECT").sum()
    ratio = nn / gg if gg > 0 else float("inf")
    split_info[name] = {"normal": nn, "glaucoma": gg, "total": len(df), "ratio": ratio}
    print(f"  {name.upper():5s}: {len(df)} images (N={nn}, G={gg}, ratio={ratio:.1f}:1)")


# ==============================================================
# STEP 3: CREATE OUTPUT DIRECTORY & COPY ORIGINALS
# ==============================================================
print("\n" + "=" * 60)
print("STEP 3: CREATING OUTPUT DIRECTORY & COPYING ORIGINALS")
print("=" * 60)

# Clean previous output if exists
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print(f"  Cleaned previous output at {OUTPUT_DIR}")

# Create directories
for split in ["train", "val", "test"]:
    for cls in ["NORMAL", "GLAUCOMA_SUSPECT"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# Copy images and build CSV label records
csv_records = {"train": [], "val": [], "test": []}

for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    copied = 0
    for _, row in split_df.iterrows():
        label = row["Majority Decision"]
        folder_label = label.replace(" ", "_")
        src = row["image_path"]
        filename = os.path.basename(src)
        dst = os.path.join(OUTPUT_DIR, split_name, folder_label, filename)
        shutil.copy2(src, dst)  # byte-for-byte copy with metadata
        csv_records[split_name].append({
            "filename": filename,
            "label": label,
            "source": "original",
            "camera": row["camera"],
            "augmentation_params": "",
        })
        copied += 1
    print(f"  {split_name.upper()}: copied {copied} original images")


# ==============================================================
# STEP 4: AUGMENT GLAUCOMA SUSPECT IN TRAINING SET
# ==============================================================
print("\n" + "=" * 60)
print("STEP 4: AUGMENTING GLAUCOMA SUSPECT (TRAIN ONLY)")
print("=" * 60)

train_glaucoma = train_df[train_df["Majority Decision"] == "GLAUCOMA SUSPECT"]
train_normal_count = split_info["train"]["normal"]
train_glaucoma_count = split_info["train"]["glaucoma"]

target_glaucoma = train_normal_count // TARGET_RATIO
aug_needed = target_glaucoma - train_glaucoma_count

print(f"  Train NORMAL: {train_normal_count}")
print(f"  Train GLAUCOMA (original): {train_glaucoma_count}")
print(f"  Target GLAUCOMA for {TARGET_RATIO}:1 ratio: {target_glaucoma}")
print(f"  Augmented images needed: {aug_needed}")

# Determine how many augmentations per image
# Each image gets at least 1 augmentation; some get 2
random.seed(RANDOM_SEED)
glaucoma_paths = train_glaucoma["image_path"].tolist()
glaucoma_cameras = train_glaucoma["camera"].tolist()
n_images = len(glaucoma_paths)

# Calculate: each image gets floor(aug_needed/n_images) copies,
# then randomly select remainder images for 1 extra copy
copies_per_image = aug_needed // n_images
remainder = aug_needed % n_images
extra_indices = set(random.sample(range(n_images), remainder))

print(f"  Base copies per glaucoma image: {copies_per_image}")
print(f"  Extra copy for {remainder} randomly selected images")
print()

aug_count = 0
aug_details = []
for idx, (img_path, camera) in enumerate(zip(glaucoma_paths, glaucoma_cameras)):
    n_copies = copies_per_image + (1 if idx in extra_indices else 0)
    for copy_idx in range(n_copies):
        aug_count += 1
        orig_name = os.path.basename(img_path)
        name_base, _ = os.path.splitext(orig_name)
        aug_filename = f"aug_{aug_count:03d}_{name_base}.jpg"
        save_path = os.path.join(OUTPUT_DIR, "train", "GLAUCOMA_SUSPECT", aug_filename)

        params = augment_image(img_path, save_path)

        csv_records["train"].append({
            "filename": aug_filename,
            "label": "GLAUCOMA SUSPECT",
            "source": "augmented",
            "camera": camera,
            "augmentation_params": f"rot={params['rotation']}° br={params['brightness']} ct={params['contrast']}",
        })
        aug_details.append({
            "aug_filename": aug_filename,
            "source_image": orig_name,
            "camera": camera,
            **params,
        })

        if aug_count % 20 == 0 or aug_count == aug_needed:
            print(f"  Augmented {aug_count}/{aug_needed} images...")

print(f"\n  ✅ Created {aug_count} augmented GLAUCOMA SUSPECT images")


# ==============================================================
# STEP 5: SAVE CSV LABELS
# ==============================================================
print("\n" + "=" * 60)
print("STEP 5: SAVING CSV LABELS")
print("=" * 60)

for split_name in ["train", "val", "test"]:
    csv_path = os.path.join(OUTPUT_DIR, f"{split_name}_labels.csv")
    df_csv = pd.DataFrame(csv_records[split_name])
    df_csv.to_csv(csv_path, index=False)
    n_orig = (df_csv["source"] == "original").sum()
    n_aug = (df_csv["source"] == "augmented").sum()
    print(f"  {split_name}_labels.csv: {len(df_csv)} rows (original={n_orig}, augmented={n_aug})")


# ==============================================================
# STEP 6: FINAL VERIFICATION
# ==============================================================
print("\n" + "=" * 60)
print("STEP 6: FINAL VERIFICATION")
print("=" * 60)

final_stats = {}
for split_name in ["train", "val", "test"]:
    normal_dir = os.path.join(OUTPUT_DIR, split_name, "NORMAL")
    glaucoma_dir = os.path.join(OUTPUT_DIR, split_name, "GLAUCOMA_SUSPECT")
    n_count = len(os.listdir(normal_dir))
    g_count = len(os.listdir(glaucoma_dir))
    total = n_count + g_count
    ratio = n_count / g_count if g_count > 0 else float("inf")
    final_stats[split_name] = {
        "normal": n_count, "glaucoma": g_count,
        "total": total, "ratio": ratio
    }
    marker = " ✅" if split_name == "train" and abs(ratio - TARGET_RATIO) < 0.5 else ""
    print(f"  {split_name.upper():5s}: NORMAL={n_count}, GLAUCOMA={g_count}, "
          f"Total={total}, Ratio={ratio:.1f}:1{marker}")

grand_total = sum(s["total"] for s in final_stats.values())
print(f"\n  Grand total: {grand_total} images")
print(f"  Augmented images added: {aug_count}")


# ==============================================================
# STEP 7: GENERATE REPORT
# ==============================================================
print("\n" + "=" * 60)
print("STEP 7: GENERATING REPORT")
print("=" * 60)

report_path = os.path.join(OUTPUT_DIR, "Data_Augmentation_Report.md")

report = f"""# Data Augmentation Report — Glaucoma Classification

**Generated**: Script `data_augment_pipeline.py`
**Seed**: {RANDOM_SEED}
**Target Train Ratio**: Normal:Glaucoma = {TARGET_RATIO}:1

---

## 1. Source Dataset

| Camera | Images in CSV | Matched on Disk | NORMAL | GLAUCOMA SUSPECT |
|--------|:---:|:---:|:---:|:---:|
"""
for cs in camera_stats:
    report += f"| **{cs['camera']}** | {cs['csv_count']} | {cs['disk_count']} | {cs['normal']} | {cs['glaucoma']} |\n"
report += f"| **Total** | — | **{len(data)}** | **{total_n}** | **{total_g}** |\n"

report += f"""
Original ratio: **{total_n/total_g:.1f}:1** (Normal:Glaucoma)

---

## 2. Train / Val / Test Split

| Parameter | Value |
|-----------|-------|
| Split Ratio | **70 / 15 / 15** |
| Stratification | Yes — by Majority Decision |
| Random Seed | {RANDOM_SEED} |

### Before Augmentation

| Split | NORMAL | GLAUCOMA SUSPECT | Total | Ratio |
|-------|:---:|:---:|:---:|:---:|
| Train | {split_info['train']['normal']} | {split_info['train']['glaucoma']} | {split_info['train']['total']} | {split_info['train']['ratio']:.1f}:1 |
| Val | {split_info['val']['normal']} | {split_info['val']['glaucoma']} | {split_info['val']['total']} | {split_info['val']['ratio']:.1f}:1 |
| Test | {split_info['test']['normal']} | {split_info['test']['glaucoma']} | {split_info['test']['total']} | {split_info['test']['ratio']:.1f}:1 |

### After Augmentation (train only)

| Split | NORMAL | GLAUCOMA SUSPECT | Total | Ratio | Notes |
|-------|:---:|:---:|:---:|:---:|-------|
| **Train** | {final_stats['train']['normal']} | **{final_stats['train']['glaucoma']}** ({split_info['train']['glaucoma']} orig + {aug_count} aug) | **{final_stats['train']['total']}** | **{final_stats['train']['ratio']:.1f}:1** ✅ | Augmented |
| Val | {final_stats['val']['normal']} | {final_stats['val']['glaucoma']} | {final_stats['val']['total']} | {final_stats['val']['ratio']:.1f}:1 | Untouched |
| Test | {final_stats['test']['normal']} | {final_stats['test']['glaucoma']} | {final_stats['test']['total']} | {final_stats['test']['ratio']:.1f}:1 | Untouched |

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
| **Augmented format** | JPEG, quality={JPEG_QUALITY} (near-lossless) |
| **Resolution** | Same as original — no resizing |
| **Augmented naming** | `aug_<index>_<original_name>.jpg` |

### Augmentation Distribution

- Total glaucoma images in train: **{split_info['train']['glaucoma']}**
- Each image received **{copies_per_image}** augmented copy(ies)
- **{remainder}** randomly selected images received 1 extra copy
- Total augmented images created: **{aug_count}**

---

## 4. Output Directory Structure

```
data_augment/
├── train/
│   ├── NORMAL/              ({final_stats['train']['normal']} originals)
│   └── GLAUCOMA_SUSPECT/    ({final_stats['train']['glaucoma']} = {split_info['train']['glaucoma']} orig + {aug_count} aug)
├── val/
│   ├── NORMAL/              ({final_stats['val']['normal']} originals)
│   └── GLAUCOMA_SUSPECT/    ({final_stats['val']['glaucoma']} originals)
├── test/
│   ├── NORMAL/              ({final_stats['test']['normal']} originals)
│   └── GLAUCOMA_SUSPECT/    ({final_stats['test']['glaucoma']} originals)
├── train_labels.csv         ({final_stats['train']['total']} rows)
├── val_labels.csv           ({final_stats['val']['total']} rows)
├── test_labels.csv          ({final_stats['test']['total']} rows)
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
4. **Augmented images use JPEG quality={JPEG_QUALITY}** — visually indistinguishable from originals
5. **Stratified splitting** ensures proportional representation of both classes across all splits
6. **Seed={RANDOM_SEED}** for full reproducibility of splits and augmentation selection
"""

with open(report_path, "w") as f:
    f.write(report)

print(f"  ✅ Report saved to: {report_path}")

print("\n" + "=" * 60)
print("ALL DONE!")
print("=" * 60)
print(f"\nOutput directory: {OUTPUT_DIR}")
