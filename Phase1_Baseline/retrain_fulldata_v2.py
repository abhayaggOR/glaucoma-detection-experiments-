import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DATASETS = [
    {
        "csv": "/home/abhay/chaksu/20123135/Train/6.0_Glaucoma_Decision/bosch_data.csv",
        "img_dir": "/home/abhay/chaksu/20123135/Train/1.0_Original_Fundus_Images/Bosch"
    },
    {
        "csv": "/home/abhay/chaksu/20123135/Train/6.0_Glaucoma_Decision/forus_data.csv",
        "img_dir": "/home/abhay/chaksu/20123135/Train/1.0_Original_Fundus_Images/Forus"
    },
    {
        "csv": "/home/abhay/chaksu/20123135/Train/6.0_Glaucoma_Decision/remidio_images_updated.csv",
        "img_dir": "/home/abhay/chaksu/20123135/Train/1.0_Original_Fundus_Images/Remidio"
    }
]

OUTPUT_DIR = "/home/abhay/yolo_glaucoma_cls"
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# --------------------------------------------------
# STEP 1: DATA PREPARATION
# --------------------------------------------------
print("=" * 60)
print("STEP 1: DATA PREPARATION")
print("=" * 60)

# Clean previous output
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print(f"Cleaned previous output at {OUTPUT_DIR}")

dfs = []
for d in DATASETS:
    df = pd.read_csv(d["csv"])
    df["image_path"] = df["Images"].apply(
        lambda x: os.path.join(d["img_dir"], x)
    )
    df["camera"] = os.path.basename(d["img_dir"])
    before = len(df)
    df = df[df["image_path"].apply(os.path.exists)]
    after = len(df)
    print(f"  {os.path.basename(d['img_dir'])}: {before} in CSV, {after} matched on disk")
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"\nTotal usable images: {len(data)}")
print(f"  NORMAL: {(data['Majority Decision'] == 'NORMAL').sum()}")
print(f"  GLAUCOMA SUSPECT: {(data['Majority Decision'] == 'GLAUCOMA SUSPECT').sum()}")

# Train / Test split
train_df, test_df = train_test_split(
    data,
    test_size=1 - TRAIN_RATIO,
    stratify=data["Majority Decision"],
    random_state=RANDOM_SEED
)
print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}")

# Create folder structure
for split, df in [("train", train_df), ("test", test_df)]:
    for label in df["Majority Decision"].unique():
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, label.replace(" ", "_")),
            exist_ok=True
        )

# Copy images
def copy_images(df, split):
    for _, row in df.iterrows():
        label = row["Majority Decision"].replace(" ", "_")
        dst = os.path.join(OUTPUT_DIR, split, label, os.path.basename(row["image_path"]))
        shutil.copy(row["image_path"], dst)

copy_images(train_df, "train")
copy_images(test_df, "test")
print(f"\n✅ YOLO classification dataset ready at: {OUTPUT_DIR}")

# --------------------------------------------------
# STEP 2: TRAIN YOLO11s-cls (epochs=300, batch=16)
# --------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: TRAINING YOLO11s-cls (epochs=300, batch=16)")
print("=" * 60)

from ultralytics import YOLO

model_s = YOLO("yolo11s-cls")
results_s = model_s.train(
    data=OUTPUT_DIR,
    epochs=300,
    imgsz=512,
    rect=True,
    batch=16,
    device=0,
    workers=0,
    project="/home/abhay/chaksu/20123135/Train/6.0_Glaucoma_Decision/runs2",
    name="train_full_s",
    exist_ok=True
)

print("\n✅ YOLO11s-cls training complete")

# --------------------------------------------------
# STEP 3: TRAIN YOLO11l-cls (epochs=300, batch=16)
# --------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: TRAINING YOLO11l-cls (epochs=300, batch=16)")
print("=" * 60)

model_l = YOLO("yolo11l-cls")
results_l = model_l.train(
    data=OUTPUT_DIR,
    epochs=300,
    imgsz=512,
    rect=True,
    batch=16,
    device=0,
    workers=0,
    project="/home/abhay/chaksu/20123135/Train/6.0_Glaucoma_Decision/runs2",
    name="train_full_l",
    exist_ok=True
)

print("\n✅ YOLO11l-cls training complete")
print("\n" + "=" * 60)
print("ALL TRAINING COMPLETE")
print("=" * 60)
