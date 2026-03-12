import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# PATHS (EDIT ONLY IF NEEDED)
# -----------------------------
IMAGE_DIR = "/home/abhay/chaksu/20123135/Train/1.0_Original_Fundus_Images/Remidio"
CSV_PATH = "/home/abhay/chaksu/20123135/Train/6.0_Glaucoma_Decision/Glaucoma_Decision_Comparison_Remidio_majority.csv"
OUTPUT_DIR = "/home/abhay/yolo_cls_data"

# -----------------------------
# LOAD CSV
# -----------------------------
df = pd.read_csv(CSV_PATH)

print("CSV loaded")
print("Total rows in CSV:", len(df))

# -----------------------------
# CLEAN LABELS
# -----------------------------
def clean_label(x):
    x = str(x).strip().upper()
    if "GLAUCOMA" in x:
        return "GLAUCOMA_SUSPECT"
    return "NORMAL"

df["Majority Decision"] = df["Majority Decision"].apply(clean_label)

# -----------------------------
# RESOLVE IMAGE PATHS
# -----------------------------
def resolve_image_path(csv_name):
    """
    CSV example: 17521.tif-17521-1.tif
    Actual file: 17521-1.tif
    """
    real_name = csv_name.split("-")[-1]
    return os.path.join(IMAGE_DIR, real_name)

df["img_path"] = df["Images"].apply(resolve_image_path)

# -----------------------------
# CHECK IMAGE EXISTENCE
# -----------------------------
missing = df[~df["img_path"].apply(os.path.exists)]

print("Missing images:", len(missing))

if len(missing) > 0:
    print("Examples of missing images:")
    print(missing["Images"].head())

# Keep only valid images
df = df[df["img_path"].apply(os.path.exists)].reset_index(drop=True)

print("Usable images after filtering:", len(df))

# HARD STOP if empty
if len(df) == 0:
    raise RuntimeError("❌ No images found. Check IMAGE_DIR and filename mapping.")

# -----------------------------
# TRAIN / VAL SPLIT
# -----------------------------
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["Majority Decision"],
    random_state=42
)

print("Train size:", len(train_df))
print("Val size:", len(val_df))

# -----------------------------
# COPY FILES INTO YOLO FORMAT
# -----------------------------
def copy_images(df_split, split_name):
    for _, row in df_split.iterrows():
        label = row["Majority Decision"]
        dst_dir = os.path.join(OUTPUT_DIR, split_name, label)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(row["img_path"], dst_dir)

copy_images(train_df, "train")
copy_images(val_df, "val")

print("\n✅ YOLOv8 classification dataset created successfully!")
print("Dataset path:", OUTPUT_DIR)

