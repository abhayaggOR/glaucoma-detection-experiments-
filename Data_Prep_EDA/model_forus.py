import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# PATHS
# -------------------------------------------------
IMAGE_DIR = "/home/abhay/chaksu/20123135/Train/1.0_Original_Fundus_Images/Forus"
CSV_PATH = "/home/abhay/chaksu/20123135/Train/6.0_Glaucoma_Decision/Glaucoma_Decision_Comparison_Forus_majority.csv"
OUTPUT_DIR = "/home/abhay/yolo_cls_forus"

# -------------------------------------------------
# LOAD CSV
# -------------------------------------------------
df = pd.read_csv(CSV_PATH)
print("CSV loaded | Rows:", len(df))

# -------------------------------------------------
# CLEAN LABELS
# -------------------------------------------------
def clean_label(x):
    x = str(x).strip().upper()
    if "GLAUCOMA" in x:
        return "GLAUCOMA_SUSPECT"
    return "NORMAL"

df["Majority Decision"] = df["Majority Decision"].apply(clean_label)

# -------------------------------------------------
# EXTRACT NUMERIC IMAGE ID
# -------------------------------------------------
def extract_numeric_id(csv_name):
    """
    Example:
    95.jpg-95-1.jpg -> 95
    1.jpg-1-1.jpg   -> 1
    """
    first_part = str(csv_name).split("-")[0]   # "95.jpg"
    numeric_id = os.path.splitext(first_part)[0]  # "95"
    return numeric_id

df["image_id"] = df["Images"].apply(extract_numeric_id)

# -------------------------------------------------
# BUILD IMAGE PATHS (.png)
# -------------------------------------------------
df["img_path"] = df["image_id"].apply(
    lambda x: os.path.join(IMAGE_DIR, f"{x}.png")
)

# -------------------------------------------------
# VERIFY IMAGE EXISTENCE
# -------------------------------------------------
missing = df[~df["img_path"].apply(os.path.exists)]

print("Images matched:", len(df) - len(missing))
print("Images missing:", len(missing))

if len(missing) > 0:
    print("Examples of missing mappings:")
    print(missing[["Images", "img_path"]].head())

# Keep only matched images
df = df[df["img_path"].apply(os.path.exists)].reset_index(drop=True)

# HARD STOP if nothing matched
if len(df) == 0:
    raise RuntimeError("❌ No images matched. Check numeric ID extraction or file extension.")

# -------------------------------------------------
# TRAIN / VAL SPLIT (STRATIFIED)
# -------------------------------------------------
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["Majority Decision"],
    random_state=42
)

print("Train images:", len(train_df))
print("Val images:", len(val_df))

# -------------------------------------------------
# COPY FILES INTO YOLO FORMAT
# -------------------------------------------------
def copy_images(df_split, split_name):
    for _, row in df_split.iterrows():
        label = row["Majority Decision"]
        dst_dir = os.path.join(OUTPUT_DIR, split_name, label)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(row["img_path"], dst_dir)

copy_images(train_df, "train")
copy_images(val_df, "val")

print("\n✅ YOLOv8 Forus dataset prepared successfully!")
print("Dataset location:", OUTPUT_DIR)

