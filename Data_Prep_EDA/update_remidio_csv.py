import pandas as pd
import os

# --------------------------------------------------
# PATHS
# --------------------------------------------------
input_csv = "/home/abhay/chaksu/20123135/Train/6.0_Glaucoma_Decision/Glaucoma_Decision_Comparison_Remidio_majority.csv"

output_csv = "/home/abhay/chaksu/20123135/Train/6.0_Glaucoma_Decision/Glaucoma_Decision_Comparison_Remidio_majority_with_jpg_names.csv"

# --------------------------------------------------
# LOAD CSV
# --------------------------------------------------
df = pd.read_csv(input_csv)

print("CSV loaded")
print("Original columns:", df.columns.tolist())

# --------------------------------------------------
# CREATE NEW IMAGE NAME COLUMN
# --------------------------------------------------
def convert_to_jpg_name(image_value):
    """
    Examples:
    17521.tif-17521-1.tif        -> 17521.jpg
    IMG_2431.tif-IMG_2431-1.tif  -> IMG_2431.jpg
    """
    first_part = str(image_value).split("-")[0]     # e.g. 17521.tif or IMG_2431.tif
    base_name = os.path.splitext(first_part)[0]     # remove .tif
    return base_name + ".jpg"

df["image_name_jpg"] = df["Images"].apply(convert_to_jpg_name)

# --------------------------------------------------
# SANITY CHECK
# --------------------------------------------------
print("\nSample check:")
print(df[["Images", "image_name_jpg"]].head(10))

# --------------------------------------------------
# SAVE NEW CSV (ORIGINAL UNCHANGED)
# --------------------------------------------------
df.to_csv(output_csv, index=False)

print("\n✅ New CSV saved at:")
print(output_csv)
