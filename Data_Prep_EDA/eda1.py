import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.inter_rater import fleiss_kappa

file_path = "/home/abhay/chaksu/20123135/Train/6.0_Glaucoma_Decision/Glaucoma_Decision_Comparison_Remidio_majority.csv"

df = pd.read_csv(file_path)

df.head()
print("\nMissing values:")
print(df.isnull().sum())
majority_counts = df["Majority Decision"].value_counts()
print(majority_counts)
majority_percent = df["Majority Decision"].value_counts(normalize=True) * 100
print(majority_percent.round(2))
import matplotlib.pyplot as plt

plt.figure(figsize=(5,4))
majority_counts.plot(kind="bar")
plt.title("Majority Glaucoma Decision Distribution")
plt.ylabel("Number of Images")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
expert_cols = ["Expert.1", "Expert.2", "Expert.3", "Expert.4", "Expert.5"]

df["glaucoma_votes"] = (df[expert_cols] == "GLAUCOMA_SUSPECT").sum(axis=1)

df["glaucoma_votes"].value_counts().sort_index()
## aggreement vs disagree
df["unique_opinions"] = df[expert_cols].nunique(axis=1)

df["unique_opinions"].value_counts().sort_index()
## borderline cases 
borderline = df[df["glaucoma_votes"].isin([2, 3])]

print("Borderline cases:", borderline.shape[0])


## expert bias analysis

expert_bias = (df[expert_cols] == "GLAUCOMA_SUSPECT").mean()

print("Expert glaucoma rates:")
print((expert_bias * 100).round(2))
## verify if majority => 3 ou of 5 experts 
df["derived_majority"] = df["glaucoma_votes"].apply(
    lambda x: "GLAUCOMA_SUSPECT" if x >= 3 else "NORMAL"
)

## class distribution majority_counts = df["Majority Decision"].value_counts()
print(majority_counts)

majority_percent = df["Majority Decision"].value_counts(normalize=True) * 100
print(majority_percent.round(2))

###confidence intervals 
expert_cols = ["Expert.1", "Expert.2", "Expert.3", "Expert.4", "Expert.5"]
# Number of glaucoma votes (ignores NaNs automatically)
df["glaucoma_votes"] = (df[expert_cols] == "GLAUCOMA_SUSPECT").sum(axis=1)

# Total number of available expert votes per image
df["total_votes"] = df[expert_cols].notna().sum(axis=1)

def confidence_level(row):
    gv = row["glaucoma_votes"]
    tv = row["total_votes"]

    if tv == 0:
        return "No expert labels"

    if gv == 0:
        return "High confidence NORMAL"
    elif gv == tv:
        return "High confidence GLAUCOMA"
    elif gv == 1:
        return "Low confidence NORMAL"
    elif gv == tv - 1:
        return "Low confidence GLAUCOMA"
    else:
        return "Ambiguous"
df["confidence_bucket"] = df.apply(confidence_level, axis=1)
print(df["confidence_bucket"].value_counts())
