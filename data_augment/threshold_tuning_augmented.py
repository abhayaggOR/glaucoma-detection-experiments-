"""
Threshold Tuning — Augmented Data YOLO v11 Models
===================================================
Evaluates YOLO11s-cls and YOLO11l-cls (trained on augmented data)
across a range of confidence thresholds on the TEST set.

Thresholds: 0.10, 0.15, 0.20, ..., 0.55
Results saved under data_augment/runs2/
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from PIL import Image

# ==============================================================
# CONFIG
# ==============================================================
BASE_DIR = "/home/abhay/chaksu/20123135/Train/data_augment"
TEST_DIR = os.path.join(BASE_DIR, "test")
RUNS2_DIR = os.path.join(BASE_DIR, "runs2")
os.makedirs(RUNS2_DIR, exist_ok=True)

# Trained model weights
MODELS = {
    "yolo11s_augmented": os.path.join(BASE_DIR, "runs", "yolo11s_augmented", "weights", "best.pt"),
    "yolo11l_augmented": os.path.join(BASE_DIR, "runs", "yolo11l_augmented", "weights", "best.pt"),
}

# Thresholds to evaluate
thresholds = np.arange(0.1, 0.6, 0.05)

# Class mapping — YOLO classification uses folder names alphabetically
# GLAUCOMA_SUSPECT = class 0, NORMAL = class 1
CLASS_NAMES = ["GLAUCOMA_SUSPECT", "NORMAL"]
POSITIVE_CLASS = "GLAUCOMA_SUSPECT"  # what we care about detecting


# ==============================================================
# STEP 1: LOAD TEST SET GROUND TRUTH
# ==============================================================
print("=" * 60)
print("STEP 1: LOADING TEST SET")
print("=" * 60)

test_images = []
test_labels = []

for cls_name in CLASS_NAMES:
    cls_dir = os.path.join(TEST_DIR, cls_name)
    if not os.path.exists(cls_dir):
        print(f"  WARNING: {cls_dir} not found, skipping")
        continue
    for fname in sorted(os.listdir(cls_dir)):
        fpath = os.path.join(cls_dir, fname)
        if os.path.isfile(fpath):
            test_images.append(fpath)
            test_labels.append(cls_name)

print(f"  Total test images: {len(test_images)}")
for cls in CLASS_NAMES:
    count = test_labels.count(cls)
    print(f"    {cls}: {count}")


# ==============================================================
# STEP 2: RUN THRESHOLD TUNING FOR EACH MODEL
# ==============================================================
for model_name, model_path in MODELS.items():
    print(f"\n{'=' * 60}")
    print(f"MODEL: {model_name}")
    print(f"  Weights: {model_path}")
    print("=" * 60)

    # Load model
    model = YOLO(model_path)

    # Get predictions (softmax probabilities) for all test images
    print("\n  Running inference on test set...")
    all_probs = []  # list of prob arrays [prob_class0, prob_class1]

    # Process in batches for efficiency
    batch_size = 32
    for i in range(0, len(test_images), batch_size):
        batch_paths = test_images[i:i + batch_size]
        results = model.predict(
            source=batch_paths,
            imgsz=512,
            device=0,
            verbose=False,
        )
        for r in results:
            probs = r.probs.data.cpu().numpy()
            all_probs.append(probs)

    all_probs = np.array(all_probs)
    print(f"  Inference complete. Shape: {all_probs.shape}")

    # Get class index mapping
    # class 0 = GLAUCOMA_SUSPECT, class 1 = NORMAL (alphabetical)
    glaucoma_idx = 0
    normal_idx = 1

    # Convert ground truth to binary: 1 = GLAUCOMA_SUSPECT (positive), 0 = NORMAL
    y_true = np.array([1 if lbl == POSITIVE_CLASS else 0 for lbl in test_labels])

    # Evaluate at each threshold
    threshold_results = []

    print(f"\n  {'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    print("  " + "-" * 75)

    for thresh in thresholds:
        # Predict GLAUCOMA_SUSPECT if its probability >= threshold
        y_pred = (all_probs[:, glaucoma_idx] >= thresh).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        threshold_results.append({
            "threshold": round(float(thresh), 2),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "total_predicted_positive": int(tp + fp),
            "total_actual_positive": int(tp + fn),
        })

        print(f"  {thresh:>10.2f} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {tp:>5d} {fp:>5d} {tn:>5d} {fn:>5d}")

    # Save results CSV
    results_df = pd.DataFrame(threshold_results)
    csv_path = os.path.join(RUNS2_DIR, f"{model_name}_threshold_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n  ✅ Results saved to: {csv_path}")

    # Find best threshold by F1
    best_f1_row = results_df.loc[results_df["f1_score"].idxmax()]
    print(f"\n  🏆 Best F1 threshold: {best_f1_row['threshold']:.2f}")
    print(f"     F1={best_f1_row['f1_score']:.4f}, Precision={best_f1_row['precision']:.4f}, Recall={best_f1_row['recall']:.4f}")

    # Find best threshold by Recall (for medical — prioritize catching glaucoma)
    best_rec_row = results_df.loc[results_df["recall"].idxmax()]
    # If multiple thresholds give same recall, pick the one with best precision
    max_recall = results_df["recall"].max()
    best_recall_rows = results_df[results_df["recall"] == max_recall]
    best_rec_row = best_recall_rows.loc[best_recall_rows["precision"].idxmax()]
    print(f"\n  🏥 Best Recall threshold: {best_rec_row['threshold']:.2f}")
    print(f"     Recall={best_rec_row['recall']:.4f}, Precision={best_rec_row['precision']:.4f}, F1={best_rec_row['f1_score']:.4f}")

    # ------------------------------------------------------------------
    # Generate confusion matrix PNGs for the best-F1 threshold (test set)
    # ------------------------------------------------------------------
    best_thresh = float(best_f1_row["threshold"])
    y_pred_best = (all_probs[:, glaucoma_idx] >= best_thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred_best, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    for normed, cm_data, suffix in [
        (False, cm, "confusion_matrix"),
        (True, cm_norm, "confusion_matrix_normalized"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_data, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        tick_labels = ["NORMAL", "GLAUCOMA_SUSPECT"]
        ax.set(
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=tick_labels, yticklabels=tick_labels,
            xlabel="Predicted label", ylabel="True label",
            title=f"{model_name} — Test Set (thresh={best_thresh:.2f})"
               + (" [normalized]" if normed else ""),
        )
        fmt = ".2f" if normed else "d"
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm_data[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm_data[i, j] > cm_data.max() / 2 else "black")
        fig.tight_layout()
        png_path = os.path.join(RUNS2_DIR, f"{model_name}_{suffix}.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"    ✅ {suffix}.png saved → {png_path}")

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()


# ==============================================================
# STEP 3: GENERATE SUMMARY REPORT
# ==============================================================
print(f"\n{'=' * 60}")
print("STEP 3: GENERATING SUMMARY REPORT")
print("=" * 60)

report = """# Threshold Tuning Report — Augmented Data Models

**Models**: YOLO11s-cls (small) and YOLO11l-cls (large) trained on augmented data
**Test Set**: Untouched test split from data_augment pipeline
**Thresholds Evaluated**: 0.10 to 0.55 (step 0.05)
**Positive Class**: GLAUCOMA_SUSPECT
**Confusion Matrices**: Generated on TEST data at the best-F1 threshold for each model

---

"""

for model_name in MODELS:
    csv_path = os.path.join(RUNS2_DIR, f"{model_name}_threshold_results.csv")
    df = pd.read_csv(csv_path)

    report += f"## {model_name}\n\n"
    report += "| Threshold | Accuracy | Precision | Recall | F1 Score | TP | FP | TN | FN |\n"
    report += "|:---------:|:--------:|:---------:|:------:|:--------:|:--:|:--:|:--:|:--:|\n"

    for _, row in df.iterrows():
        report += f"| {row['threshold']:.2f} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1_score']:.4f} | {int(row['tp'])} | {int(row['fp'])} | {int(row['tn'])} | {int(row['fn'])} |\n"

    # Best F1
    best_f1 = df.loc[df["f1_score"].idxmax()]
    report += f"\n**Best F1**: threshold={best_f1['threshold']:.2f}, F1={best_f1['f1_score']:.4f}, Precision={best_f1['precision']:.4f}, Recall={best_f1['recall']:.4f}\n"

    # Best Recall
    max_recall = df["recall"].max()
    best_recall_rows = df[df["recall"] == max_recall]
    best_rec = best_recall_rows.loc[best_recall_rows["precision"].idxmax()]
    report += f"\n**Best Recall**: threshold={best_rec['threshold']:.2f}, Recall={best_rec['recall']:.4f}, Precision={best_rec['precision']:.4f}, F1={best_rec['f1_score']:.4f}\n"

    # Confusion matrix reference
    report += f"\n**Confusion Matrix (test set)**: `{model_name}_confusion_matrix.png`\n"
    report += f"**Normalized Confusion Matrix (test set)**: `{model_name}_confusion_matrix_normalized.png`\n"

    report += "\n---\n\n"

report_path = os.path.join(RUNS2_DIR, "Threshold_Tuning_Report.md")
with open(report_path, "w") as f:
    f.write(report)

print(f"  ✅ Report saved to: {report_path}")

print(f"\n{'=' * 60}")
print("ALL THRESHOLD TUNING COMPLETE")
print("=" * 60)
print(f"\nResults directory: {RUNS2_DIR}")
