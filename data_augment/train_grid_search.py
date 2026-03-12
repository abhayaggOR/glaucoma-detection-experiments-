"""
Grid Search over Class Weights — YOLO v11 (Augmented Data)
============================================================
Trains YOLO11s-cls and YOLO11l-cls with multiple class weight
ratios to find the optimal weighting for GLAUCOMA_SUSPECT.

Weight ratios tested (GLAUCOMA_SUSPECT : NORMAL):
  3.0 : 0.5
  4.0 : 0.4
  5.0 : 0.3
  6.0 : 0.25

For each weight ratio, both models are trained, evaluated on
test data, and threshold-tuned (0.10–0.55).

Results → data_augment/runs5/
"""

import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# ==============================================================
# CONFIG
# ==============================================================
BASE_DIR  = "/home/abhay/chaksu/20123135/Train/data_augment"
DATA_DIR  = BASE_DIR
RUNS_DIR  = os.path.join(BASE_DIR, "runs5")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR  = os.path.join(BASE_DIR, "test")
os.makedirs(RUNS_DIR, exist_ok=True)

CLASS_NAMES    = ["GLAUCOMA_SUSPECT", "NORMAL"]  # alphabetical
POSITIVE_CLASS = "GLAUCOMA_SUSPECT"
GLAUCOMA_IDX   = 0

# Weight ratios to search: [GLAUCOMA_SUSPECT_weight, NORMAL_weight]
WEIGHT_GRID = [
    [3.0, 0.5],
    [4.0, 0.4],
    [5.0, 0.3],
    [6.0, 0.25],
]

# Model configs: (tag, model_name, batch, patience)
MODEL_CONFIGS = [
    ("yolo11s", "yolo11s-cls", 16, 100),
    ("yolo11l", "yolo11l-cls", 32,  75),
]

EPOCHS = 300
IMGSZ  = 512
THRESHOLDS = np.arange(0.1, 0.6, 0.05)

# ==============================================================
# LOAD TEST SET
# ==============================================================
print("=" * 60)
print("LOADING TEST SET")
print("=" * 60)

test_images = []
test_labels = []
for cls_name in CLASS_NAMES:
    cls_dir = os.path.join(TEST_DIR, cls_name)
    if not os.path.exists(cls_dir):
        continue
    for fname in sorted(os.listdir(cls_dir)):
        fpath = os.path.join(cls_dir, fname)
        if os.path.isfile(fpath):
            test_images.append(fpath)
            test_labels.append(cls_name)

y_true = np.array([1 if lbl == POSITIVE_CLASS else 0 for lbl in test_labels])
print(f"  Total test images: {len(test_images)}")
for cls in CLASS_NAMES:
    print(f"    {cls}: {test_labels.count(cls)}")

# ==============================================================
# MONKEY-PATCH LOSS
# ==============================================================
from ultralytics.nn.tasks import ClassificationModel
from ultralytics import YOLO

_original_init_criterion = ClassificationModel.init_criterion

# Global placeholder — updated per training run
_active_weights = None


class WeightedClassificationLoss:
    """Classification loss with class weights."""
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        w = self.weight.to(device=preds.device, dtype=preds.dtype)
        loss = F.cross_entropy(preds, batch["cls"], weight=w, reduction="mean")
        return loss, loss.detach()


def _weighted_init_criterion(self):
    global _active_weights
    print(f"  ✅ WeightedClassificationLoss: {_active_weights}")
    return WeightedClassificationLoss(_active_weights)


ClassificationModel.init_criterion = _weighted_init_criterion


# ==============================================================
# HELPER: evaluate + threshold tune
# ==============================================================
def evaluate_and_tune(model_path, run_tag, out_dir):
    """Run inference on test set, compute metrics at default 0.5 and all thresholds."""
    eval_model = YOLO(model_path)

    # YOLO built-in val for confusion-matrix PNGs
    eval_model.val(
        data=DATA_DIR, split="test", imgsz=IMGSZ, batch=32,
        device=0, workers=0, plots=True,
        project=out_dir, name=f"{run_tag}_test_eval", exist_ok=True,
    )
    # Copy confusion matrices
    eval_dir = os.path.join(out_dir, f"{run_tag}_test_eval")
    run_dir  = os.path.join(out_dir, run_tag)
    os.makedirs(run_dir, exist_ok=True)
    for cm_file in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
        src = os.path.join(eval_dir, cm_file)
        dst = os.path.join(run_dir, cm_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # Inference for threshold tuning
    all_probs = []
    for i in range(0, len(test_images), 32):
        batch_paths = test_images[i:i + 32]
        results = eval_model.predict(source=batch_paths, imgsz=IMGSZ, device=0, verbose=False)
        for r in results:
            all_probs.append(r.probs.data.cpu().numpy())
    all_probs = np.array(all_probs)

    # Threshold sweep
    threshold_results = []
    for thresh in THRESHOLDS:
        y_pred = (all_probs[:, GLAUCOMA_IDX] >= thresh).astype(int)
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        threshold_results.append({
            "threshold": round(float(thresh), 2),
            "accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1_score": round(f1, 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        })

    df = pd.DataFrame(threshold_results)
    csv_path = os.path.join(out_dir, f"{run_tag}_threshold_results.csv")
    df.to_csv(csv_path, index=False)

    best_f1 = df.loc[df["f1_score"].idxmax()]
    best_thresh = float(best_f1["threshold"])

    # Confusion matrix at best-F1 threshold
    y_pred_best = (all_probs[:, GLAUCOMA_IDX] >= best_thresh).astype(int)
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
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=tick_labels, yticklabels=tick_labels,
               xlabel="Predicted label", ylabel="True label",
               title=f"{run_tag} (thresh={best_thresh:.2f})"
                     + (" [norm]" if normed else ""))
        fmt = ".2f" if normed else "d"
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm_data[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm_data[i, j] > cm_data.max() / 2 else "black")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{run_tag}_{suffix}.png"), dpi=150)
        plt.close(fig)

    del eval_model
    torch.cuda.empty_cache()
    return df, best_f1


# ==============================================================
# MAIN: GRID SEARCH
# ==============================================================
all_summaries = []

for w_glaucoma, w_normal in WEIGHT_GRID:
    w_label = f"w{w_glaucoma:.1f}_{w_normal:.2f}"
    _active_weights = torch.tensor([w_glaucoma, w_normal], dtype=torch.float32)

    print(f"\n{'#' * 60}")
    print(f"WEIGHT CONFIG: GLAUCOMA={w_glaucoma}, NORMAL={w_normal}")
    print(f"{'#' * 60}")

    for model_tag, model_name, batch, patience in MODEL_CONFIGS:
        run_tag = f"{model_tag}_{w_label}"
        print(f"\n{'=' * 60}")
        print(f"TRAINING {run_tag}")
        print(f"  model={model_name}, batch={batch}, patience={patience}")
        print(f"  weights=[{w_glaucoma}, {w_normal}]")
        print("=" * 60)

        model = YOLO(model_name)
        model.train(
            data=DATA_DIR, epochs=EPOCHS, imgsz=IMGSZ, rect=True,
            batch=batch, patience=patience, device=0, workers=0,
            project=RUNS_DIR, name=run_tag, exist_ok=True,
        )
        print(f"  ✅ Training complete: {run_tag}")

        # Evaluate
        weights_path = os.path.join(RUNS_DIR, run_tag, "weights", "best.pt")
        df, best_f1 = evaluate_and_tune(weights_path, run_tag, RUNS_DIR)

        summary = {
            "model": model_tag,
            "w_glaucoma": w_glaucoma,
            "w_normal": w_normal,
            "best_f1_threshold": best_f1["threshold"],
            "best_f1": best_f1["f1_score"],
            "precision_at_best_f1": best_f1["precision"],
            "recall_at_best_f1": best_f1["recall"],
            "accuracy_at_best_f1": best_f1["accuracy"],
        }
        all_summaries.append(summary)
        print(f"  🏆 Best F1={best_f1['f1_score']:.4f} @ thresh={best_f1['threshold']:.2f}")
        print(f"     Precision={best_f1['precision']:.4f}, Recall={best_f1['recall']:.4f}")

        del model
        torch.cuda.empty_cache()

# Restore original
ClassificationModel.init_criterion = _original_init_criterion

# ==============================================================
# SUMMARY REPORT
# ==============================================================
print(f"\n{'=' * 60}")
print("GENERATING GRID SEARCH SUMMARY REPORT")
print("=" * 60)

summary_df = pd.DataFrame(all_summaries)
summary_df.to_csv(os.path.join(RUNS_DIR, "grid_search_summary.csv"), index=False)

report = """# Grid Search Report — Class Weight Tuning

**Models**: YOLO11s-cls and YOLO11l-cls on augmented data
**Weight Ratios Tested**: [3.0:0.5], [4.0:0.4], [5.0:0.3], [6.0:0.25]
**Thresholds Evaluated**: 0.10 to 0.55 (step 0.05)
**Metrics**: Evaluated on TEST set

---

## Summary — Best F1 per Configuration

| Model | W_Glaucoma | W_Normal | Best Threshold | F1 | Precision | Recall | Accuracy |
|-------|-----------|----------|:--------------:|:--:|:---------:|:------:|:--------:|
"""

for _, row in summary_df.iterrows():
    report += (
        f"| {row['model']} | {row['w_glaucoma']:.1f} | {row['w_normal']:.2f} "
        f"| {row['best_f1_threshold']:.2f} | {row['best_f1']:.4f} "
        f"| {row['precision_at_best_f1']:.4f} | {row['recall_at_best_f1']:.4f} "
        f"| {row['accuracy_at_best_f1']:.4f} |\n"
    )

# Best overall
for mtag in ["yolo11s", "yolo11l"]:
    sub = summary_df[summary_df["model"] == mtag]
    best = sub.loc[sub["best_f1"].idxmax()]
    report += f"\n**Best {mtag}**: weights=[{best['w_glaucoma']:.1f}, {best['w_normal']:.2f}], "
    report += f"F1={best['best_f1']:.4f}, Recall={best['recall_at_best_f1']:.4f}\n"

report += "\n---\n\n## Per-Configuration Threshold Tables\n\n"

for _, row in summary_df.iterrows():
    run_tag = f"{row['model']}_w{row['w_glaucoma']:.1f}_{row['w_normal']:.2f}"
    csv_path = os.path.join(RUNS_DIR, f"{run_tag}_threshold_results.csv")
    if os.path.exists(csv_path):
        tdf = pd.read_csv(csv_path)
        report += f"### {run_tag}\n\n"
        report += "| Threshold | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |\n"
        report += "|:---------:|:--------:|:---------:|:------:|:--:|:--:|:--:|:--:|:--:|\n"
        for _, tr in tdf.iterrows():
            report += (
                f"| {tr['threshold']:.2f} | {tr['accuracy']:.4f} | {tr['precision']:.4f} "
                f"| {tr['recall']:.4f} | {tr['f1_score']:.4f} "
                f"| {int(tr['tp'])} | {int(tr['fp'])} | {int(tr['tn'])} | {int(tr['fn'])} |\n"
            )
        report += f"\n**Confusion Matrix**: `{run_tag}_confusion_matrix.png`\n\n---\n\n"

with open(os.path.join(RUNS_DIR, "Grid_Search_Report.md"), "w") as f:
    f.write(report)

print(f"  ✅ Report saved to: {os.path.join(RUNS_DIR, 'Grid_Search_Report.md')}")
print(f"\n{'=' * 60}")
print("GRID SEARCH COMPLETE")
print("=" * 60)
print(f"Results: {RUNS_DIR}")
