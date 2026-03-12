"""
Focal Loss Training — YOLO v11 (Augmented Data)
=================================================
Trains YOLO11s-cls and YOLO11l-cls using Focal Loss to handle
class imbalance. Focal loss down-weights easy examples and
focuses learning on hard misclassified samples.

Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

We test two gamma values: 1.0 (mild) and 2.0 (standard).
Alpha is set based on inverse class frequency.

Small model:  batch=16, epochs=300, patience=100, imgsz=512
Large model:  batch=32, epochs=300, patience=75,  imgsz=512

Results → data_augment/runs6/
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
RUNS_DIR  = os.path.join(BASE_DIR, "runs6")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR  = os.path.join(BASE_DIR, "test")
os.makedirs(RUNS_DIR, exist_ok=True)

CLASS_NAMES    = ["GLAUCOMA_SUSPECT", "NORMAL"]
POSITIVE_CLASS = "GLAUCOMA_SUSPECT"
GLAUCOMA_IDX   = 0

# Focal loss configs: (gamma, alpha_glaucoma, alpha_normal, label)
#   alpha values: higher for minority class (GLAUCOMA_SUSPECT)
FOCAL_CONFIGS = [
    (1.0, 0.75, 0.25, "gamma1.0_alpha0.75"),
    (2.0, 0.75, 0.25, "gamma2.0_alpha0.75"),
    (2.0, 0.80, 0.20, "gamma2.0_alpha0.80"),
    (3.0, 0.75, 0.25, "gamma3.0_alpha0.75"),
]

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
# FOCAL LOSS IMPLEMENTATION
# ==============================================================
class FocalClassificationLoss:
    """
    Focal Loss for classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: per-class weights tensor [alpha_class0, alpha_class1]
        gamma: focusing parameter (0 = standard CE, 2 = standard focal)
    """
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
        print(f"  ✅ FocalClassificationLoss: gamma={gamma}, alpha={alpha}")

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        targets = batch["cls"]

        # Compute softmax probabilities
        log_probs = F.log_softmax(preds.float(), dim=1)
        probs = log_probs.exp()

        # Gather the probabilities for the true classes
        targets_long = targets.long()
        p_t = probs.gather(1, targets_long.unsqueeze(1)).squeeze(1)

        # Per-sample alpha based on class
        alpha = self.alpha.to(device=preds.device, dtype=preds.dtype)
        alpha_t = alpha.gather(0, targets_long)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Loss: -alpha_t * focal_weight * log(p_t)
        ce_loss = F.nll_loss(log_probs, targets_long, reduction="none")
        loss = (alpha_t * focal_weight * ce_loss).mean()

        return loss, loss.detach()


# ==============================================================
# MONKEY-PATCH
# ==============================================================
from ultralytics.nn.tasks import ClassificationModel
from ultralytics import YOLO

_original_init_criterion = ClassificationModel.init_criterion
_active_focal_config = None  # (alpha_tensor, gamma)


def _focal_init_criterion(self):
    global _active_focal_config
    alpha, gamma = _active_focal_config
    return FocalClassificationLoss(alpha, gamma)


ClassificationModel.init_criterion = _focal_init_criterion


# ==============================================================
# HELPER: evaluate + threshold tune
# ==============================================================
def evaluate_and_tune(model_path, run_tag, out_dir):
    eval_model = YOLO(model_path)

    # YOLO built-in val for confusion-matrix PNGs
    eval_model.val(
        data=DATA_DIR, split="test", imgsz=IMGSZ, batch=32,
        device=0, workers=0, plots=True,
        project=out_dir, name=f"{run_tag}_test_eval", exist_ok=True,
    )
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
# MAIN: FOCAL LOSS TRAINING
# ==============================================================
all_summaries = []

for gamma, alpha_g, alpha_n, config_label in FOCAL_CONFIGS:
    alpha_tensor = torch.tensor([alpha_g, alpha_n], dtype=torch.float32)
    _active_focal_config = (alpha_tensor, gamma)

    print(f"\n{'#' * 60}")
    print(f"FOCAL LOSS CONFIG: {config_label}")
    print(f"  gamma={gamma}, alpha=[{alpha_g}, {alpha_n}]")
    print(f"{'#' * 60}")

    for model_tag, model_name, batch, patience in MODEL_CONFIGS:
        run_tag = f"{model_tag}_{config_label}"
        print(f"\n{'=' * 60}")
        print(f"TRAINING {run_tag}")
        print(f"  model={model_name}, batch={batch}, patience={patience}")
        print("=" * 60)

        model = YOLO(model_name)
        model.train(
            data=DATA_DIR, epochs=EPOCHS, imgsz=IMGSZ, rect=True,
            batch=batch, patience=patience, device=0, workers=0,
            project=RUNS_DIR, name=run_tag, exist_ok=True,
        )
        print(f"  ✅ Training complete: {run_tag}")

        weights_path = os.path.join(RUNS_DIR, run_tag, "weights", "best.pt")
        df, best_f1 = evaluate_and_tune(weights_path, run_tag, RUNS_DIR)

        summary = {
            "model": model_tag,
            "gamma": gamma,
            "alpha_glaucoma": alpha_g,
            "alpha_normal": alpha_n,
            "config": config_label,
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
print("GENERATING FOCAL LOSS SUMMARY REPORT")
print("=" * 60)

summary_df = pd.DataFrame(all_summaries)
summary_df.to_csv(os.path.join(RUNS_DIR, "focal_loss_summary.csv"), index=False)

report = """# Focal Loss Report — YOLO v11

**Models**: YOLO11s-cls and YOLO11l-cls on augmented data
**Loss**: Focal Loss  FL(p_t) = -α_t · (1-p_t)^γ · log(p_t)
**Configs Tested**:
- γ=1.0, α=[0.75, 0.25] (mild focusing)
- γ=2.0, α=[0.75, 0.25] (standard focal)
- γ=2.0, α=[0.80, 0.20] (stronger minority bias)
- γ=3.0, α=[0.75, 0.25] (aggressive focusing)

**Thresholds**: 0.10 to 0.55 (step 0.05)
**Metrics**: Evaluated on TEST set

---

## Summary — Best F1 per Configuration

| Model | γ | α_Glaucoma | α_Normal | Best Thresh | F1 | Precision | Recall | Accuracy |
|-------|---|-----------|----------|:-----------:|:--:|:---------:|:------:|:--------:|
"""

for _, row in summary_df.iterrows():
    report += (
        f"| {row['model']} | {row['gamma']:.1f} | {row['alpha_glaucoma']:.2f} | {row['alpha_normal']:.2f} "
        f"| {row['best_f1_threshold']:.2f} | {row['best_f1']:.4f} "
        f"| {row['precision_at_best_f1']:.4f} | {row['recall_at_best_f1']:.4f} "
        f"| {row['accuracy_at_best_f1']:.4f} |\n"
    )

for mtag in ["yolo11s", "yolo11l"]:
    sub = summary_df[summary_df["model"] == mtag]
    best = sub.loc[sub["best_f1"].idxmax()]
    report += f"\n**Best {mtag}**: γ={best['gamma']:.1f}, α=[{best['alpha_glaucoma']:.2f}, {best['alpha_normal']:.2f}], "
    report += f"F1={best['best_f1']:.4f}, Recall={best['recall_at_best_f1']:.4f}\n"

report += "\n---\n\n## Per-Configuration Threshold Tables\n\n"

for _, row in summary_df.iterrows():
    run_tag = f"{row['model']}_{row['config']}"
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

with open(os.path.join(RUNS_DIR, "Focal_Loss_Report.md"), "w") as f:
    f.write(report)

print(f"  ✅ Report saved to: {os.path.join(RUNS_DIR, 'Focal_Loss_Report.md')}")
print(f"\n{'=' * 60}")
print("FOCAL LOSS TRAINING COMPLETE")
print("=" * 60)
print(f"Results: {RUNS_DIR}")
