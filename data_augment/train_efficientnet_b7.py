"""
EfficientNet B7 Training — Augmented Data with Label Smoothing
==================================================================
Trains EfficientNet-B7 (torchvision, pretrained on ImageNet)
on the augmented dataset with label smoothing (0.1).

Uses a custom wrapper to make torchvision models compatible with
ultralytics' training loop (which passes batch dicts).

B7: batch=8, epochs=300, patience=100 → runs11

All models evaluated on test set with metrics + confusion matrices.
"""

import os
import shutil
import numpy as np
import torch
import torch.nn as nn
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
TEST_DIR  = os.path.join(BASE_DIR, "test")

CLASS_NAMES    = ["GLAUCOMA_SUSPECT", "NORMAL"]
POSITIVE_CLASS = "GLAUCOMA_SUSPECT"
GLAUCOMA_IDX   = 0
IMGSZ          = 512
EPOCHS         = 300
LABEL_SMOOTHING = 0.1

# (model_name, runs_dir_name, batch_size, patience)
# We use a batch size of 8 for B7 as it is much larger and might cause OOM on standard GPUs
MODEL_CONFIGS = [
    ("efficientnet_b7", "runs11",  8, 100),
]

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
# WRAPPER: Make torchvision models compatible with ultralytics
# ==============================================================
class TorchvisionClassificationWrapper(nn.Module):
    """
    Wraps a torchvision classification model so it works with
    ultralytics' BaseTrainer training loop.

    When called with a dict batch (from the trainer), extracts
    images and computes loss. When called with a tensor (inference),
    passes through to the underlying model.
    """
    def __init__(self, backbone, label_smoothing=0.1):
        super().__init__()
        self.model = backbone
        self.label_smoothing = label_smoothing
        # Set attributes that ultralytics expects
        self.stride = torch.tensor([1])
        self.names = {}
        self.yaml = {"nc": 2}
        self.args = {"imgsz": IMGSZ}
        self.transforms = None  # set by trainer
        self.criterion = None   # ultralytics checks hasattr(model.criterion, "update")

    def forward(self, x, **kwargs):
        """Forward pass — handle both dict (training) and tensor (inference)."""
        if isinstance(x, dict):
            # Training mode: extract images and compute loss
            imgs = x["img"]
            preds = self.model(imgs)
            loss = F.cross_entropy(
                preds.float(), x["cls"],
                label_smoothing=self.label_smoothing,
                reduction="mean",
            )
            return loss, loss.detach()
        else:
            # Inference mode: just return predictions
            return self.model(x)

    def loss(self, batch, preds=None):
        """Compute loss (called by BaseTrainer)."""
        if preds is None:
            preds = self.model(batch["img"])
        loss = F.cross_entropy(
            preds.float(), batch["cls"],
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )
        return loss, loss.detach()


# ==============================================================
# CUSTOM TRAINER
# ==============================================================
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.nn.tasks import ClassificationModel
from ultralytics import YOLO
from ultralytics.utils import RANK
import torchvision


class EfficientNetTrainer(ClassificationTrainer):
    """Custom trainer that properly wraps torchvision models."""

    def setup_model(self):
        """Load torchvision model and wrap it."""
        model_name = str(self.model)
        # Note: torchvision b7 is available
        if model_name in torchvision.models.__dict__:
            backbone = torchvision.models.__dict__[model_name](
                weights="IMAGENET1K_V1" if self.args.pretrained else None
            )
            # Reshape output to match num classes
            ClassificationModel.reshape_outputs(backbone, self.data["nc"])
            # Wrap it
            self.model = TorchvisionClassificationWrapper(
                backbone, label_smoothing=LABEL_SMOOTHING
            )
            print(f"  ✅ Loaded {model_name} with label_smoothing={LABEL_SMOOTHING}")
            return None
        else:
            # handle efficientnet_b7 explicitly if not in __dict__ just in case
            if model_name == "efficientnet_b7":
                backbone = torchvision.models.efficientnet_b7(weights="IMAGENET1K_V1" if self.args.pretrained else None)
                ClassificationModel.reshape_outputs(backbone, self.data["nc"])
                self.model = TorchvisionClassificationWrapper(
                    backbone, label_smoothing=LABEL_SMOOTHING
                )
                print(f"  ✅ Loaded {model_name} with label_smoothing={LABEL_SMOOTHING}")
                return None
            return super().setup_model()

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Not used for torchvision models, but required by parent."""
        return self.model


# ==============================================================
# HELPER: Evaluate on test set
# ==============================================================
from torchvision import transforms
from PIL import Image


def load_model_from_checkpoint(model_path, device="cuda:0"):
    """Load torchvision model from ultralytics checkpoint."""
    # Register wrapper class in __main__ so torch.load can unpickle it
    import __main__
    __main__.TorchvisionClassificationWrapper = TorchvisionClassificationWrapper

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    # ultralytics saves EMA under "ema" key, but for torchvision wrappers
    # it may be None — fall back to "model" key
    saved_model = ckpt.get("ema") or ckpt.get("model")
    if saved_model is None:
        raise ValueError(f"No model found in checkpoint: {model_path}")
    saved_model = saved_model.float().to(device)
    saved_model.eval()
    # Extract inner backbone from wrapper
    if hasattr(saved_model, "model"):
        backbone = saved_model.model
    else:
        backbone = saved_model
    backbone.eval()
    return backbone


def evaluate_on_test(model_path, run_tag, out_dir):
    """Evaluate model on test set: metrics at threshold 0.5 + confusion matrix."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(model_path, device)

    # Inference transforms (match ultralytics classify preprocessing)
    eval_transforms = transforms.Compose([
        transforms.Resize((IMGSZ, IMGSZ)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Run inference
    all_probs = []
    with torch.no_grad():
        for img_path in test_images:
            img = Image.open(img_path).convert("RGB")
            img_t = eval_transforms(img).unsqueeze(0).to(device)
            logits = model(img_t)
            probs = F.softmax(logits.float(), dim=1)
            all_probs.append(probs.cpu().numpy()[0])
    all_probs = np.array(all_probs)
    print(f"  Inference complete. Shape: {all_probs.shape}")

    # Metrics at threshold 0.5
    y_pred = (all_probs[:, GLAUCOMA_IDX] >= 0.5).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    print(f"\n  TEST METRICS (threshold=0.50):")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    print(f"    TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    # Generate confusion matrix PNGs
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    run_dir = os.path.join(out_dir, run_tag)
    os.makedirs(run_dir, exist_ok=True)
    for normed, cm_data, suffix in [
        (False, cm, "confusion_matrix_test"),
        (True, cm_norm, "confusion_matrix_test_normalized"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_data, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        tick_labels = ["NORMAL", "GLAUCOMA_SUSPECT"]
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=tick_labels, yticklabels=tick_labels,
               xlabel="Predicted label", ylabel="True label",
               title=f"{run_tag} — Test Set (thresh=0.50)"
                     + (" [normalized]" if normed else ""))
        fmt = ".2f" if normed else "d"
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm_data[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm_data[i, j] > cm_data.max() / 2 else "black")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{run_tag}_{suffix}.png"), dpi=150)
        plt.close(fig)
        print(f"    ✅ {suffix}.png saved")

    # Save metrics to text file
    metrics_path = os.path.join(out_dir, f"{run_tag}_test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Model: {run_tag}\n")
        f.write(f"Threshold: 0.50\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}\n")
    print(f"    ✅ Metrics saved to: {metrics_path}")

    del model
    torch.cuda.empty_cache()
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# ==============================================================
# MAIN: TRAIN ALL MODELS
# ==============================================================
all_results = []

for model_name, runs_name, batch_size, patience in MODEL_CONFIGS:
    runs_dir = os.path.join(BASE_DIR, runs_name)
    os.makedirs(runs_dir, exist_ok=True)
    run_tag = model_name

    print(f"\n{'#' * 60}")
    print(f"MODEL: {model_name}")
    print(f"  Output: {runs_dir}")
    print(f"  Batch: {batch_size}, Epochs: {EPOCHS}, Patience: {patience}")
    print(f"  Label Smoothing: {LABEL_SMOOTHING}")
    print(f"{'#' * 60}")

    # Train using custom EfficientNetTrainer
    trainer = EfficientNetTrainer(overrides={
        "model":     model_name,
        "data":      DATA_DIR,
        "epochs":    EPOCHS,
        "imgsz":     IMGSZ,
        "rect":      True,
        "batch":     batch_size,
        "patience":  patience,
        "device":    0,
        "workers":   0,
        "pretrained": True,
        "project":   runs_dir,
        "name":      run_tag,
        "exist_ok":  True,
    })
    trainer.train()
    print(f"\n  ✅ {model_name} training complete")

    # Evaluate on test set
    weights_path = os.path.join(runs_dir, run_tag, "weights", "best.pt")
    print(f"\n  Evaluating on test set...")
    metrics = evaluate_on_test(weights_path, run_tag, runs_dir)

    all_results.append({
        "model": model_name,
        "runs": runs_name,
        **metrics,
    })

    del trainer
    torch.cuda.empty_cache()

# ==============================================================
# SUMMARY REPORT
# ==============================================================
print(f"\n{'=' * 60}")
print("GENERATING SUMMARY REPORT")
print("=" * 60)

report = f"""# EfficientNet B7 Training Report — Augmented Data with Label Smoothing

**Dataset**: Augmented data (data_augment)
**Label Smoothing**: {LABEL_SMOOTHING}
**Image Size**: {IMGSZ}
**Epochs**: {EPOCHS}
**Pretrained**: ImageNet1K_V1

---

## Training Configuration

| Model | Runs Dir | Batch Size | Patience |
|-------|:--------:|:----------:|:--------:|
| EfficientNet-B7 | runs11 | 8 | 100 |

---

## Test Set Metrics (threshold = 0.50)

| Model | Accuracy | Precision | Recall | F1 Score | TP | FP | TN | FN |
|-------|:--------:|:---------:|:------:|:--------:|:--:|:--:|:--:|:--:|
"""

for r in all_results:
    report += (
        f"| {r['model']} | {r['accuracy']:.4f} | {r['precision']:.4f} "
        f"| {r['recall']:.4f} | {r['f1']:.4f} "
        f"| {r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} |\n"
    )

report += "\n---\n\n## Confusion Matrices\n\n"
for r in all_results:
    report += f"### {r['model']} ({r['runs']})\n"
    report += f"- `{r['model']}_confusion_matrix_test.png`\n"
    report += f"- `{r['model']}_confusion_matrix_test_normalized.png`\n\n"

# Save combined report
combined_report_path = os.path.join(BASE_DIR, "EfficientNet_B7_Training_Report.md")
with open(combined_report_path, "w") as f:
    f.write(report)
print(f"  ✅ Combined report: {combined_report_path}")

# Also save report in each runs dir
for r in all_results:
    runs_dir = os.path.join(BASE_DIR, r["runs"])
    with open(os.path.join(runs_dir, f"{r['model']}_report.md"), "w") as f:
        f.write(f"# {r['model']} Report\n\n")
        f.write(f"**Runs Dir**: {r['runs']}\n")
        f.write(f"**Label Smoothing**: {LABEL_SMOOTHING}\n\n")
        f.write(f"## Test Metrics (threshold=0.50)\n\n")
        f.write(f"| Metric | Value |\n|--------|:-----:|\n")
        f.write(f"| Accuracy | {r['accuracy']:.4f} |\n")
        f.write(f"| Precision | {r['precision']:.4f} |\n")
        f.write(f"| Recall | {r['recall']:.4f} |\n")
        f.write(f"| F1 Score | {r['f1']:.4f} |\n")
        f.write(f"| TP | {r['tp']} |\n| FP | {r['fp']} |\n")
        f.write(f"| TN | {r['tn']} |\n| FN | {r['fn']} |\n")

print(f"\n{'=' * 60}")
print("ALL EFFICIENTNET TRAINING COMPLETE")
print("=" * 60)
