"""
EfficientNet — Evaluate B0 + Train B1, B2, B3
===============================================
B0 already trained → just evaluate on test data.
B1, B2, B3 → train + evaluate.
"""

import os
import sys
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
from torchvision import transforms
from PIL import Image
import torchvision

BASE_DIR  = "/home/abhay/chaksu/20123135/Train/data_augment"
DATA_DIR  = BASE_DIR
TEST_DIR  = os.path.join(BASE_DIR, "test")
CLASS_NAMES    = ["GLAUCOMA_SUSPECT", "NORMAL"]
POSITIVE_CLASS = "GLAUCOMA_SUSPECT"
GLAUCOMA_IDX   = 0
IMGSZ          = 512
EPOCHS         = 300
LABEL_SMOOTHING = 0.1

# Load test set
test_images, test_labels = [], []
for cls_name in CLASS_NAMES:
    cls_dir = os.path.join(TEST_DIR, cls_name)
    if not os.path.exists(cls_dir): continue
    for fname in sorted(os.listdir(cls_dir)):
        fpath = os.path.join(cls_dir, fname)
        if os.path.isfile(fpath):
            test_images.append(fpath)
            test_labels.append(cls_name)
y_true = np.array([1 if lbl == POSITIVE_CLASS else 0 for lbl in test_labels])
print(f"Test: {len(test_images)} images ({test_labels.count('GLAUCOMA_SUSPECT')} G, {test_labels.count('NORMAL')} N)")


# ==============================================================
# WRAPPER (needed for both checkpoint loading and training)
# ==============================================================
class TorchvisionClassificationWrapper(nn.Module):
    def __init__(self, backbone, label_smoothing=0.1):
        super().__init__()
        self.model = backbone
        self.label_smoothing = label_smoothing
        self.stride = torch.tensor([1])
        self.names = {}
        self.yaml = {"nc": 2}
        self.args = {"imgsz": IMGSZ}
        self.transforms = None
        self.criterion = None

    def forward(self, x, **kwargs):
        if isinstance(x, dict):
            imgs = x["img"]
            preds = self.model(imgs)
            loss = F.cross_entropy(preds.float(), x["cls"],
                                   label_smoothing=self.label_smoothing, reduction="mean")
            return loss, loss.detach()
        return self.model(x)

    def loss(self, batch, preds=None):
        if preds is None:
            preds = self.model(batch["img"])
        loss = F.cross_entropy(preds.float(), batch["cls"],
                               label_smoothing=self.label_smoothing, reduction="mean")
        return loss, loss.detach()


# Register for unpickling
import __main__
__main__.TorchvisionClassificationWrapper = TorchvisionClassificationWrapper


# ==============================================================
# EVALUATE FUNCTION
# ==============================================================
def evaluate_on_test(model_path, run_tag, out_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    saved_model = ckpt.get("ema") or ckpt.get("model")
    saved_model = saved_model.float().to(device).eval()
    backbone = saved_model.model if hasattr(saved_model, "model") else saved_model
    backbone.eval()

    eval_transforms = transforms.Compose([
        transforms.Resize((IMGSZ, IMGSZ)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_probs = []
    with torch.no_grad():
        for img_path in test_images:
            img = Image.open(img_path).convert("RGB")
            img_t = eval_transforms(img).unsqueeze(0).to(device)
            logits = backbone(img_t)
            probs = F.softmax(logits.float(), dim=1)
            all_probs.append(probs.cpu().numpy()[0])
    all_probs = np.array(all_probs)

    y_pred = (all_probs[:, GLAUCOMA_IDX] >= 0.5).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    print(f"\n  TEST METRICS ({run_tag}, threshold=0.50):")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    print(f"    TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    run_dir = os.path.join(out_dir, run_tag)
    os.makedirs(run_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    for normed, cm_data, suffix in [
        (False, cm, "confusion_matrix_test"),
        (True, cm_norm, "confusion_matrix_test_normalized"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_data, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        tl = ["NORMAL", "GLAUCOMA_SUSPECT"]
        ax.set(xticks=[0,1], yticks=[0,1], xticklabels=tl, yticklabels=tl,
               xlabel="Predicted", ylabel="True",
               title=f"{run_tag} (thresh=0.50)" + (" [norm]" if normed else ""))
        fmt = ".2f" if normed else "d"
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm_data[i,j], fmt), ha="center", va="center",
                        color="white" if cm_data[i,j] > cm_data.max()/2 else "black")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{run_tag}_{suffix}.png"), dpi=150)
        plt.close(fig)
        print(f"    ✅ {suffix}.png saved")

    with open(os.path.join(out_dir, f"{run_tag}_test_metrics.txt"), "w") as f:
        f.write(f"Model: {run_tag}\nThreshold: 0.50\n")
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n")
        f.write(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}\n")

    del backbone, saved_model
    torch.cuda.empty_cache()
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# ==============================================================
# CUSTOM TRAINER
# ==============================================================
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.nn.tasks import ClassificationModel


class EfficientNetTrainer(ClassificationTrainer):
    def setup_model(self):
        model_name = str(self.model)
        if model_name in torchvision.models.__dict__:
            backbone = torchvision.models.__dict__[model_name](
                weights="IMAGENET1K_V1" if self.args.pretrained else None)
            ClassificationModel.reshape_outputs(backbone, self.data["nc"])
            self.model = TorchvisionClassificationWrapper(backbone, label_smoothing=LABEL_SMOOTHING)
            print(f"  ✅ Loaded {model_name} with label_smoothing={LABEL_SMOOTHING}")
            return None
        return super().setup_model()

    def get_model(self, cfg=None, weights=None, verbose=True):
        return self.model


# ==============================================================
# MAIN
# ==============================================================
all_results = []

# --- STEP 1: Evaluate B0 (already trained) ---
print("\n" + "#" * 60)
print("STEP 1: EVALUATE efficientnet_b0 (already trained)")
print("#" * 60)
b0_weights = os.path.join(BASE_DIR, "runs7/efficientnet_b0/weights/best.pt")
if os.path.exists(b0_weights):
    metrics = evaluate_on_test(b0_weights, "efficientnet_b0", os.path.join(BASE_DIR, "runs7"))
    all_results.append({"model": "efficientnet_b0", "runs": "runs7", **metrics})
else:
    print("  ❌ B0 weights not found, skipping")

# --- STEP 2: Train B1, B2, B3 ---
REMAINING_MODELS = [
    ("efficientnet_b1", "runs8",  16, 100),
    ("efficientnet_b2", "runs9",  32, 100),
    ("efficientnet_b3", "runs10", 32, 100),
]

for model_name, runs_name, batch_size, patience in REMAINING_MODELS:
    runs_dir = os.path.join(BASE_DIR, runs_name)
    os.makedirs(runs_dir, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(f"TRAINING: {model_name} → {runs_name}")
    print(f"  Batch: {batch_size}, Epochs: {EPOCHS}, Patience: {patience}")
    print(f"{'#' * 60}")

    trainer = EfficientNetTrainer(overrides={
        "model": model_name, "data": DATA_DIR, "epochs": EPOCHS,
        "imgsz": IMGSZ, "rect": True, "batch": batch_size,
        "patience": patience, "device": 0, "workers": 0,
        "pretrained": True, "project": runs_dir, "name": model_name,
        "exist_ok": True,
    })
    trainer.train()
    print(f"  ✅ {model_name} training complete")

    weights_path = os.path.join(runs_dir, model_name, "weights", "best.pt")
    metrics = evaluate_on_test(weights_path, model_name, runs_dir)
    all_results.append({"model": model_name, "runs": runs_name, **metrics})

    del trainer
    torch.cuda.empty_cache()

# --- SUMMARY ---
print(f"\n{'=' * 60}")
print("SUMMARY")
print("=" * 60)
report = f"""# EfficientNet B0–B3 Training Report — Augmented Data with Label Smoothing

**Label Smoothing**: {LABEL_SMOOTHING} | **Image Size**: {IMGSZ} | **Epochs**: {EPOCHS} | **Pretrained**: ImageNet1K_V1

| Model | Runs | Batch | Patience | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |
|-------|:----:|:-----:|:--------:|:--------:|:---------:|:------:|:--:|:--:|:--:|:--:|:--:|
"""
for r in all_results:
    report += (f"| {r['model']} | {r['runs']} | - | - | {r['accuracy']:.4f} | "
               f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | "
               f"{r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} |\n")
    print(f"  {r['model']}: Acc={r['accuracy']:.4f} Prec={r['precision']:.4f} "
          f"Rec={r['recall']:.4f} F1={r['f1']:.4f}")

with open(os.path.join(BASE_DIR, "EfficientNet_Training_Report.md"), "w") as f:
    f.write(report)
print(f"\n  ✅ Report: {BASE_DIR}/EfficientNet_Training_Report.md")
print("DONE!")
