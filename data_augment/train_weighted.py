"""
YOLO v11 Training — Class-Weighted Cross Entropy (Augmented Data)
==================================================================
Trains YOLO11s-cls and YOLO11l-cls on the augmented dataset using
class-weighted cross entropy loss to handle class imbalance.

Class weights are computed using inverse frequency:
    weight[c] = total_samples / (num_classes * count[c])

Small model:  batch=16, epochs=300, patience=100, imgsz=512
Large model:  batch=32, epochs=300, patience=75,  imgsz=512

Results (incl. test-set metrics) → data_augment/runs3/
"""

import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F

# ==============================================================
# CONFIG
# ==============================================================
BASE_DIR  = "/home/abhay/chaksu/20123135/Train/data_augment"
DATA_DIR  = BASE_DIR                  # train/val/test already here
RUNS_DIR  = os.path.join(BASE_DIR, "runs3")
TRAIN_DIR = os.path.join(BASE_DIR, "train")

CLASS_NAMES = sorted(os.listdir(TRAIN_DIR))   # alphabetical → [GLAUCOMA_SUSPECT, NORMAL]
os.makedirs(RUNS_DIR, exist_ok=True)

# ==============================================================
# STEP 0: COMPUTE CLASS WEIGHTS
# ==============================================================
print("=" * 60)
print("STEP 0: COMPUTING CLASS WEIGHTS")
print("=" * 60)

class_counts = []
for cls_name in CLASS_NAMES:
    cls_dir = os.path.join(TRAIN_DIR, cls_name)
    count = len([f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))])
    class_counts.append(count)
    print(f"  {cls_name}: {count} images")

class_counts = np.array(class_counts, dtype=np.float64)
total = class_counts.sum()
num_classes = len(class_counts)
class_weights = total / (num_classes * class_counts)

print(f"\n  Total training images: {int(total)}")
print(f"  Class weights (inverse frequency):")
for i, cls in enumerate(CLASS_NAMES):
    print(f"    {cls} (class {i}): {class_weights[i]:.4f}")

# Convert to tensor
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# ==============================================================
# STEP 1: MONKEY-PATCH CLASSIFICATION LOSS
# ==============================================================
print("\n" + "=" * 60)
print("STEP 1: PATCHING CLASSIFICATION LOSS WITH CLASS WEIGHTS")
print("=" * 60)

from ultralytics.nn.tasks import ClassificationModel


class WeightedClassificationLoss:
    """Classification loss with class weights for imbalanced datasets."""

    def __init__(self, weight):
        self.weight = weight
        print(f"  ✅ WeightedClassificationLoss initialised with weights: {weight}")

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        w = self.weight.to(device=preds.device, dtype=preds.dtype)
        loss = F.cross_entropy(preds, batch["cls"], weight=w, reduction="mean")
        return loss, loss.detach()


# Store the original init_criterion
_original_init_criterion = ClassificationModel.init_criterion


def _weighted_init_criterion(self):
    """Return weighted classification loss instead of the default."""
    return WeightedClassificationLoss(class_weights_tensor)


# Patch it
ClassificationModel.init_criterion = _weighted_init_criterion
print("  ✅ ClassificationModel.init_criterion patched")


# ==============================================================
# STEP 2: TRAIN YOLO11s-cls
# ==============================================================
print("\n" + "=" * 60)
print("STEP 2: TRAINING YOLO11s-cls (class-weighted CE)")
print("  epochs=300, batch=16, patience=100, imgsz=512")
print("=" * 60)

from ultralytics import YOLO

model_s = YOLO("yolo11s-cls")
results_s = model_s.train(
    data=DATA_DIR,
    epochs=300,
    imgsz=512,
    rect=True,
    batch=16,
    patience=100,
    device=0,
    workers=0,
    project=RUNS_DIR,
    name="yolo11s_weighted",
    exist_ok=True,
)

print("\n✅ YOLO11s-cls training complete")

# ==============================================================
# STEP 3: TRAIN YOLO11l-cls
# ==============================================================
print("\n" + "=" * 60)
print("STEP 3: TRAINING YOLO11l-cls (class-weighted CE)")
print("  epochs=300, batch=32, patience=75, imgsz=512")
print("=" * 60)

model_l = YOLO("yolo11l-cls")
results_l = model_l.train(
    data=DATA_DIR,
    epochs=300,
    imgsz=512,
    rect=True,
    batch=32,
    patience=75,
    device=0,
    workers=0,
    project=RUNS_DIR,
    name="yolo11l_weighted",
    exist_ok=True,
)

print("\n✅ YOLO11l-cls training complete")

# ==============================================================
# STEP 4: EVALUATE BOTH MODELS ON TEST SET
# ==============================================================
print("\n" + "=" * 60)
print("STEP 4: EVALUATING ON TEST SET")
print("=" * 60)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load test ground truth
TEST_DIR = os.path.join(BASE_DIR, "test")
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

print(f"  Total test images: {len(test_images)}")
for cls in CLASS_NAMES:
    print(f"    {cls}: {test_labels.count(cls)}")

POSITIVE_CLASS = "GLAUCOMA_SUSPECT"
glaucoma_idx = 0   # alphabetical: GLAUCOMA_SUSPECT=0, NORMAL=1
y_true = np.array([1 if lbl == POSITIVE_CLASS else 0 for lbl in test_labels])

for tag in ["yolo11s_weighted", "yolo11l_weighted"]:
    weights_path = os.path.join(RUNS_DIR, tag, "weights", "best.pt")
    print(f"\n  --- {tag} ---")
    print(f"  Weights: {weights_path}")

    eval_model = YOLO(weights_path)

    # Also run YOLO's built-in val for confusion matrix PNGs
    eval_model.val(
        data=DATA_DIR,
        split="test",
        imgsz=512,
        batch=32,
        device=0,
        workers=0,
        plots=True,
        project=RUNS_DIR,
        name=f"{tag}_test_eval",
        exist_ok=True,
    )

    # Copy confusion matrix PNGs into main run folder
    eval_dir = os.path.join(RUNS_DIR, f"{tag}_test_eval")
    run_dir  = os.path.join(RUNS_DIR, tag)
    for cm_file in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
        src = os.path.join(eval_dir, cm_file)
        dst = os.path.join(run_dir, cm_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"    ✅ {cm_file} → {run_dir}")

    # Compute metrics at default threshold (0.5)
    all_probs = []
    batch_size = 32
    for i in range(0, len(test_images), batch_size):
        batch_paths = test_images[i:i + batch_size]
        results = eval_model.predict(source=batch_paths, imgsz=512, device=0, verbose=False)
        for r in results:
            probs = r.probs.data.cpu().numpy()
            all_probs.append(probs)
    all_probs = np.array(all_probs)

    y_pred = (all_probs[:, glaucoma_idx] >= 0.5).astype(int)
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

    del eval_model
    torch.cuda.empty_cache()

# Restore original
ClassificationModel.init_criterion = _original_init_criterion

print("\n" + "=" * 60)
print("ALL TRAINING & TEST EVALUATION COMPLETE")
print("=" * 60)
print(f"\nRuns saved at: {RUNS_DIR}")
