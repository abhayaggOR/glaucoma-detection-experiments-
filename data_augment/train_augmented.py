"""
YOLO v11 Training — Augmented Data (3:1 Ratio)
================================================
Trains YOLO11s-cls and YOLO11l-cls on the augmented dataset.

Small model:  batch=16, epochs=300, patience=100
Large model:  batch=32, epochs=300, patience=75

Runs are saved under data_augment/runs/
"""

import os

# ==============================================================
# CONFIG
# ==============================================================
BASE_DIR = "/home/abhay/chaksu/20123135/Train/data_augment"
DATA_DIR = BASE_DIR                  # train/val/test already here
RUNS_DIR = os.path.join(BASE_DIR, "runs")

# ==============================================================
# STEP 1: TRAIN YOLO11s-cls
# ==============================================================
print("=" * 60)
print("STEP 1: TRAINING YOLO11s-cls")
print("  epochs=300, batch=16, patience=100")
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
    name="yolo11s_augmented",
    exist_ok=True,
)

print("\n✅ YOLO11s-cls training complete")

# ==============================================================
# STEP 2: TRAIN YOLO11l-cls
# ==============================================================
print("\n" + "=" * 60)
print("STEP 2: TRAINING YOLO11l-cls")
print("  epochs=300, batch=32, patience=75")
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
    name="yolo11l_augmented",
    exist_ok=True,
)

print("\n✅ YOLO11l-cls training complete")

# ==============================================================
# STEP 3: EVALUATE BOTH MODELS ON TEST SET
# ==============================================================
# YOLO's default confusion matrices are generated on the val split.
# This step re-evaluates on the test split so the confusion matrices
# saved in runs/ reflect performance on the held-out 15% test data.
print("\n" + "=" * 60)
print("STEP 3: EVALUATING ON TEST SET (confusion matrices)")
print("=" * 60)

import shutil

for tag, mdl_obj in [("yolo11s_augmented", None), ("yolo11l_augmented", None)]:
    weights = os.path.join(RUNS_DIR, tag, "weights", "best.pt")
    print(f"\n  Evaluating {tag} on test split ...")
    eval_model = YOLO(weights)
    eval_model.val(
        data=DATA_DIR,
        split="test",
        imgsz=512,
        batch=32,
        device=0,
        workers=0,
        plots=True,            # generates confusion_matrix.png
        project=RUNS_DIR,
        name=f"{tag}_test_eval",
        exist_ok=True,
    )
    # Copy confusion matrix PNGs back into the main run folder,
    # replacing the val-based ones.
    eval_dir = os.path.join(RUNS_DIR, f"{tag}_test_eval")
    run_dir  = os.path.join(RUNS_DIR, tag)
    for cm_file in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
        src = os.path.join(eval_dir, cm_file)
        dst = os.path.join(run_dir, cm_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"    ✅ {cm_file} → {run_dir}")
        else:
            print(f"    ⚠️  {cm_file} not found in {eval_dir}")

print("\n" + "=" * 60)
print("ALL TRAINING & TEST EVALUATION COMPLETE")
print("=" * 60)
print(f"\nRuns saved at: {RUNS_DIR}")
