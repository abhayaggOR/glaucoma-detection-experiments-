"""
YOLO v11 Fine-Tuning (Last 5 Layers Only) - LARGE MODEL ONLY
======================================================
Trains YOLO11l-cls on the augmented dataset.
FREEZES the first 6 modules (0 to 5).
Only modules 6, 7, 8, 9, 10 (the last 5) are trained.

Large model:  batch=32, epochs=200, patience=75

Runs are saved under data_augment/runs14/
"""

import os
import shutil
from ultralytics import YOLO

# ==============================================================
# CONFIG
# ==============================================================
BASE_DIR = "/home/abhay/chaksu/20123135/Train/data_augment"
DATA_DIR = BASE_DIR                  # train/val/test already here
RUNS_DIR = os.path.join(BASE_DIR, "runs14")

# Total modules = 11 (0 to 10)
FREEZE_LAYERS = 6

# ==============================================================
# STEP 2: TRAIN YOLO11l-cls (Fine-Tune Last 5 Layers)
# ==============================================================
print("\n" + "=" * 60)
print("STEP 2: FINE-TUNING YOLO11l-cls (Last 5 Layers)")
print("  epochs=200, batch=16, patience=75 (Reduced batch to fix CUDA OOM)")
print("=" * 60)

model_l = YOLO("yolo11l-cls.pt")
# NOTE: Set resume=False carefully. We are starting from scratch because the previous attempt failed at epoch 1.
# Or we can just let it overwrite. exist_ok=True will overwrite unless resume=True is passed.
results_l = model_l.train(
    data=DATA_DIR,
    epochs=200,
    imgsz=512,
    rect=True,
    batch=16,
    patience=75,
    device=0,
    workers=0,
    freeze=FREEZE_LAYERS,
    project=RUNS_DIR,
    name="yolo11l_finetune_5layers",
    exist_ok=True,
)

print("\n✅ YOLO11l-cls 5-layer fine-tuning complete")

# ==============================================================
# STEP 3: EVALUATE BOTH MODELS ON TEST SET
# ==============================================================
print("\n" + "=" * 60)
print("STEP 3: EVALUATING ON TEST SET")
print("=" * 60)

for tag in ["yolo11s_finetune_5layers", "yolo11l_finetune_5layers"]:
    weights = os.path.join(RUNS_DIR, tag, "weights", "best.pt")
    if not os.path.exists(weights):
        print(f"⚠️ Weights not found at {weights}, skipping test evaluation for {tag}.")
        continue

    print(f"\n  Evaluating {tag} on test split ...")
    eval_model = YOLO(weights)
    metrics = eval_model.val(
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
    
    # Copy confusion matrix PNGs back into the main run folder
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
print("ALL FINE-TUNING & TEST EVALUATION COMPLETE")
print("=" * 60)
print(f"\nRuns saved at: {RUNS_DIR}")
