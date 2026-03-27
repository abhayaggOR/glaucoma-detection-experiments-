"""
Progressive Unfreezing Training — YOLO v11 (Augmented Data)
======================================================
Trains YOLO11s-cls and YOLO11l-cls using a 3-Stage sequential pipeline:
Stage 1: freeze=10, epochs=20, lr=1e-3
Stage 2: freeze=5,  epochs=40, lr=5e-4
Stage 3: freeze=0,  epochs=80, lr=1e-4

Global configs: optimizer='AdamW', batch=16, patience=75, Focal Loss.
Runs are saved under data_augment/runs16/
"""

import os
import shutil
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.tasks import ClassificationModel

# ==============================================================
# CONFIG
# ==============================================================
BASE_DIR = "/home/abhay/chaksu/20123135/Train/data_augment"
DATA_DIR = BASE_DIR
RUNS_DIR = os.path.join(BASE_DIR, "runs16")
os.makedirs(RUNS_DIR, exist_ok=True)

# ==============================================================
# FOCAL LOSS MONKEY-PATCH
# ==============================================================
class FocalClassificationLoss:
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        targets = batch["cls"]
        log_probs = F.log_softmax(preds.float(), dim=1)
        probs = log_probs.exp()
        targets_long = targets.long()
        p_t = probs.gather(1, targets_long.unsqueeze(1)).squeeze(1)
        alpha = self.alpha.to(device=preds.device, dtype=preds.dtype)
        alpha_t = alpha.gather(0, targets_long)
        focal_weight = (1.0 - p_t) ** self.gamma
        ce_loss = F.nll_loss(log_probs, targets_long, reduction="none")
        loss = (alpha_t * focal_weight * ce_loss).mean()
        return loss, loss.detach()

_original_init_criterion = ClassificationModel.init_criterion
def _focal_init_criterion(self):
    return FocalClassificationLoss(torch.tensor([0.85, 0.15]), 2.0)
ClassificationModel.init_criterion = _focal_init_criterion

# ==============================================================
# EVALUATION METRIC HELPER
# ==============================================================
def evaluate_model(tag, weights_path):
    if not os.path.exists(weights_path):
        print(f"⚠️ {weights_path} not found.")
        return
        
    print(f"\n--- Evaluating {tag} ---\n")
    eval_model = YOLO(weights_path)
    metrics = eval_model.val(
        data=DATA_DIR, split="test", imgsz=512, batch=16,
        device=0, workers=0, plots=True, project=RUNS_DIR,
        name=f"{tag}_test_eval", exist_ok=True
    )
    
    # Calculate specialized metrics
    cm = metrics.confusion_matrix.matrix
    tp, fn, fp, tn = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (prec * recall) / (prec + recall) if (prec + recall) > 0 else 0
    acc = (tp + tn) / cm.sum()
    
    print(f"TP: {int(tp)}, FN: {int(fn)}, FP: {int(fp)}, TN: {int(tn)}")
    print(f"Recall: {recall:.4f}, Precision: {prec:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}\n")
    
    # Copy confusion matrices into run
    eval_dir = os.path.join(RUNS_DIR, f"{tag}_test_eval")
    run_dir  = os.path.join(RUNS_DIR, f"{tag}_stage3") # The final result resides here
    for cm_file in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
        src = os.path.join(eval_dir, cm_file)
        dst = os.path.join(run_dir, cm_file)
        if os.path.exists(src): shutil.copy2(src, dst)


# ==============================================================
# MAIN TRAINING LOOP
# ==============================================================
if __name__ == "__main__":
    configs = [("yolo11s", "yolo11s-cls.pt"), ("yolo11l", "yolo11l-cls.pt")]
    
    for tag, model_pt in configs:
        print("\n" + "=" * 60)
        print(f"STARTING PROGRESSIVE UNFREEZING: {tag}")
        print("=" * 60)
        
        # Stage 1 (Head training: freeze=10, epochs=20, lr=1e-3)
        print(f"\n>> {tag} | STAGE 1: Head Training (freeze=10) <<")
        stage1_name = f"{tag}_stage1"
        model1 = YOLO(model_pt)
        model1.train(
            data=DATA_DIR, epochs=20, imgsz=512, rect=True, batch=16, patience=75,
            device=0, workers=0, optimizer="AdamW", lr0=1e-3, freeze=10,
            project=RUNS_DIR, name=stage1_name, exist_ok=True
        )
        stage1_weights = os.path.join(RUNS_DIR, stage1_name, "weights", "best.pt")
        
        # Stage 2 (Partial unfreeze: freeze=5, epochs=40, lr=5e-4)
        print(f"\n>> {tag} | STAGE 2: Partial Unfreeze (freeze=5) <<")
        stage2_name = f"{tag}_stage2"
        # Notice we initialize entirely a new YOLO with the Stage 1 weights.
        model2 = YOLO(stage1_weights) 
        model2.train(
            data=DATA_DIR, epochs=40, imgsz=512, rect=True, batch=16, patience=75,
            device=0, workers=0, optimizer="AdamW", lr0=5e-4, freeze=5,
            project=RUNS_DIR, name=stage2_name, exist_ok=True
        )
        stage2_weights = os.path.join(RUNS_DIR, stage2_name, "weights", "best.pt")
        
        # Stage 3 (Full fine-tuning: freeze=0, epochs=80, lr=1e-4)
        print(f"\n>> {tag} | STAGE 3: Full Fine-Tuning (freeze=0) <<")
        stage3_name = f"{tag}_stage3"
        model3 = YOLO(stage2_weights)
        model3.train(
            data=DATA_DIR, epochs=80, imgsz=512, rect=True, batch=16, patience=75,
            device=0, workers=0, optimizer="AdamW", lr0=1e-4, freeze=0,
            project=RUNS_DIR, name=stage3_name, exist_ok=True
        )
        
    print("\n" + "=" * 60)
    print("FINISHED ALL TARGET TRAINING, PROCEEDING TO EVALUATION")
    print("=" * 60)
    
    ClassificationModel.init_criterion = _original_init_criterion
    
    for tag, _ in configs:
        stage3_weights = os.path.join(RUNS_DIR, f"{tag}_stage3", "weights", "best.pt")
        evaluate_model(tag, stage3_weights)
