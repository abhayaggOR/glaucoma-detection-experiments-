"""
Differential Learning Rate Training — YOLO v11 (Augmented Data)
======================================================
Trains YOLO11s-cls and YOLO11l-cls using PyTorch gradient hooks to mimic
differential learning rates across the architecture while maintaining Focal Loss.

Early layers (0-3): lr=1e-5 (scale=0.1)
Middle layers (4-7): lr=5e-5 (scale=0.5)
Final layers (8-10): lr=1e-4 (scale=1.0)

Global configs: optimizer='AdamW', lr0=1e-4, epochs=200, batch=16, patience=75.
Runs are saved under data_augment/runs15/
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
RUNS_DIR = os.path.join(BASE_DIR, "runs15")
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
# GRADIENT HOOKS FOR DIFFERENTIAL LR
# ==============================================================
def apply_differential_lr_hooks(trainer):
    """
    Registers backward hooks on module parameters to dynamically scale gradients
    prior to the optimizer step, bypassing YOLO's global LR scheduler lock.
    """
    model = trainer.model
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        parts = name.split('.')
        idx = -1
        if 'model' in parts:
            try:
                idx = int(parts[parts.index('model') + 1])
            except ValueError:
                pass
                
        # Differential Ratios
        if 0 <= idx <= 3:
            scale = 0.1   # -> 1e-4 * 0.1 = 1e-5
        elif 4 <= idx <= 7:
            scale = 0.5   # -> 1e-4 * 0.5 = 5e-5
        else:
            scale = 1.0   # -> 1e-4 * 1.0 = 1e-4
            
        param.register_hook(lambda grad, s=scale: grad * s)
    
    print(f"\n✅ DIFFERENTIAL LR HOOKS APPLIED (Early: x0.1, Middle: x0.5, Final: x1.0)\n")

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
    run_dir  = os.path.join(RUNS_DIR, tag)
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
        run_tag = f"{tag}_difflr_focal"
        print("=" * 60)
        print(f"TRAINING DIFFERENTIAL LR: {run_tag}")
        print("=" * 60)
        
        model = YOLO(model_pt)
        model.add_callback("on_train_start", apply_differential_lr_hooks)
        
        # We MUST explicitly pass optimizer='AdamW' to prevent optimizer='auto' from overriding lr0
        model.train(
            data=DATA_DIR,
            epochs=200,
            imgsz=512,
            rect=True,
            batch=16,
            patience=75,
            device=0,
            workers=0,
            optimizer="AdamW",
            lr0=1e-4,          # Global LR representing the 'Final' layers
            project=RUNS_DIR,
            name=run_tag,
            exist_ok=True,
        )
        
    print("\n" + "=" * 60)
    print("FINISHED ALL TARGET TRAINING, PROCEEDING TO EVALUATION")
    print("=" * 60)
    
    # Restore original init criterion before evaluation loop to keep metrics pure
    ClassificationModel.init_criterion = _original_init_criterion
    
    for tag, _ in configs:
        evaluate_model(f"{tag}_difflr_focal", os.path.join(RUNS_DIR, f"{tag}_difflr_focal", "weights", "best.pt"))
