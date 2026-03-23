import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO
import os

BASE_DIR = "/home/abhay/chaksu/20123135/Train/data_augment"
TEST_DIR = os.path.join(BASE_DIR, "test")

def evaluate_yolo(model_path, model_name):
    if not os.path.exists(model_path):
        print(f"Skipping {model_name}, weights not found.")
        return

    print(f"--- Evaluating {model_name} ---")
    model = YOLO(model_path)
    
    # Run evaluation
    metrics = model.val(data=BASE_DIR, split='test', plots=False, verbose=False)
    
    # We want exact TP/TN/FP/FN. We can get it directly from the confusion matrix object
    # The confusion matrix is inside metrics.confusion_matrix.matrix
    cm = metrics.confusion_matrix.matrix
    # YOLO classes: 0 is GLAUCOMA_SUSPECT, 1 is NORMAL by default alphabetical ordering
    # Let's verify by printing names
    print(f"Classes: {model.names}")
    
    # Print the raw matrix
    print(f"Confusion Matrix (Raw):\n{cm}")
    
    # Calculate precision, recall
    # Assuming standard orientation: rows are true, cols are pred
    # For class 0 (GLAUCOMA_SUSPECT):
    tp_glaucoma = cm[0, 0]
    fn_glaucoma = cm[0, 1]
    fp_glaucoma = cm[1, 0]
    tn_glaucoma = cm[1, 1]
    
    recall_g = tp_glaucoma / (tp_glaucoma + fn_glaucoma) if (tp_glaucoma + fn_glaucoma) > 0 else 0
    prec_g = tp_glaucoma / (tp_glaucoma + fp_glaucoma) if (tp_glaucoma + fp_glaucoma) > 0 else 0
    f1_g = 2 * (prec_g * recall_g) / (prec_g + recall_g) if (prec_g + recall_g) > 0 else 0
    
    acc = (tp_glaucoma + tn_glaucoma) / cm.sum()
    
    print(f"GLAUCOMA_SUSPECT -> TP: {int(tp_glaucoma)}, FN: {int(fn_glaucoma)}, FP: {int(fp_glaucoma)}, TN: {int(tn_glaucoma)}")
    print(f"GLAUCOMA_SUSPECT -> Recall: {recall_g:.4f}, Precision: {prec_g:.4f}, F1: {f1_g:.4f}, Accuracy: {acc:.4f}\n")

evaluate_yolo(f"{BASE_DIR}/runs12/yolo11s_simple/weights/best.pt", "YOLO11s-cls")
evaluate_yolo(f"{BASE_DIR}/runs12/yolo11l_simple/weights/best.pt", "YOLO11l-cls")
