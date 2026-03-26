import os
from ultralytics import YOLO

def evaluate(path):
    if not os.path.exists(path):
        print(f"Not found: {path}\n")
        return
    print(f"--- Evaluating {path} ---")
    model = YOLO(path)
    metrics = model.val(data="/home/abhay/chaksu/20123135/Train/data_augment", split='test', plots=False, verbose=False)
    cm = metrics.confusion_matrix.matrix
    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (prec * recall) / (prec + recall) if (prec + recall) > 0 else 0
    acc = (tp + tn) / cm.sum()
    print(f"TP: {int(tp)}, FN: {int(fn)}, FP: {int(fp)}, TN: {int(tn)}")
    print(f"Recall: {recall:.4f}, Precision: {prec:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}\n")

evaluate("/home/abhay/chaksu/20123135/Train/data_augment/runs13/yolo11s_finetune/weights/best.pt")
evaluate("/home/abhay/chaksu/20123135/Train/data_augment/runs13/yolo11l_finetune/weights/best.pt")
evaluate("/home/abhay/chaksu/20123135/Train/data_augment/runs14/yolo11s_finetune_5layers/weights/best.pt")
evaluate("/home/abhay/chaksu/20123135/Train/data_augment/runs14/yolo11l_finetune_5layers/weights/best.pt")
