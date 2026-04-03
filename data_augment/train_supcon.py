import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE
import logging
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = "/home/abhay/chaksu/20123135/Train/data_augment"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
OUT_DIR = os.path.join(BASE_DIR, "runs17")
os.makedirs(OUT_DIR, exist_ok=True)

# YOLO Backbone (Runs6 best overall recall)
YOLO_WEIGHTS = os.path.join(BASE_DIR, "runs6", "yolo11s_gamma2.0_alpha0.80", "weights", "best.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
temperature = 0.07
supcon_epochs = 150
finetune_epochs = 40
img_size = 512

logging.basicConfig(
    filename=os.path.join(OUT_DIR, "supcon_training.log"),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
logger = logging.getLogger()
logger.info("Starting Runs17: Supervised Contrastive Learning Pipeline")

# ==========================================
# LOSS FUNCTION
# ==========================================
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss"""
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features shape: [batch_size, n_views, embed_dim]
        # labels shape: [batch_size]
        device = features.device
        batch_size = features.shape[0]
        
        # Merge batch and views for pairwise similarity
        # [2*B, embed_dim]
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        # [2*B, 2*B]
        sim_matrix = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # Create mask of positive pairs
        # labels repeated for the two views: [label1, label2... label1_v2, label2_v2...]
        labels = labels.contiguous().view(-1, 1)
        labels_views = torch.cat([labels, labels], dim=0)
        mask = torch.eq(labels_views, labels_views.T).float().to(device)
        
        # Remove self-contrast cases (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Numerical stability trick
        exp_logits = torch.exp(sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0])
        exp_logits = exp_logits * logits_mask
        
        log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive cases
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss

# ==========================================
# DATA & AUGMENTATIONS
# ==========================================
# View Generator: returns two augmented views of the same image
class TwoCropTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]

# Strong augmentations for Contrastive Learning
train_transform_supcon = transforms.Compose([
    transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Standard aug for finetuning
train_transform_standard = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

supcon_dataset = datasets.ImageFolder(TRAIN_DIR, transform=TwoCropTransform(train_transform_supcon))
standard_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform_standard)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

# Note: GLAUCOMA_SUSPECT maps to class 0 natively in ImageFolder.
# We will invert to standard interpretation: 1=GLAUCOMA, 0=NORMAL later during metric calc.

supcon_loader = DataLoader(supcon_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
train_loader = DataLoader(standard_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ==========================================
# ARCHITECTURE
# ==========================================
class YOLOBackboneExtractor(nn.Module):
    def __init__(self, yolo_pt_path):
        super().__init__()
        # Load the raw PyTorch model from Ultralytics checkpoints
        checkpoint = torch.load(yolo_pt_path, map_location=DEVICE, weights_only=False)
        self.yolo = checkpoint['model'].float()
        
        # In YOLOv11s-cls, the backbone+neck feature dimension before the classification head is 1024.
        # The class head is usually the last layer (often index 10).
        # We strip the final linear layer and pooling from the ultralytics head if it exists.
        # model.model is a Sequential of blocks.
        # Often model.model[-1] is the Classify head. Let's just slice it out.
        self.features = nn.Sequential(*list(self.yolo.model.children())[:-1])
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        # We need to manually pass through the sequential blocks
        # because YOLO uses route/shortcut connections internally.
        # Actually YOLO's internal `.forward(x)` supports `embed=True` or `features=True` 
        # in recent Ultralytics but sometimes requires the full graph.
        
        # Safest way to extract features correctly supporting all internal YOLO routes:
        # Pass through the full model but catch the output right before the classifier.
        # Classify module is dict/list index specific. 
        # But wait, YOLOv11 Classification models actually are purely sequential!
        # The routing is mainly in detection models. Classification backbones are sequential 
        # Conv/C2f/v11 blocks.
        
        # Let's extract safely:
        x = self.features(x)
        x = self.gap(x)
        return torch.flatten(x, 1)

class SupConModel(nn.Module):
    """Wrapper that attaches the Projection MLP to the Backbone"""
    def __init__(self, backbone, feat_dim=1024, proj_dim=128):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim)
        )
        
    def forward(self, x):
        feat = self.backbone(x)
        proj = self.head(feat)
        # L2 Normalize
        proj = F.normalize(proj, dim=1)
        return feat, proj

class ClassifierFinetune(nn.Module):
    """Attaches a linear classifier head to a backbone"""
    def __init__(self, backbone, feat_dim=1024, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(feat_dim, num_classes)
        
    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)

def extract_embeddings(model, loader, device):
    """Utility to generate embeddings for TSNE"""
    model.eval()
    embeds = []
    labels_list = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feat = model(images)
            embeds.append(feat.cpu().numpy())
            labels_list.append(labels.numpy())
    return np.vstack(embeds), np.concatenate(labels_list)

def plot_tsne(embeds, labels, title, save_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    proj = tsne.fit_transform(embeds)
    plt.figure(figsize=(8,6))
    colors = ['blue' if l == 1 else 'red' for l in labels] # 1=NORMAL natively, 0=GLAUCOMA natively
    plt.scatter(proj[:,0], proj[:,1], c=colors, alpha=0.6, s=20)
    plt.title(title)
    # Legend manually
    import matplotlib.patches as mpatches
    re_patch = mpatches.Patch(color='red', label='Glaucoma (native 0)')
    bl_patch = mpatches.Patch(color='blue', label='Normal (native 1)')
    plt.legend(handles=[re_patch, bl_patch])
    plt.savefig(save_path, dpi=150)
    plt.close()

# ==========================================
# STAGE 2: SUPCON PRETRAINING
# ==========================================
logger.info("Initializing Backbone from Runs6...")
base_backbone = YOLOBackboneExtractor(YOLO_WEIGHTS).to(DEVICE)
supcon_model = SupConModel(base_backbone).to(DEVICE)

# Run t-sne BEFORE SupCon (using standard Test Loader without 2-crops)
logger.info("Extracting pre-SupCon embeddings for TSNE...")
feat_pre, labels_pre = extract_embeddings(base_backbone, test_loader, DEVICE)
plot_tsne(feat_pre, labels_pre, "t-SNE: YOLOv11s Backbone (Before SupCon)", os.path.join(OUT_DIR, "tsne_pre_supcon.png"))

criterion_supcon = SupConLoss(temperature=temperature)
optimizer_supcon = optim.AdamW(supcon_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_supcon = optim.lr_scheduler.CosineAnnealingLR(optimizer_supcon, T_max=supcon_epochs)

logger.info("Starting Stage 2: SupCon Pretraining")
supcon_model.train()
for epoch in range(1, supcon_epochs + 1):
    total_loss = 0
    for images, labels in supcon_loader:
        # Images is a list of two augmented views: [view1, view2]
        images = torch.cat([images[0], images[1]], dim=0).to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer_supcon.zero_grad()
        
        # Forward pass
        _, proj = supcon_model(images)
        
        # Split views [2B, Dim] -> [B, 2, Dim]
        bsz = labels.shape[0]
        f1, f2 = torch.split(proj, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        # Compute loss
        loss = criterion_supcon(features, labels)
        loss.backward()
        optimizer_supcon.step()
        
        total_loss += loss.item()
        
    scheduler_supcon.step()
    if epoch % 10 == 0 or epoch == 1:
        logger.info(f"SupCon Epoch [{epoch}/{supcon_epochs}], Loss: {total_loss/len(supcon_loader):.4f}")

# Save the contrastively trained backbone
torch.save(supcon_model.backbone.state_dict(), os.path.join(OUT_DIR, "backbone_supcon.pth"))

# Run t-sne AFTER SupCon
logger.info("Extracting post-SupCon embeddings for TSNE...")
supcon_model.eval()
feat_pos, labels_pos = extract_embeddings(supcon_model.backbone, test_loader, DEVICE)
plot_tsne(feat_pos, labels_pos, "t-SNE: YOLOv11s Backbone (After SupCon)", os.path.join(OUT_DIR, "tsne_post_supcon.png"))


# ==========================================
# STAGE 3: CLASSIFIER FINETUNING
# ==========================================
def train_evaluate_finetune(model, train_loader, test_loader, exp_name, lr_backbone, lr_classifier, epochs=40):
    logger.info(f"--- Starting Fine-tuning: {exp_name} ---")
    model = model.to(DEVICE)
    
    # Differential learning rates
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': lr_backbone},
        {'params': model.classifier.parameters(), 'lr': lr_classifier}
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_f1 = 0
    best_recall = 0
    patience = 10
    epochs_no_improve = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Eval
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Native: 0=GLAUCOMA, 1=NORMAL. Sklearn assumes 1=Positive.
        # Invert so 1=GLAUCOMA for metric calculation!
        y_true_inv = 1 - y_true
        y_pred_inv = 1 - y_pred
        
        rec = recall_score(y_true_inv, y_pred_inv, zero_division=0)
        prec = precision_score(y_true_inv, y_pred_inv, zero_division=0)
        f1 = f1_score(y_true_inv, y_pred_inv, zero_division=0)
        
        if rec > best_recall or (rec == best_recall and f1 > best_f1):
            best_recall = rec
            best_f1 = f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{exp_name}_best.pth"))
        else:
            epochs_no_improve += 1
            
        logger.info(f"[{exp_name}] Epoch {epoch}: Loss={train_loss/len(train_loader):.4f} | R={rec:.4f} P={prec:.4f} F1={f1:.4f}")
        if epochs_no_improve >= patience:
            logger.info(f"[{exp_name}] Early stopping triggered at epoch {epoch}")
            break
            
    # Final evaluation on best weights
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, f"{exp_name}_best.pth")))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_inv = 1 - y_true
    y_pred_inv = 1 - y_pred
    
    acc = accuracy_score(y_true_inv, y_pred_inv)
    rec = recall_score(y_true_inv, y_pred_inv, zero_division=0)
    prec = precision_score(y_true_inv, y_pred_inv, zero_division=0)
    f1 = f1_score(y_true_inv, y_pred_inv, zero_division=0)
    
    # cm for True=GLAUCOMA at top
    cm = confusion_matrix(y_true_inv, y_pred_inv, labels=[1, 0]) 
    tn, fp, fn, tp = cm.ravel() 
    # Since labels=[1, 0] inside the matrix:
    # Row 0 = True 1 (GLAUCOMA), Row 1 = True 0 (NORMAL)
    # Col 0 = Pred 1 (GLAUCOMA), Col 1 = Pred 0 (NORMAL)
    # So cm[0,0] = TP, cm[0,1] = FN, cm[1,0] = FP, cm[1,1] = TN.
    TP = cm[0,0]
    FN = cm[0,1]
    FP = cm[1,0]
    TN = cm[1,1]
    
    logger.info(f"FINAL RESULT {exp_name}:")
    logger.info(f"  Accuracy:  {acc:.4f}")
    logger.info(f"  Recall:    {rec:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    logger.info(f"  TP={TP}, FN={FN}, FP={FP}, TN={TN}")
    
    # Save CM Image
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    labels_txt = ["GLAUCOMA", "NORMAL"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels_txt, yticklabels=labels_txt, 
           title=f"Confusion Matrix: {exp_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.savefig(os.path.join(OUT_DIR, f"{exp_name}_cm.png"))
    plt.close()

# ----------------------------------------
# Exp 1: Baseline (No SupCon)
# ----------------------------------------
# Fresh extraction from Runs6 PT
baseline_bb = YOLOBackboneExtractor(YOLO_WEIGHTS)
model_baseline = ClassifierFinetune(baseline_bb)
train_evaluate_finetune(model_baseline, train_loader, test_loader, "baseline_no_supcon", lr_backbone=1e-5, lr_classifier=1e-3)

# ----------------------------------------
# Exp 2: SupCon + Frozen Backbone
# ----------------------------------------
frozen_bb = YOLOBackboneExtractor(YOLO_WEIGHTS)
frozen_bb.load_state_dict(torch.load(os.path.join(OUT_DIR, "backbone_supcon.pth")))
# Freeze
for param in frozen_bb.parameters():
    param.requires_grad = False
model_frozen = ClassifierFinetune(frozen_bb)
train_evaluate_finetune(model_frozen, train_loader, test_loader, "supcon_frozen", lr_backbone=0.0, lr_classifier=1e-3)

# ----------------------------------------
# Exp 3: SupCon + Partial Finetuning
# ----------------------------------------
partial_bb = YOLOBackboneExtractor(YOLO_WEIGHTS)
partial_bb.load_state_dict(torch.load(os.path.join(OUT_DIR, "backbone_supcon.pth")))
# Freeze everything except the last block
# In our Sequential, the last block is partial_bb.features[-1]
for name, param in partial_bb.named_parameters():
    param.requires_grad = False
for param in partial_bb.features[-1].parameters():
    param.requires_grad = True
model_partial = ClassifierFinetune(partial_bb)
train_evaluate_finetune(model_partial, train_loader, test_loader, "supcon_partial", lr_backbone=1e-5, lr_classifier=1e-3)

# ----------------------------------------
# Exp 4: SupCon + Full Finetuning
# ----------------------------------------
full_bb = YOLOBackboneExtractor(YOLO_WEIGHTS)
full_bb.load_state_dict(torch.load(os.path.join(OUT_DIR, "backbone_supcon.pth")))
model_full = ClassifierFinetune(full_bb)
train_evaluate_finetune(model_full, train_loader, test_loader, "supcon_full", lr_backbone=1e-5, lr_classifier=1e-3)

logger.info("All Runs17 Experiments Completed Successfully.")
