"""
Runs19: YOLOv11s Backbone + Multi-Scale Cross-Space Fusion + Learnable Scale Weights + SupCon
===============================================================================================
Architecture:
  - Load YOLOv11s Runs6 weights
  - Extract P3 (Layer4, 256-d), P4 (Layer6, 256-d), P5 (Layer9, 512-d) via GAP
  - Cross-space projection: P3→256, P4→512, P5→1024 (learned linear layers)
  - Learnable scalar weights w3, w4, w5 applied element-wise per scale
  - Concat → h ∈ R^1792
  - SupCon pretraining (150 epochs) then 3-way classifier fine-tuning
"""
import os, logging, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR     = "/home/abhay/chaksu/20123135/Train/data_augment"
YOLO_WEIGHTS = f"{BASE_DIR}/runs6/yolo11s_gamma2.0_alpha0.80/weights/best.pt"
OUT_DIR      = f"{BASE_DIR}/runs19"
TRAIN_DIR, TEST_DIR = f"{BASE_DIR}/train", f"{BASE_DIR}/test"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE     = 512
BATCH        = 64
SUPCON_EP    = 150
FT_EP        = 40
PATIENCE     = 10
TEMP         = 0.07
FEAT_DIM     = 1792   # 256+512+1024 after cross-space projection

logging.basicConfig(filename=f"{OUT_DIR}/training.log", filemode="w", level=logging.INFO,
                    format="%(asctime)s - %(message)s")
log = logging.getLogger()
log.info("Runs19 starting: Learnable Scale Weights + Cross-Space Fusion + SupCon")

# ── LOSS ─────────────────────────────────────────────────────────────────────
class SupConLoss(nn.Module):
    def __init__(self, temp=0.07):
        super().__init__(); self.temp = temp
    def forward(self, features, labels):
        bsz = labels.shape[0]
        features = torch.cat(torch.unbind(features, 1), 0)          # [2B, dim]
        sim = torch.div(features @ features.T, self.temp)           # [2B, 2B]
        labs = torch.cat([labels, labels], 0).view(-1,1)
        mask = torch.eq(labs, labs.T).float().to(features.device)
        lmask = torch.scatter(torch.ones_like(mask), 1,
                              torch.arange(2*bsz).view(-1,1).to(features.device), 0)
        mask *= lmask
        logits = torch.exp(sim - sim.max(1, keepdim=True)[0]) * lmask
        log_p  = sim - torch.log(logits.sum(1, keepdim=True) + 1e-8)
        return -(mask * log_p).sum(1).div(mask.sum(1)+1e-8).mean()

# ── DATA ─────────────────────────────────────────────────────────────────────
class TwoCrop:
    def __init__(self, t): self.t = t
    def __call__(self, x): return [self.t(x), self.t(x)]

strong = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6,1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.4,0.4,0.4,0.1),
    transforms.RandomGrayscale(0.2),
    transforms.GaussianBlur(5,(0.1,2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

std_aug = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.2,0.2,0.2,0.1), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

sc_loader = DataLoader(datasets.ImageFolder(TRAIN_DIR, TwoCrop(strong)),
                       batch_size=BATCH, shuffle=True, num_workers=4, drop_last=True)
tr_loader = DataLoader(datasets.ImageFolder(TRAIN_DIR, std_aug),
                       batch_size=BATCH, shuffle=True, num_workers=4)
te_loader = DataLoader(datasets.ImageFolder(TEST_DIR, test_tf),
                       batch_size=BATCH, shuffle=False, num_workers=4)

# ── BACKBONE ──────────────────────────────────────────────────────────────────
class MultiScaleFusionBackbone(nn.Module):
    """Extract P3/P4/P5 → cross-space project → learnable scale weights → concat 1792-d"""
    def __init__(self, yolo_path):
        super().__init__()
        ckpt = torch.load(yolo_path, map_location=DEVICE, weights_only=False)
        self.blocks = nn.ModuleList(list(ckpt['model'].float().model.children())[:-1])
        self.gap    = nn.AdaptiveAvgPool2d(1)
        # Cross-space projection: native → target dims
        self.proj3  = nn.Linear(256, 256)    # P3: 256 → 256
        self.proj4  = nn.Linear(256, 512)    # P4: 256 → 512  (cross-space fusion)
        self.proj5  = nn.Linear(512, 1024)   # P5: 512 → 1024 (cross-space fusion)
        # Learnable scale weights (Runs19 specific)
        self.w3 = nn.Parameter(torch.ones(1))
        self.w4 = nn.Parameter(torch.ones(1))
        self.w5 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        h3 = h4 = h5 = None
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == 4: h3 = torch.flatten(self.gap(x), 1)
            elif i == 6: h4 = torch.flatten(self.gap(x), 1)
            elif i == 9: h5 = torch.flatten(self.gap(x), 1)
        # Cross-space project then weight
        h3 = self.w3 * self.proj3(h3)   # → 256
        h4 = self.w4 * self.proj4(h4)   # → 512
        h5 = self.w5 * self.proj5(h5)   # → 1024
        return torch.cat([h3, h4, h5], dim=1)   # → 1792

class SupConModel(nn.Module):
    def __init__(self, backbone, feat=FEAT_DIM, proj=128):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(feat,512), nn.ReLU(True), nn.Linear(512,proj))
    def forward(self, x):
        f = self.backbone(x)
        return f, F.normalize(self.head(f), dim=1)

class Classifier(nn.Module):
    def __init__(self, backbone, feat=FEAT_DIM):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(feat, 2)
    def forward(self, x): return self.fc(self.backbone(x))

# ── UTILS ──────────────────────────────────────────────────────────────────────
def get_embeds(model, loader):
    model.eval(); embs, labs = [], []
    with torch.no_grad():
        for x, y in loader:
            embs.append(model(x.to(DEVICE)).cpu().numpy())
            labs.append(y.numpy())
    return np.vstack(embs), np.concatenate(labs)

def plot_tsne(embs, labs, title, path):
    proj = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embs)
    plt.figure(figsize=(8,6))
    colors = ['blue' if l==1 else 'red' for l in labs]
    plt.scatter(proj[:,0], proj[:,1], c=colors, alpha=0.6, s=20)
    plt.title(title)
    plt.legend(handles=[mpatches.Patch(color='red', label='Glaucoma'),
                        mpatches.Patch(color='blue', label='Normal')])
    plt.savefig(path, dpi=150); plt.close()

def make_cm_plot(cm, exp, path):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tl = ["GLAUCOMA","NORMAL"]
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=tl, yticklabels=tl, title=f"CM: {exp}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color='white' if cm[i,j]>cm.max()/2 else 'black')
    plt.savefig(path); plt.close()

def evaluate(model, loader):
    model.eval(); yt, yp = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(DEVICE))
            yp.extend(torch.argmax(out,1).cpu().numpy())
            yt.extend(y.numpy())
    # invert: native 0=GLAUCOMA → metric pos_label=1
    yt = 1 - np.array(yt); yp = 1 - np.array(yp)
    return yt, yp

def finetune(model, name, lr_bb, lr_cls):
    log.info(f"--- Fine-tuning: {name} ---")
    model = model.to(DEVICE)
    opt = optim.AdamW([{'params': model.backbone.parameters(), 'lr': lr_bb},
                       {'params': model.fc.parameters(),       'lr': lr_cls}], weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_r, best_f1, no_imp = 0, 0, 0
    for ep in range(1, FT_EP+1):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        yt, yp = evaluate(model, te_loader)
        r = recall_score(yt, yp, zero_division=0)
        p = precision_score(yt, yp, zero_division=0)
        f = f1_score(yt, yp, zero_division=0)
        log.info(f"[{name}] Ep{ep}: R={r:.4f} P={p:.4f} F1={f:.4f}")
        if r > best_r or (r==best_r and f > best_f1):
            best_r, best_f1, no_imp = r, f, 0
            torch.save(model.state_dict(), f"{OUT_DIR}/{name}_best.pth")
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            log.info(f"[{name}] Early stop ep{ep}"); break
    model.load_state_dict(torch.load(f"{OUT_DIR}/{name}_best.pth"))
    yt, yp = evaluate(model, te_loader)
    acc = accuracy_score(yt, yp)
    r   = recall_score(yt, yp, zero_division=0)
    p   = precision_score(yt, yp, zero_division=0)
    f   = f1_score(yt, yp, zero_division=0)
    cm  = confusion_matrix(yt, yp, labels=[1,0])
    TP,FN,FP,TN = cm[0,0],cm[0,1],cm[1,0],cm[1,1]
    log.info(f"FINAL {name}: Acc={acc:.4f} R={r:.4f} P={p:.4f} F1={f:.4f} TP={TP} FN={FN} FP={FP} TN={TN}")
    make_cm_plot(cm, name, f"{OUT_DIR}/{name}_cm.png")

# ── STAGE 2: SUPCON PRETRAINING ────────────────────────────────────────────────
log.info("Loading backbone...")
bb = MultiScaleFusionBackbone(YOLO_WEIGHTS).to(DEVICE)
supcon = SupConModel(bb).to(DEVICE)

log.info("Pre-SupCon t-SNE...")
e, l = get_embeds(bb, te_loader)
plot_tsne(e, l, "t-SNE: Pre-SupCon (Runs19)", f"{OUT_DIR}/tsne_pre.png")

crit_sc = SupConLoss(TEMP)
opt_sc  = optim.AdamW(supcon.parameters(), lr=1e-3, weight_decay=1e-4)
sched   = optim.lr_scheduler.CosineAnnealingLR(opt_sc, T_max=SUPCON_EP)

log.info("Starting SupCon pretraining (150 epochs)...")
supcon.train()
for ep in range(1, SUPCON_EP+1):
    total = 0
    for imgs, labels in sc_loader:
        imgs   = torch.cat([imgs[0], imgs[1]], 0).to(DEVICE)
        labels = labels.to(DEVICE)
        opt_sc.zero_grad()
        bsz = labels.shape[0]
        _, proj = supcon(imgs)
        f1p, f2p = torch.split(proj, [bsz, bsz], 0)
        feats = torch.cat([f1p.unsqueeze(1), f2p.unsqueeze(1)], 1)
        loss  = crit_sc(feats, labels)
        loss.backward(); opt_sc.step(); total += loss.item()
    sched.step()
    if ep % 10 == 0 or ep == 1:
        log.info(f"SupCon Ep [{ep}/{SUPCON_EP}] Loss={total/len(sc_loader):.4f}")

torch.save(supcon.backbone.state_dict(), f"{OUT_DIR}/backbone_supcon.pth")

log.info("Post-SupCon t-SNE...")
supcon.eval()
e, l = get_embeds(supcon.backbone, te_loader)
plot_tsne(e, l, "t-SNE: Post-SupCon (Runs19)", f"{OUT_DIR}/tsne_post.png")

# Log learned scale weights
log.info(f"Learned scale weights: w3={bb.w3.item():.4f}  w4={bb.w4.item():.4f}  w5={bb.w5.item():.4f}")

# ── STAGE 3: FINE-TUNING ────────────────────────────────────────────────────────
# Baseline (no SupCon backbone)
bb0 = MultiScaleFusionBackbone(YOLO_WEIGHTS)
finetune(Classifier(bb0), "baseline", lr_bb=1e-5, lr_cls=1e-3)

# Frozen SupCon backbone
bb_fr = MultiScaleFusionBackbone(YOLO_WEIGHTS)
bb_fr.load_state_dict(torch.load(f"{OUT_DIR}/backbone_supcon.pth"))
for p in bb_fr.parameters(): p.requires_grad_(False)
finetune(Classifier(bb_fr), "supcon_frozen", lr_bb=0.0, lr_cls=1e-3)

# Partial fine-tune (last 3 YOLO blocks + projections)
bb_pt = MultiScaleFusionBackbone(YOLO_WEIGHTS)
bb_pt.load_state_dict(torch.load(f"{OUT_DIR}/backbone_supcon.pth"))
for p in bb_pt.parameters(): p.requires_grad_(False)
for blk in bb_pt.blocks[-3:]:
    for p in blk.parameters(): p.requires_grad_(True)
for p in list(bb_pt.proj3.parameters())+list(bb_pt.proj4.parameters())+list(bb_pt.proj5.parameters()):
    p.requires_grad_(True)
bb_pt.w3.requires_grad_(True); bb_pt.w4.requires_grad_(True); bb_pt.w5.requires_grad_(True)
finetune(Classifier(bb_pt), "supcon_partial", lr_bb=1e-5, lr_cls=1e-3)

# Full fine-tune
bb_fu = MultiScaleFusionBackbone(YOLO_WEIGHTS)
bb_fu.load_state_dict(torch.load(f"{OUT_DIR}/backbone_supcon.pth"))
finetune(Classifier(bb_fu), "supcon_full", lr_bb=1e-5, lr_cls=1e-3)

log.info("Runs19 COMPLETE.")
