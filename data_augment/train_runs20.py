"""
Runs20: YOLOv11s Backbone + Multi-Scale Cross-Space Fusion + SE Channel Attention + SupCon
============================================================================================
Architecture:
  - Extract P3 (256-d), P4 (256-d), P5 (512-d) via GAP
  - Cross-space project: P3→256, P4→512, P5→1024 → concat h ∈ R^1792
  - SE block: h' = σ(W2·ReLU(W1·h)) ⊙ h  (squeeze=1792/4=448)
  - SupCon pretraining on h', then 3-way classifier fine-tuning
"""
import os, logging, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR     = "/home/abhay/chaksu/20123135/Train/data_augment"
YOLO_WEIGHTS = f"{BASE_DIR}/runs6/yolo11s_gamma2.0_alpha0.80/weights/best.pt"
OUT_DIR      = f"{BASE_DIR}/runs20"
TRAIN_DIR, TEST_DIR = f"{BASE_DIR}/train", f"{BASE_DIR}/test"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE  = 512; BATCH = 64; SUPCON_EP = 150; FT_EP = 40; PATIENCE = 10; TEMP = 0.07
FEAT_DIM  = 1792

logging.basicConfig(filename=f"{OUT_DIR}/training.log", filemode="w", level=logging.INFO,
                    format="%(asctime)s - %(message)s")
log = logging.getLogger()
log.info("Runs20: SE Channel Attention + Cross-Space Fusion + SupCon")

class SupConLoss(nn.Module):
    def __init__(self, temp=0.07):
        super().__init__(); self.temp = temp
    def forward(self, features, labels):
        bsz = labels.shape[0]
        features = torch.cat(torch.unbind(features, 1), 0)
        sim  = torch.div(features @ features.T, self.temp)
        labs = torch.cat([labels, labels], 0).view(-1,1)
        mask = torch.eq(labs, labs.T).float().to(features.device)
        lm   = torch.scatter(torch.ones_like(mask), 1,
                             torch.arange(2*bsz).view(-1,1).to(features.device), 0)
        mask *= lm
        exp  = torch.exp(sim - sim.max(1, keepdim=True)[0]) * lm
        lp   = sim - torch.log(exp.sum(1, keepdim=True) + 1e-8)
        return -(mask * lp).sum(1).div(mask.sum(1)+1e-8).mean()

class TwoCrop:
    def __init__(self, t): self.t = t
    def __call__(self, x): return [self.t(x), self.t(x)]

strong = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE,(0.6,1.0)), transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.4,0.4,0.4,0.1), transforms.RandomGrayscale(0.2),
    transforms.GaussianBlur(5,(0.1,2.0)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
std_aug = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.2,0.2,0.2,0.1), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

sc_loader = DataLoader(datasets.ImageFolder(TRAIN_DIR,TwoCrop(strong)),
                       batch_size=BATCH,shuffle=True,num_workers=4,drop_last=True)
tr_loader = DataLoader(datasets.ImageFolder(TRAIN_DIR,std_aug),
                       batch_size=BATCH,shuffle=True,num_workers=4)
te_loader = DataLoader(datasets.ImageFolder(TEST_DIR,test_tf),
                       batch_size=BATCH,shuffle=False,num_workers=4)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention"""
    def __init__(self, dim, r=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(dim, dim//r), nn.ReLU(True),
            nn.Linear(dim//r, dim), nn.Sigmoid())
    def forward(self, x): return x * self.se(x)

class MultiScaleSEBackbone(nn.Module):
    def __init__(self, yolo_path):
        super().__init__()
        ckpt = torch.load(yolo_path, map_location=DEVICE, weights_only=False)
        self.blocks = nn.ModuleList(list(ckpt['model'].float().model.children())[:-1])
        self.gap    = nn.AdaptiveAvgPool2d(1)
        # Cross-space projections
        self.proj3 = nn.Linear(256,  256)
        self.proj4 = nn.Linear(256,  512)
        self.proj5 = nn.Linear(512, 1024)
        # SE block on 1792-d concat
        self.se    = SEBlock(FEAT_DIM, r=4)   # 1792/4 = 448

    def forward(self, x):
        h3 = h4 = h5 = None
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == 4:  h3 = torch.flatten(self.gap(x), 1)
            elif i == 6: h4 = torch.flatten(self.gap(x), 1)
            elif i == 9: h5 = torch.flatten(self.gap(x), 1)
        h = torch.cat([self.proj3(h3), self.proj4(h4), self.proj5(h5)], 1)  # 1792
        return self.se(h)   # channel-attended h'

class SupConModel(nn.Module):
    def __init__(self, bb, feat=FEAT_DIM, proj=128):
        super().__init__()
        self.backbone = bb
        self.head = nn.Sequential(nn.Linear(feat,512), nn.ReLU(True), nn.Linear(512,proj))
    def forward(self, x):
        f = self.backbone(x)
        return f, F.normalize(self.head(f), dim=1)

class Classifier(nn.Module):
    def __init__(self, bb, feat=FEAT_DIM):
        super().__init__()
        self.backbone = bb; self.fc = nn.Linear(feat, 2)
    def forward(self, x): return self.fc(self.backbone(x))

def get_embeds(m, ldr):
    m.eval(); embs, labs = [], []
    with torch.no_grad():
        for x,y in ldr:
            embs.append(m(x.to(DEVICE)).cpu().numpy()); labs.append(y.numpy())
    return np.vstack(embs), np.concatenate(labs)

def plot_tsne(embs, labs, title, path):
    p = TSNE(n_components=2,perplexity=30,random_state=42).fit_transform(embs)
    plt.figure(figsize=(8,6))
    plt.scatter(p[:,0],p[:,1],c=['blue' if l==1 else 'red' for l in labs],alpha=0.6,s=20)
    plt.title(title)
    plt.legend(handles=[mpatches.Patch(color='red',label='Glaucoma'),
                        mpatches.Patch(color='blue',label='Normal')])
    plt.savefig(path,dpi=150); plt.close()

def make_cm(cm, name, path):
    fig,ax = plt.subplots()
    im = ax.imshow(cm, cmap=plt.cm.Blues); ax.figure.colorbar(im,ax=ax)
    tl=["GLAUCOMA","NORMAL"]
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=tl,yticklabels=tl,title=f"CM:{name}")
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',
                    color='white' if cm[i,j]>cm.max()/2 else 'black')
    plt.savefig(path); plt.close()

def evaluate(m, ldr):
    m.eval(); yt,yp=[],[]
    with torch.no_grad():
        for x,y in ldr:
            yp.extend(torch.argmax(m(x.to(DEVICE)),1).cpu().numpy()); yt.extend(y.numpy())
    return 1-np.array(yt), 1-np.array(yp)

def finetune(model, name, lr_bb, lr_cls):
    log.info(f"--- Fine-tuning: {name} ---")
    model = model.to(DEVICE)
    opt  = optim.AdamW([{'params':model.backbone.parameters(),'lr':lr_bb},
                        {'params':model.fc.parameters(),'lr':lr_cls}], weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_r,best_f1,no_imp = 0,0,0
    for ep in range(1,FT_EP+1):
        model.train()
        for x,y in tr_loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad(); crit(model(x),y).backward(); opt.step()
        yt,yp = evaluate(model,te_loader)
        r=recall_score(yt,yp,zero_division=0); p=precision_score(yt,yp,zero_division=0)
        f=f1_score(yt,yp,zero_division=0)
        log.info(f"[{name}] Ep{ep}: R={r:.4f} P={p:.4f} F1={f:.4f}")
        if r>best_r or (r==best_r and f>best_f1):
            best_r,best_f1,no_imp=r,f,0
            torch.save(model.state_dict(),f"{OUT_DIR}/{name}_best.pth")
        else:
            no_imp+=1
        if no_imp>=PATIENCE: log.info(f"[{name}] Early stop ep{ep}"); break
    model.load_state_dict(torch.load(f"{OUT_DIR}/{name}_best.pth"))
    yt,yp = evaluate(model,te_loader)
    acc=accuracy_score(yt,yp); r=recall_score(yt,yp,zero_division=0)
    p=precision_score(yt,yp,zero_division=0); f=f1_score(yt,yp,zero_division=0)
    cm=confusion_matrix(yt,yp,labels=[1,0])
    TP,FN,FP,TN=cm[0,0],cm[0,1],cm[1,0],cm[1,1]
    log.info(f"FINAL {name}: Acc={acc:.4f} R={r:.4f} P={p:.4f} F1={f:.4f} TP={TP} FN={FN} FP={FP} TN={TN}")
    make_cm(cm, name, f"{OUT_DIR}/{name}_cm.png")

# ── SUPCON PRETRAINING ─────────────────────────────────────────────────────────
log.info("Loading SE backbone..."); bb=MultiScaleSEBackbone(YOLO_WEIGHTS).to(DEVICE)
supcon=SupConModel(bb).to(DEVICE)
log.info("Pre-SupCon t-SNE..."); e,l=get_embeds(bb,te_loader)
plot_tsne(e,l,"t-SNE: Pre-SupCon (Runs20)",f"{OUT_DIR}/tsne_pre.png")

crit_sc=SupConLoss(TEMP)
opt_sc=optim.AdamW(supcon.parameters(),lr=1e-3,weight_decay=1e-4)
sched=optim.lr_scheduler.CosineAnnealingLR(opt_sc,T_max=SUPCON_EP)

log.info("SupCon pretraining (150 epochs)...")
supcon.train()
for ep in range(1,SUPCON_EP+1):
    tot=0
    for imgs,labs in sc_loader:
        imgs=torch.cat([imgs[0],imgs[1]],0).to(DEVICE); labs=labs.to(DEVICE)
        opt_sc.zero_grad()
        bsz=labs.shape[0]; _,proj=supcon(imgs)
        f1p,f2p=torch.split(proj,[bsz,bsz],0)
        feats=torch.cat([f1p.unsqueeze(1),f2p.unsqueeze(1)],1)
        loss=crit_sc(feats,labs); loss.backward(); opt_sc.step(); tot+=loss.item()
    sched.step()
    if ep%10==0 or ep==1: log.info(f"SupCon Ep[{ep}/{SUPCON_EP}] Loss={tot/len(sc_loader):.4f}")

torch.save(supcon.backbone.state_dict(),f"{OUT_DIR}/backbone_supcon.pth")
log.info("Post-SupCon t-SNE..."); supcon.eval(); e,l=get_embeds(supcon.backbone,te_loader)
plot_tsne(e,l,"t-SNE: Post-SupCon (Runs20)",f"{OUT_DIR}/tsne_post.png")

# ── FINE-TUNING ─────────────────────────────────────────────────────────────────
bb0=MultiScaleSEBackbone(YOLO_WEIGHTS); finetune(Classifier(bb0),"baseline",1e-5,1e-3)

bb_fr=MultiScaleSEBackbone(YOLO_WEIGHTS)
bb_fr.load_state_dict(torch.load(f"{OUT_DIR}/backbone_supcon.pth"))
for p in bb_fr.parameters(): p.requires_grad_(False)
finetune(Classifier(bb_fr),"supcon_frozen",0.0,1e-3)

bb_pt=MultiScaleSEBackbone(YOLO_WEIGHTS)
bb_pt.load_state_dict(torch.load(f"{OUT_DIR}/backbone_supcon.pth"))
for p in bb_pt.parameters(): p.requires_grad_(False)
for blk in bb_pt.blocks[-3:]:
    for p in blk.parameters(): p.requires_grad_(True)
for p in list(bb_pt.proj3.parameters())+list(bb_pt.proj4.parameters())+list(bb_pt.proj5.parameters())+list(bb_pt.se.parameters()):
    p.requires_grad_(True)
finetune(Classifier(bb_pt),"supcon_partial",1e-5,1e-3)

bb_fu=MultiScaleSEBackbone(YOLO_WEIGHTS)
bb_fu.load_state_dict(torch.load(f"{OUT_DIR}/backbone_supcon.pth"))
finetune(Classifier(bb_fu),"supcon_full",1e-5,1e-3)

log.info("Runs20 COMPLETE.")
