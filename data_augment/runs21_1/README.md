# Runs21.1 — Normalized Multi-Scale + Sigmoid Scale Attention + SupCon @ 1024×1024

**Fix for Runs21 softmax constraint**: Replace softmax (sum=1) with independent sigmoid gates — each scale can be independently up/down-weighted.

## Key Changes vs Runs21
- Sigmoid instead of softmax: `a_i = sigmoid(s_i)` — scales independent
- Init: s3=s4=s5=0 → sigmoid(0)=0.5 (equal start)
- `h = a3*P3' + a4*P4' + a5*P5'` (weighted sum, not constrained)

## Config
| Parameter | Value |
|-----------|-------|
| Image Size | **1024×1024** |
| Feature Dim | 512 (all scales projected to 512-d) |
| Gates | a3=sigmoid(s3), a4=sigmoid(s4), a5=sigmoid(s5) |
| Batch | 32 |
| SupCon Epochs | 150 |
| Projection | Linear(512→256)→ReLU→Linear(256→128) |
| Classifier | Linear(512→2) |

## Test Set Results
| Experiment | Accuracy | Precision | Recall | F1 | TP | FN | FP | TN |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline | TBD | TBD | TBD | TBD | - | - | - | - |
| SupCon Frozen | TBD | TBD | TBD | TBD | - | - | - | - |
| SupCon Partial | TBD | TBD | TBD | TBD | - | - | - | - |
| SupCon Full | TBD | TBD | TBD | TBD | - | - | - | - |
