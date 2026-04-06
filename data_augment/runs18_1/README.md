# Runs18.1 — L2-Normalized Multi-Scale + SupCon @ 1024×1024

**Fix for Runs18**: L2-normalize P3, P4, P5 independently before concatenation to eliminate feature magnitude imbalance.

## Key Change vs Runs18
```python
p3 = F.normalize(p3, dim=1)  # Previously raw GAP output
p4 = F.normalize(p4, dim=1)
p5 = F.normalize(p5, dim=1)
h = concat([p3, p4, p5]) ∈ R^1024
```

## Config
| Parameter | Value |
|-----------|-------|
| Image Size | **1024×1024** |
| Feature Dim | 1024 (256+256+512, L2 normalized) |
| Batch | 32 |
| SupCon Epochs | 150 |
| FT Epochs | 200, patience=100 |
| LR | backbone=1e-5, head=1e-3 |
| Projection | Linear(1024→512)→ReLU→Linear(512→128) |
| Classifier | Linear(1024→2) |

## Test Set Results
| Experiment | Accuracy | Precision | Recall | F1 | TP | FN | FP | TN |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline | TBD | TBD | TBD | TBD | - | - | - | - |
| SupCon Frozen | TBD | TBD | TBD | TBD | - | - | - | - |
| SupCon Partial | TBD | TBD | TBD | TBD | - | - | - | - |
| SupCon Full | TBD | TBD | TBD | TBD | - | - | - | - |
