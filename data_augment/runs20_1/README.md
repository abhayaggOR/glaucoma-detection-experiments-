# Runs20.1 — Normalized Multi-Scale + Lightweight SE Block (residual) + SupCon @ 1024×1024

**Fix for Runs20 overfitting**: Lightweight SE (squeeze=112 vs 448), Dropout(0.3), residual connection.

## Key Changes vs Runs20
- Smaller squeeze ratio (r=16 → dim 64 instead of r=4/448)
- Dropout(0.3) inside SE for regularization
- **Residual**: `h_out = h + h * SE(h)` (prevents over-suppression)

## Config
| Parameter | Value |
|-----------|-------|
| Image Size | **1024×1024** |
| Feature Dim | 1024 |
| SE squeeze | 112 |
| SE dropout | 0.3 |
| Residual | Yes |
| Batch | 32 |
| SupCon Epochs | 125 |
| FT Epochs | 200, patience=100 |

## Test Set Results
| Experiment | Accuracy | Precision | Recall | F1 | TP | FN | FP | TN |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline | TBD | TBD | TBD | TBD | - | - | - | - |
| SupCon Frozen | TBD | TBD | TBD | TBD | - | - | - | - |
| SupCon Partial | TBD | TBD | TBD | TBD | - | - | - | - |
| SupCon Full | TBD | TBD | TBD | TBD | - | - | - | - |
