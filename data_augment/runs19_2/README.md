# Runs19.2 — Normalized Multi-Scale + Learnable Weights (biased init) + SupCon @ 1024×1024

**Improvement on Runs19**: Combine L2 normalization with biased learnable scale weights to encode domain prior (P3 fine-grained > P5 global).

## Key Change vs Runs19
- L2 normalize before applying weights (stability)
- **Biased initialization**: w3=1.2, w4=1.0, w5=0.8

```python
h = [w3*norm(P3) || w4*norm(P4) || w5*norm(P5)] ∈ R^1024
```

## Config
| Parameter | Value |
|-----------|-------|
| Image Size | **1024×1024** |
| Feature Dim | 1024 |
| Init weights | w3=1.2, w4=1.0, w5=0.8 |
| Batch | 32 |
| SupCon Epochs | 150 |
| FT Epochs | 200, patience=100 |

## Test Set Results
| Experiment | Accuracy | Precision | Recall | F1 | TP | FN | FP | TN |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline | TBD | TBD | TBD | TBD | - | - | - | - |
| SupCon Frozen | TBD | TBD | TBD | TBD | - | - | - | - |
| SupCon Partial | TBD | TBD | TBD | TBD | - | - | - | - |
| SupCon Full | TBD | TBD | TBD | TBD | - | - | - | - |
