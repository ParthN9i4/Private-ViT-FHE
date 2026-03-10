# PolyTransformer (IBM, 2022)

> "Towards Practical Homomorphic Evaluation of Transformer Inference" — IBM Research

## Why This Matters

PolyTransformer is a systematic treatment of **replacing every non-polynomial operation in a transformer** while preserving accuracy. It introduces the **range-loss training trick** to tighten the input range of activations, making low-degree polynomial approximations more accurate.

## Key Ideas

### 1. Polynomial Softmax (Power-Softmax variant)

Standard softmax has `exp` — transcendental, needs high-degree poly.

PolyTransformer replaces softmax with:
```
PolyAttn(Q, K) = (Q @ K.T) ^ p  (element-wise power)
normalized per row to sum to 1
```

For `p=2`: `a_ij = (q_i · k_j)^2 / Σ_j (q_i · k_j)^2`

Depth: 1 level for the square.

**Range-loss** tightens QK.T values so the quadratic approximation is accurate:
```
L_range = λ * mean(max(0, |QK.T| - R)^2)
```
This penalizes attention logits outside `[-R, R]`, so the polynomial covers the full range.

### 2. Polynomial GELU / FFN

GELU replaced with a degree-2 or degree-4 polynomial (depending on accuracy target):
```
PolyGELU_2(x) = 0.125 x^2 + 0.5 x + 0.25   (degree 2, depth 1)
PolyGELU_4(x) = 0.5 x + 0.1972 x^3 + ...   (degree 4, depth 2)
```

With range-loss training, degree-2 often suffices where degree-4 was previously required.

### 3. LinearNorm (remove LayerNorm)

LayerNorm requires `1/sqrt(var + ε)` — expensive in FHE (2-3 levels).

PolyTransformer replaces with a **learnable affine scaling** per channel:
```
LinearNorm(x) = γ ⊙ x + β
```
(Same as BatchNorm with fixed stats, or RMS Norm without the division.)

Depth: 0 levels (multiply by a constant is free).

### 4. Full Depth Budget

```
Per block:
  LinearNorm:    0 levels
  PolyAttn (p=2): 2 levels  (Q@K.T = 1, square = 1)
  LinearNorm:    0 levels
  PolyGELU-2:    1 level
  Block total:   3 levels   ← much lower than BOLT's 8

12 blocks:       36 levels  ← fits in n=2^15 comfortably
```

Compare to BOLT (8 levels/block): PolyTransformer achieves similar depth with better accuracy by using range-loss instead of more restrictive architectural changes.

## Range-Loss Intuition

Without range-loss, attention logits QK.T can range widely (e.g., [-20, 20]).
A degree-2 poly approximating softmax over `[-20, 20]` has high error.

With range-loss, training constrains QK.T to `[-R, R]` (e.g., R=5).
A degree-2 poly over `[-5, 5]` is highly accurate.

**Key insight**: constrain the input range during training rather than increasing polynomial degree.

## Results

| Model | Dataset | Standard | + PolyTransformer | Depth |
|-------|---------|----------|-------------------|-------|
| BERT-Base | GLUE | 84.6 | 82.1 | 36 |
| ViT-B/16 | ImageNet | 81.8% | 79.3% | 36 |

## Implementation

See [`implementation.py`](implementation.py) for:
- `RangeLoss` — penalty on out-of-range attention logits
- `PolyAttention` — quadratic attention with optional normalization
- `LinearNorm` — learnable affine scaling, no division
- `PolyFFN` — polynomial GELU FFN
- `PolyTransformerBlock` — full FHE-friendly block

## Open Questions

1. What is the optimal `R` for CIFAR-10 vs. ImageNet?
2. Can range-loss replace the need for degree-4 GELU entirely?
3. Does LinearNorm + range-loss match ScalarNorm (BOLT) in training stability?
