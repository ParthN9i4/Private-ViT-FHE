# BOLT (2023)

> Xu et al., "BOLT: Privacy-Preserving, Accurate and Efficient Inference for Transformers", IEEE S&P 2023

## Why This is the Main Target

BOLT is the closest paper to our goal: **private ViT inference without bootstrapping**. It introduces BoBa-ViT — a ViT variant co-designed with FHE constraints.

## Key Ideas

### 1. Bootstrapping-Free Design

Standard FHE transformers need bootstrapping because their depth exceeds the noise budget. BOLT avoids this by:
- Reducing depth of each component
- Co-training with FHE constraints from scratch

### 2. Linear Attention (No Softmax)

Standard softmax: `softmax(QK^T/√d) · V`

BOLT replaces this with a linear variant:
```
Attention(Q, K, V) = (Q · K^T) · V / n_tokens
```
No `exp` → no approximation needed → saves ~5 multiplicative levels.

**Trade-off**: Linear attention loses some long-range modeling capability.

### 3. No LayerNorm

Standard LayerNorm requires `1/sqrt(var + eps)` — a polynomial approximation that costs 2-3 levels.

BOLT replaces with a learnable scalar `γ` applied per channel:
```
Norm(x) = γ ⊙ x   (element-wise scale)
```
This is **free** in FHE (scalar multiply = 0 levels).

**Trade-off**: Slightly lower accuracy; training is more unstable.

### 4. Degree-4 GELU

GELU approximated as:
```
f(x) ≈ 0.5x + 0.1972x³ + 0.0012x⁴
```
Depth: `ceil(log2(4)) = 2` levels.

Compare to degree-27 GELU (Iron): 5 levels.

## Architecture: BoBa-ViT

```
Input: 32x32 image (CIFAR-10) → 4x4 patches → 64 tokens × 64 dim

For each of 6 transformer blocks:
  - Scalar norm (free)
  - Linear attention: Q, K, V projections (1 level) + QK^T·V (2 levels)
  - Output projection (1 level)
  - Scalar norm (free)
  - FC1 (1 level) + degree-4 GELU (2 levels) + FC2 (1 level)
  Block subtotal: 8 levels

6 blocks: 48 levels
Patch embed: 1 level
Classifier: 1 level
Total: 50 levels → fits in n=2^15 (128-bit security)
```

## Results (CIFAR-10)

| Model | Accuracy | Depth | Inference Time |
|-------|----------|-------|----------------|
| ViT-S (plaintext) | 92.1% | N/A | ~50ms |
| BoBa-ViT (BOLT) | 88.3% | 50 | 6.7s |
| CryptoNets-style | 75% | 2 | fast |

## Implementation Plan

1. Train plaintext BoBa-ViT on CIFAR-10 — establish accuracy baseline
2. Profile multiplicative depth using `depth_counter.py`
3. Encode and encrypt a single image patch sequence
4. Run FHE forward pass layer by layer
5. Benchmark and compare to plaintext

## Open Research Questions (from this paper)

1. Can we improve accuracy with a better linear attention variant?
2. Does batch normalization (instead of scalar norm) help with training stability?
3. How does performance scale from CIFAR-10 to Tiny-ImageNet?
