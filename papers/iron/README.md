# Iron (NeurIPS 2022)

> Hao et al., "Iron: Private Inference on Transformers", NeurIPS 2022

## Why This Matters

Iron is the **first private inference system for BERT-scale transformers**. It shows that full transformer inference (attention + GELU + LayerNorm) is achievable under CKKS, establishing the techniques that BOLT and later work refine for ViTs.

## Key Ideas

### 1. Softmax Approximation via Domain Decomposition

Standard softmax: `softmax(x_i) = exp(x_i) / Σ exp(x_j)`

Problem: `exp` is transcendental — requires very high degree polynomial.

Iron's solution — domain decomposition:
1. Shift inputs: `x' = x - max(x)` (now x' ∈ [-M, 0])
2. Decompose domain into subintervals
3. Use low-degree polynomial on each subinterval
4. Stitch together using indicator functions

This keeps degree manageable (~15-27) while covering the needed range.

### 2. GELU Approximation (Degree-27 Minimax)

GELU: `f(x) = x · Φ(x)` where `Φ` is the CDF of the standard normal.

Iron uses a degree-27 minimax polynomial approximation:
- Fit over `x ∈ [-10, 10]`
- Uses Chebyshev nodes for stability
- Depth cost: `ceil(log2(27)) = 5` levels

### 3. LayerNorm Approximation

LayerNorm: `(x - μ) / sqrt(σ² + ε) · γ + β`

Iron approximates `1/sqrt(·)` using:
- Initial estimate via lookup
- Newton-Raphson refinement steps
- Costs ~2-3 multiplicative levels

### 4. Attention Ciphertext Packing

Multiple attention heads packed into a single ciphertext using SIMD slots:
- All heads computed in parallel
- Rotation operations for head-wise operations
- Reduces ciphertext count by number of heads

## Architecture (BERT-Base)

```
Input: sequence of 128 tokens × 768 dim

For each of 12 transformer blocks:
  LayerNorm: ~2 levels
  QKV projections: 1 level
  Scaled dot-product + softmax approx: ~5 levels
  Output projection: 1 level
  LayerNorm: ~2 levels
  FC1 + GELU approx: 1 + 5 = 6 levels
  FC2: 1 level
  Block subtotal: ~18 levels

12 blocks: ~216 levels
→ Requires bootstrapping at multiple points
```

## Performance

| Task | Latency | Accuracy |
|------|---------|----------|
| BERT-base SST-2 | 14.4s | ~93% |
| BERT-base MRPC | ~14s | ~89% |

## Relevance to ViT

Iron is for BERT (text). ViT adaptation requires:
- Image patches instead of text tokens
- Patch embedding instead of word embedding
- 2D positional encoding vs 1D
- Classification token handling

The approximation techniques (softmax, GELU, LayerNorm) transfer directly.

## Implementation Plan

1. Implement degree-27 minimax GELU approximation — see `utils/poly_approx.py`
2. Implement softmax domain decomposition
3. Test approximation accuracy on sample inputs
4. Profile depth cost vs accuracy trade-off

## Open Questions

1. Can degree-27 GELU be reduced to degree-15 without significant accuracy loss for ViT?
2. Does domain decomposition for softmax generalize to ViT's attention patterns?
3. Can LayerNorm approximation error be bounded formally?

## Status

- [ ] Read paper
- [ ] Implement softmax domain decomposition
- [ ] Implement LayerNorm approximation
- [ ] Compare GELU degree-27 vs degree-4 (BOLT) accuracy/depth trade-off
