# Power-Softmax and Polynomial Attention Variants

Covering three polynomial softmax replacements for FHE-compatible attention:
1. **Power-Softmax** — simple quadratic normalization
2. **MGF-Softmax** — moment-generating function reformulation
3. **L2Q** — learnable quadratic from SAL-ViT (see also `papers/sal_vit/`)

## Why Replace Softmax?

Standard softmax: `a_i = exp(q·k_i) / Σ_j exp(q·k_j)`

Problems for FHE:
- `exp` is transcendental — needs high-degree polynomial (degree 15–27 via domain decomposition)
- Each domain-decomposition approximation costs 5–8 multiplicative levels
- This is the single most expensive operation in private transformer inference

## Approach 1: Power-Softmax

Replace `exp` with a power function:
```
a_ij = (q_i · k_j)^p / Σ_j (q_i · k_j)^p
```

For `p=2` (degree-2 polynomial):
- No `exp` — no high-degree approximation needed
- FHE depth: 1 level for the square
- Division for normalization: done in plaintext post-decryption, or approximated

**Training issue**: Power-Softmax needs range control (attention logits can't be too large).
Solution: scale QK.T by a smaller factor, or use range-loss (PolyTransformer).

### Training Trick — Temperature Scaling

```
a_ij = (q_i · k_j / sqrt(d) / T)^2 / Σ_j (q_i · k_j / sqrt(d) / T)^2
```
Smaller `T` compresses the logit range; larger `p` sharpens the distribution.

## Approach 2: MGF-Softmax

The moment-generating function M(t) = E[exp(tX)] is related to softmax.

MGF-Softmax uses a polynomial approximation of `exp` around 0:
```
exp(x) ≈ 1 + x + x²/2 + x³/6 + ...   (Taylor series)
```

Using degree-7 truncation: `exp(x) ≈ Σ_{k=0}^{7} x^k / k!`

This has better approximation quality than Power-Softmax but costs more:
- FHE depth: `ceil(log2(7)) = 3` levels for the polynomial

**When to use**: when Power-Softmax has too much accuracy loss but you can afford 3 levels.

## Approach 3: L2Q (Learnable 2-Quad)

From SAL-ViT — per-head learnable quadratic:
```
a_ij = (α · (q_i · k_j)^2 + β · (q_i · k_j) + γ) / normalization
```

Where `α, β, γ` are learned parameters per attention head.

Depth: 1 level (same as Power-Softmax) but higher expressivity.

**Key advantage**: the coefficients `α, β, γ` are tuned per layer and per head during training,
so the model learns the best quadratic approximation for each attention pattern.

## Depth and Accuracy Comparison

| Method | FHE Depth | Approx Quality | Training |
|--------|-----------|----------------|---------|
| Softmax (standard) | 5–8 levels | Exact | Standard |
| Power-Softmax p=2 | 1 level | Low-Medium | Needs range control |
| MGF-Softmax deg-7 | 3 levels | Medium-High | Standard |
| L2Q (learnable) | 1 level | Medium-High | Learned coefficients |
| Linear Attn (BOLT) | 0 levels | Low | No normalization |

## Implementation

See [`implementation.py`](implementation.py) for:
- `PowerSoftmax` — `(Q@K.T).pow(p) / norm`
- `MGFSoftmax` — Taylor series approximation of exp
- `L2Q` — learnable quadratic (see also `papers/sal_vit/` for the full SAL-ViT context)
- `benchmark_softmax_replacements()` — compare all three on a toy ViT block

## Open Questions

1. Which method best preserves ViT attention patterns on CIFAR-10?
2. Does MGF-Softmax quality improve enough at degree-15 to justify the extra 2 levels vs. degree-7?
3. Can L2Q coefficients be initialized from a trained standard-softmax model?
4. What is the minimum Power-Softmax power `p` for competitive accuracy?
