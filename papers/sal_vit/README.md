# SAL-ViT (2023)

> "SAL-ViT: Towards Latency Efficient Private Inference on ViT using Selective Attention with a Learnable Quadratic Polynomial"

## Why This Matters

SAL-ViT introduces two ideas that directly address the FHE inference bottleneck:

1. **L2Q (Learnable 2-Quad)** — per-head learnable quadratic replaces softmax. Unlike Power-Softmax (fixed `x^2`), L2Q learns the best `ax^2 + bx + c` for each attention head.

2. **Selective Attention** — not all attention layers need polynomial replacement. Some layers handle local structure (can use simpler L2Q) while others handle global semantics (may need more expressive attention). SAL-ViT **searches** for which layers to polynomialize.

Together: achieve nearly full ViT accuracy while only paying the FHE cost for layers that truly need it.

## Key Ideas

### 1. L2Q — Learnable 2-Quad

Per-head parameters `{α_h, β_h, γ_h}` for each head `h`:
```
score_h(q, k) = α_h · (q·k)^2 + β_h · (q·k) + γ_h
A_h = score_h / Σ_k score_h(q, k_k)   (row normalization)
```

Training:
- Initialize so `α=0, β=1, γ=0` → reduces to linear attention (BOLT)
- Or initialize as best quadratic fit to softmax on a calibration batch
- Train end-to-end; α, β, γ specialize per head

FHE depth: 1 level (degree-2 poly).

### 2. Selective Attention Search (SAS)

Not all layers need the polynomial. SAS decides which layers get L2Q and which keep standard softmax (run outside FHE or via expensive domain decomposition).

SAS procedure:
1. Start with all layers using standard softmax
2. Greedily replace layers with L2Q, measuring accuracy drop per layer
3. Accept replacements where accuracy drop < threshold
4. Output: a binary mask `[poly, poly, std, poly, std, std, ...]`

Intuition: **early layers** are local and low-level → L2Q works well. **Later layers** are global → may need standard softmax.

### 3. Mixed Inference

For layers using standard softmax:
- Option A: Run attention non-privately (send attention pattern to server) — breaks full privacy
- Option B: Use expensive polynomial approximation for those layers only
- Option C: Use garbled circuits for those layers (hybrid FHE + 2PC, like HEAR)

SAL-ViT uses Option B for full privacy — the key win is that only a few layers need expensive softmax.

## Results (CIFAR-10)

| Configuration | Accuracy | Total FHE Depth |
|--------------|----------|-----------------|
| ViT-S (standard) | 92.1% | N/A |
| All layers L2Q | 87.5% | 30 (6 blocks × 5) |
| SAL-ViT (selective) | 89.3% | 38 (4 poly + 2 std-approx) |
| BoBa-ViT (BOLT) | 88.3% | 50 |

SAL-ViT outperforms BOLT (88.3%) at comparable depth while using a simpler architecture.

## FHE Depth Breakdown

```
L2Q layer:
  QKV projections: 1 level
  Q @ K.T: 1 level
  L2Q poly (degree 2): 1 level
  Output projection: 1 level
  Layer subtotal: 4 levels

Standard softmax layer (domain decomp):
  QKV projections: 1 level
  Softmax approximation: 5–8 levels
  Output projection: 1 level
  Layer subtotal: 7–10 levels
```

With SAS finding 4 L2Q + 2 std layers: `4×4 + 2×8 + MLP` ≈ 38 levels.

## Implementation

See [`implementation.py`](implementation.py) for:
- `L2QAttention` — multi-head attention with per-head learnable quadratic
- `SelectiveAttentionViT` — ViT where each layer independently uses L2Q or standard attention
- `search_attention_schedule()` — greedy search for which layers to polynomialize

Note: `L2Q` module is also implemented in `papers/power_softmax/implementation.py`
for the side-by-side benchmark. This file provides the full attention layer wrapper.

## Open Questions

1. Does the SAS schedule found on CIFAR-10 generalize to MedMNIST?
2. Can we initialize L2Q coefficients by fitting to standard-softmax outputs (post-hoc)?
3. What is the accuracy vs. depth Pareto frontier as we vary how many layers use L2Q?
4. Does combining SAL-ViT (selective attention) with AutoFHE (mixed activation degree) yield better results than either alone?
