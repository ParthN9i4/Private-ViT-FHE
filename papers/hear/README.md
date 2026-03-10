# HEAR (2021)

> Song et al., "HEAR: Human Action Recognition via Neural Networks on Homomorphically Encrypted Data", 2021

## Why This Matters

HEAR introduces the **hybrid FHE + garbled circuits** approach that became a standard pattern. Rather than approximating everything polynomially, it evaluates linear layers in CKKS and non-linearities via 2PC garbled circuits.

## Key Ideas

### 1. Hybrid Protocol

Split computation by layer type:
- **Linear layers** (matmul, conv): evaluate in CKKS ciphertext domain — no noise growth from non-linearities
- **Non-linear layers** (ReLU): evaluate via garbled circuits (2PC)

### 2. Communication-Efficient FHE ↔ GC Switching

Switching from FHE to garbled circuits requires:
1. Server sends garbled circuit + garbled inputs
2. Client uses OT (Oblivious Transfer) to get labels for their ciphertext bits
3. Client evaluates GC, sends result back encrypted

Cost: ~1 communication round per non-linear layer.

### 3. No Polynomial Approximation Needed for Activations

ReLU is evaluated exactly (not approximated). This avoids:
- Accuracy loss from approximation error
- High-degree polynomial overhead (depth, noise)

**Trade-off**: Requires client-server interaction; not non-interactive.

## Architecture

```
For each layer:
  Linear(x)        → evaluate in CKKS (no interaction)
  ReLU(x)          → switch to garbled circuit (1 round)
  BatchNorm(x)     → CKKS (precompute γ, β as plaintext)
```

## Comparison vs Pure FHE

| Aspect | HEAR (hybrid) | Pure FHE |
|--------|--------------|----------|
| Non-linearities | Exact (GC) | Approx polynomial |
| Communication | Multiple rounds | Non-interactive |
| Depth budget | Lower (no GC depth) | Higher |
| Accuracy | Same as plaintext | Slight degradation |

## Relevance to ViT

ViT non-linearities:
- **Softmax**: HEAR would use GC — avoids complex domain-decomposition approx
- **GELU**: Exact via GC vs degree-27 poly (Iron) — latency trade-off
- **LayerNorm**: Approximate or GC

## Implementation Notes

This paper is primarily a reference for the hybrid approach. For this repo:
- Understand the FHE ↔ GC switching protocol
- Compare latency vs polynomial approximation approaches

## Open Questions

1. Is garbled circuit overhead less than high-degree polynomial approximation for GELU?
2. Can we reduce the number of communication rounds by batching non-linearities?
3. For a 6-layer ViT, how many GC rounds total?

## Status

- [ ] Read paper
- [ ] Implement toy hybrid example (1 layer: linear in CKKS + ReLU in GC)
- [ ] Benchmark vs polynomial approximation approach
