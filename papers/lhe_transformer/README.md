# LHE-Transformer (2024)

> To be filled after reading. Leveled Homomorphic Encryption for transformer inference without bootstrapping.

## Why This Matters

LHE-Transformer pushes the state of the art on **bootstrapping-free** private transformer inference. By carefully managing the noise budget across all layers, it avoids the expensive bootstrapping operation that dominates latency in earlier work.

## Key Ideas

*To be filled after reading the paper.*

### 1. Leveled HE Noise Management

Core challenge: transformer depth exceeds the noise budget of leveled CKKS, requiring bootstrapping. LHE-Transformer avoids this by:
- [ ] (to be filled)

### 2. Approximation Strategy

- [ ] (to be filled — which ops are approximated and how)

### 3. Architecture Modifications

- [ ] (to be filled — what architectural changes are made)

## Depth Budget Analysis

*To be filled after reading.*

| Component | Levels |
|-----------|--------|
| Patch embedding | ? |
| Per-block subtotal | ? |
| N blocks total | ? |
| Classifier head | ? |
| **Total** | **?** |

## Performance

*To be filled after reading.*

| Dataset | Accuracy | Latency | Bootstrapping? |
|---------|----------|---------|----------------|
| CIFAR-10 | ? | ? | No |
| Tiny-ImageNet | ? | ? | No |

## Comparison to BOLT

| Aspect | BOLT | LHE-Transformer |
|--------|------|-----------------|
| Year | 2023 | 2024 |
| Bootstrapping | No | No |
| Depth | 50 | ? |
| Accuracy (CIFAR-10) | 88.3% | ? |
| Latency | 6.7s | ? |

## Implementation Plan

1. Read the paper and fill in the above sections
2. Identify key techniques that differ from BOLT
3. Implement any new approximation methods
4. Benchmark against BOLT implementation

## Open Questions

1. How does LHE-Transformer manage depth vs BOLT?
2. Is there an accuracy improvement over BOLT on the same datasets?
3. Does it generalize to larger models / datasets?

## Status

- [ ] Find and read paper
- [ ] Fill in key techniques
- [ ] Implement and benchmark
