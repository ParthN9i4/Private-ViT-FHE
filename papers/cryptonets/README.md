# CryptoNets (2016)

> Gilad-Bachrach et al., "CryptoNets: Applying Neural Networks to Encrypted Data with High Throughput and Accuracy", ICML 2016

## Why Start Here

CryptoNets established the blueprint that everything after it follows:
1. Replace non-polynomial activations with polynomial ones (x² instead of ReLU)
2. Design the network depth to fit within the noise budget
3. Accept some accuracy loss in exchange for FHE compatibility

Every paper in this repo is a refinement of this core idea.

## What the Paper Does

- Simple CNN on MNIST (handwritten digit recognition)
- Uses **square activation** (f(x) = x²) instead of ReLU
- Runs entirely in SEAL (BFV scheme at the time)
- ~250 seconds per image (2016 hardware)

## Key Insight

If you train a network **with square activations from scratch**, it can achieve competitive accuracy. The polynomial constraint is not as catastrophic as applying polynomial approximation to a pre-trained ReLU network.

## Architecture

```
Input: 28x28 image → flatten to 784 values → encrypt as CKKS vector

Layer 1: Conv(5x5, 5 filters) → Square activation
Layer 2: Mean Pool (2x2)
Layer 3: Conv(5x5, 50 filters) → Square activation
Layer 4: Mean Pool (2x2)
Layer 5: FC(100) → Square activation
Layer 6: FC(10) → argmax (plaintext after decryption)

Multiplicative depth: 2 (one per square activation)
```

## Implementation

See [`implementation.py`](implementation.py) for a TenSEAL-based reimplementation.

## Results

| Metric | Original Paper | This Implementation |
|--------|---------------|---------------------|
| MNIST Test Accuracy | 98.95% | [ ] |
| Inference Time | ~250s (2016) | [ ] |
| Multiplicative Depth | 2 | 2 |

## Exercises

1. Reproduce the accuracy on MNIST
2. Profile time per layer — where is the bottleneck?
3. Try degree-3 instead of degree-2 activation — does accuracy improve?
4. Scale to Fashion-MNIST — what changes?
