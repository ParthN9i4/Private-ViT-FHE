# Private-ViT-FHE

> **Research**: Efficient Private Inference for Vision Transformers under Fully Homomorphic Encryption (CKKS)

PhD research repository — systematic paper implementations, experiments, and benchmarks toward practical private inference on transformer-based vision models.

---

## The Core Problem

Vision Transformers (ViTs) achieve SOTA on image classification, but inference requires sending raw images to a server. FHE allows a client to encrypt their image, have the server run the ViT on the ciphertext, and receive an encrypted prediction — the server never sees the image.

**The hard part**: ViTs contain operations that break or explode under CKKS:
- `softmax` in attention — requires division and `exp`, both non-polynomial
- `LayerNorm` — requires square root and division
- `GELU`/`ReLU` activations — non-polynomial, require high-degree approximation
- Attention matrix multiply — quadratic in sequence length, enormous multiplicative depth

This repo tracks the state-of-the-art and implements solutions bottom-up.

---

## Paper Roadmap

Work through these in order — each builds on the last:

| # | Paper | Year | Key Contribution | Status |
|---|-------|------|-----------------|--------|
| 1 | [CryptoNets](papers/cryptonets/) | 2016 | First neural net on FHE | [ ] |
| 2 | [HEAR](papers/hear/) | 2021 | Hybrid FHE+garbled circuits for non-linearities | [ ] |
| 3 | [Iron](papers/iron/) | 2022 | Transformer private inference (BERT) | [ ] |
| 4 | [BOLT](papers/bolt/) | 2023 | Bootstrapping-free private ViT inference | [ ] |
| 5 | [AutoPrivacy / THE-X](papers/autoprivacy/) | 2022-24 | Automated FHE-friendly model design | [ ] |
| 6 | [LHE-Transformer](papers/lhe_transformer/) | 2024 | Leveled HE for transformers, no bootstrapping | [ ] |

---

## Repository Structure

```
Private-ViT-FHE/
├── papers/                  # One directory per paper
│   ├── cryptonets/          # CryptoNets (2016) implementation + notes
│   ├── hear/                # HEAR (2021)
│   ├── iron/                # Iron (2022)
│   ├── bolt/                # BOLT (2023)
│   ├── autoprivacy/         # AutoPrivacy / THE-X
│   └── lhe_transformer/     # LHE-Transformer (2024)
├── experiments/
│   ├── baseline/            # Plaintext ViT baseline (PyTorch)
│   ├── approximations/      # Polynomial approximation of non-linearities
│   ├── fhe_inference/       # FHE inference experiments
│   └── benchmarks/          # Latency/throughput across schemes
├── utils/
│   ├── poly_approx.py       # Minimax polynomial approximation tools
│   ├── depth_counter.py     # Multiplicative depth analysis
│   ├── ckks_helpers.py      # CKKS parameter selection utilities
│   └── benchmark.py         # Unified benchmarking harness
├── notebooks/               # Exploratory analysis and visualization
├── benchmarks/              # Benchmark results (JSON/CSV)
└── docs/                    # Extended notes, derivations, reading log
```

---

## Setup

```bash
# Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# For FHE operations (choose one or more):
# TenSEAL (Python bindings for SEAL — easiest to start)
pip install tenseal

# Concrete-ML (Zama — sklearn/PyTorch compatible)
pip install concrete-ml

# OpenFHE Python bindings
# See: https://github.com/openfheorg/openfhe-python
```

---

## Research Questions (Evolving)

1. What is the minimum multiplicative depth required for a usable ViT?
2. Can attention be approximated with degree ≤ 15 polynomials without >5% accuracy drop?
3. What is the latency/accuracy Pareto frontier for private ViT inference on CIFAR-10 vs ImageNet?
4. Is bootstrapping avoidable for a 6-layer ViT?
5. How does CKKS noise growth interact with LayerNorm approximation error?

---

## Reading Log

See [`docs/reading_log.md`](docs/reading_log.md) for annotated notes on every paper read.

---

## Benchmarks

See [`benchmarks/`](benchmarks/) for reproducible timing results. All benchmarks run on:
- CPU: [to be filled]
- RAM: [to be filled]
- SEAL version: [to be filled]
- TenSEAL version: [to be filled]
