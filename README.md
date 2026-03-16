# Private-ViT-FHE

> **Research**: Efficient Private Inference for Vision Transformers under Fully Homomorphic Encryption (CKKS)

PhD research repository — systematic paper implementations, experiments, and benchmarks toward practical private inference on transformer-based vision models using knowledge distillation and FHE.

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

## Phase A: FHE Foundations for Transformers

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

## Phase B: KD + Medical Imaging + FHE

The next phase: distill large ViTs into FHE-friendly compact models for medical image analysis.

| # | Paper | Year | Key Contribution | Status |
|---|-------|------|-----------------|--------|
| 7 | [DeiT](papers/deit/) | 2021 | Data-efficient ViT via distillation token | [ ] |
| 8 | [TinyViT](papers/tinyvit/) | 2022 | Sparse KD from large IN-21k teacher to tiny ViT | [ ] |
| 9 | [PolyTransformer](papers/poly_transformer/) | 2024 | First polynomial ViT for pure FHE inference | [ ] |
| 10 | [MedBlindTuner](papers/medblindtuner/) | 2024 | DeiT + CKKS on biomedical images | [ ] |
| 11 | [SAL-ViT](papers/sal_vit/) | 2023 | Secure & accurate ViT via lightweight MPC | [ ] |
| 12 | [AutoFHE](papers/autofhe/) | 2023 | Evolutionary search for HE-friendly activations | [ ] |
| 13 | [Power-Softmax](papers/power_softmax/) | 2024 | Polynomial softmax for secure LLM inference | [ ] |

---

## Research Questions (Evolving)

**Phase A — FHE fundamentals:**
1. What is the minimum multiplicative depth required for a usable ViT?
2. Can attention be approximated with degree ≤ 15 polynomials without >5% accuracy drop?
3. What is the latency/accuracy Pareto frontier for private ViT inference on CIFAR-10 vs ImageNet?
4. Is bootstrapping avoidable for a 6-layer ViT?
5. How does CKKS noise growth interact with LayerNorm approximation error?

**Phase B — Novelty gaps (open problems):**
6. Can KD be used to train a fully-polynomial ViT that is both FHE-friendly *and* competitive on medical datasets (MedMNIST)?
7. Is a KD pipeline (ReLU teacher → polynomial student) sufficient to close the accuracy gap for encrypted ViT inference?
8. Does plaintext-domain KD generalize across medical imaging tasks (retinal, dermatology, pathology)?
9. Can the encrypted classification head approach (HETAL/MedBlindTuner) scale to multi-class medical datasets?
10. What is the privacy-accuracy-latency tradeoff for pure FHE (all layers encrypted) vs. hybrid (encrypted head only)?

---

## Repository Structure

```
Private-ViT-FHE/
├── papers/                  # One directory per paper
│   ├── cryptonets/          # CryptoNets (2016)
│   ├── hear/                # HEAR (2021)
│   ├── iron/                # Iron (2022)
│   ├── bolt/                # BOLT (2023) — BoBa-ViT, no bootstrapping
│   ├── autoprivacy/         # AutoPrivacy / THE-X (2022-24)
│   ├── lhe_transformer/     # LHE-Transformer (2024)
│   ├── deit/                # DeiT (2021) — distillation token
│   ├── tinyvit/             # TinyViT (2022) — sparse pre-training KD
│   ├── poly_transformer/    # PolyTransformer (2024) — first polynomial ViT
│   ├── medblindtuner/       # MedBlindTuner (2024) — DeiT + CKKS on MedMNIST
│   ├── sal_vit/             # SAL-ViT (2023) — MPC-based private ViT
│   ├── autofhe/             # AutoFHE (2023) — EvoReLU
│   └── power_softmax/       # Power-Softmax (2024) — polynomial softmax
├── experiments/
│   ├── baseline/
│   │   └── vit_plaintext_baseline.py      # Plaintext ViT accuracy baseline
│   ├── approximations/
│   │   └── gelu_approx_study.py           # Polynomial GELU approximation study
│   ├── week1_kd_basics/                   # Week 1: KD pipeline experiments
│   │   ├── 01_vit_medical_baseline.py     # ViT-Base teacher on RetinaMNIST
│   │   ├── 02_deit_kd_pipeline.py         # DeiT-Tiny KD from ViT-Base teacher
│   │   ├── 03_poly_gelu_kd.py             # PolyGELU (degree-4) with/without KD
│   │   ├── 04_poly_softmax_study.py       # Polynomial attention mechanism comparison
│   │   └── 05_layernorm_ablation.py       # LayerNorm variant ablation
│   ├── week2_fhe_inference/               # Week 2: FHE inference experiments
│   │   ├── 01_single_block_encryption.py  # Single PolyTransformerBlock under CKKS
│   │   ├── 02_medmnist_fhe_baseline.py    # End-to-end FHE inference on MedMNIST
│   │   └── 03_kd_fhe_comparison.py        # KD student vs FHE baseline comparison
│   ├── fhe_inference/                     # End-to-end FHE forward pass experiments
│   └── benchmarks/                        # Latency/throughput benchmark scripts
├── utils/
│   ├── poly_approx.py       # Minimax & Chebyshev polynomial approximation
│   ├── depth_counter.py     # Multiplicative depth analysis and budget estimation
│   ├── ckks_helpers.py      # CKKS parameter selection utilities
│   └── benchmark.py         # Unified benchmarking harness
├── notebooks/
│   ├── 01_gelu_approximation.ipynb        # GELU polynomial approximation explorer
│   └── 02_depth_budget_analysis.ipynb     # Depth budget analysis for ViT configs
├── benchmarks/              # Benchmark results (JSON/CSV)
└── docs/
    ├── reading_log.md              # Annotated notes on every paper read
    ├── ckks_parameter_guide.md     # CKKS parameter selection quick reference
    ├── kd_fhe_literature_review.md # Full 6-part literature review
    ├── learning_roadmap.md         # 6-phase structured learning path
    ├── weekly_study_plan.md        # Day-by-day Week 1-2 schedule
    └── paper_tiers.md              # All papers tiered by implementation priority
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

### MedMNIST Setup

```bash
pip install medmnist

# Quick check: list all available datasets
python -c "import medmnist; print(medmnist.INFO.keys())"

# Datasets used in Phase B experiments:
# - RetinaMNIST  (1600 train, 400 val, 400 test; 28×28 retinal fundus)
# - DermaMNIST   (7007 train; 28×28 dermatoscopy, 7-class)
# - BreastMNIST  (546 train; 28×28 ultrasound, binary)
```

---

## Utilities

**`utils/poly_approx.py`** — Polynomial approximation for FHE-friendly activations:
- `minimax_approx(fn, degree, domain)` — L-infinity optimal approximation via Nelder-Mead
- `chebyshev_approx(fn, degree, domain)` — Fast Chebyshev approximation (good starting point)
- `approx_gelu_degree4` — BOLT paper's degree-4 GELU for domain `[-5, 5]`
- `eval_polynomial_ckks(coeffs, x)` — Horner's method evaluation (minimizes multiplications)
- `approx_error(fn, coeffs, domain)` — Max/mean absolute error and depth metrics

**`utils/depth_counter.py`** — Multiplicative depth analysis:
- `vit_depth_budget(n_layers, gelu_degree, attention_type, layernorm_type)` — Per-component depth breakdown
- `print_depth_report()` — Formatted table for common configs (Vanilla ViT, BOLT-style, Iron-style)

**`utils/ckks_helpers.py`** — CKKS parameter selection utilities

**`utils/benchmark.py`** — Unified benchmarking harness

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| [`01_gelu_approximation.ipynb`](notebooks/01_gelu_approximation.ipynb) | Interactive GELU polynomial approximation explorer — compare degree vs. accuracy vs. multiplicative depth |
| [`02_depth_budget_analysis.ipynb`](notebooks/02_depth_budget_analysis.ipynb) | Depth budget analysis across ViT configurations — visualize level consumption per layer |

---

## Key Related Work (External Repos)

| Paper | Code |
|-------|------|
| HETAL (ICML'23 Oral) | https://github.com/CryptoLabInc/HETAL |
| CaPriDe (CVPR'23) | https://github.com/tnurbek/capride-learning |
| BOLT | https://github.com/inpluslab/bolt |
| AutoFHE | https://github.com/HungYiHo/AutoFHE |
| TinyViT | https://github.com/microsoft/Cream/tree/main/TinyViT |
| DeiT | https://github.com/facebookresearch/deit |

---

## Reading Log

See [`docs/reading_log.md`](docs/reading_log.md) for annotated notes on every paper read.

## Literature Review

See [`docs/kd_fhe_literature_review.md`](docs/kd_fhe_literature_review.md) for the full survey of KD+FHE+ViT+Medical papers.

## Paper Tiers

See [`docs/paper_tiers.md`](docs/paper_tiers.md) for all papers ranked by implementation priority.

---

## Benchmarks

See [`benchmarks/`](benchmarks/) for reproducible timing results. All benchmarks run on:
- CPU: [to be filled]
- RAM: [to be filled]
- SEAL version: [to be filled]
- TenSEAL version: [to be filled]
