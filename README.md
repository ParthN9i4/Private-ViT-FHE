This repository consists of all the work ongoing as part of ACM Anveshan Setu Fellowship (Feb - April 2026). 

The main idea is to keep a track of works done till now i.e. Experiments for Private Inference in DEiT-Tiny.

# Polynomial Vision Transformer for CKKS-Encrypted Inference

Privacy-preserving diabetic retinopathy screening using polynomial Vision Transformers with CKKS homomorphic encryption.

**Author:** Parth Nagar 
**Affiliation:** PhD Researcher, SSSIHL | ACM Anveshan Setu Fellow, IIT Hyderabad  
**Advisor:** Prof. Vineeth N Balasubramanian  
**Status:** Active research (Phase 1 complete, Phase 2 in progress)


## Problem

Cloud-based DR screening systems (Google ARDA, MeEK) process hundreds of thousands of fundus images in **plaintext**. Fundus images carry biomarkers beyond DR - CKD, CVD, anaemia, Alzheimer's — enabling re-analysis without consent. We build a DeiT-Tiny that runs entirely under **CKKS homomorphic encryption**, so patient data is never decrypted on the server.

**The core constraint:** CKKS supports only additions and multiplications. Three ViT operations are *mathematically impossible* (not merely slow) under CKKS:
- **GELU** - uses Gaussian CDF
- **Softmax** - uses exp and division  
- **LayerNorm** - uses mean, variance, square root

All three must be replaced with polynomial approximations.

## Approach

| Component | Standard ViT | Our Replacement | CKKS Cost |
|-----------|-------------|-----------------|-----------|
| Activation | GELU | PolyGELU (trainable ax²+bx+c, LSQ-fitted) | 1 level |
| Attention | Softmax | PolyAttn (ax²+bx+c, row-normalized) | 1–20 levels |
| Normalization | LayerNorm | BatchNorm (affine at inference) | 0 levels |
| Training | Standard | Knowledge Distillation from standard teacher | N/A |

## Key Results

### Substitution Ablation (1 seed)
| Config | What's Replaced | Accuracy | Status |
|--------|----------------|----------|--------|
| A | GELU only | 76.47% | Beats teacher |
| B | Softmax only | 70.82% | Beats teacher |
| C | LayerNorm only | 82.28% | Beats teacher |
| D | GELU + Softmax | 71.84% | Beats teacher |
| **E** | **All three** | **11.06%** | **Collapsed** |

**Finding:** Each substitution works individually. The collapse is an *interaction effect* - only when all three guardrails are removed simultaneously.

### Root Cause
Activation magnitudes at layer 5: **90** (with LayerNorm) vs **29,311** (with BatchNorm) - a 326× difference. Without per-token normalization, unbounded polynomial attention compounds through layers.

### Three Fixes - Verified (5 seeds)
| Config | Mean | Std | p-value vs Teacher |
|--------|------|-----|--------------------|
| Teacher (standard) | 76.54% | 3.86% | - |
| Config E (broken) | 40.08% | 20.33% | - |
| Fix 1 (clamped PolyAttn) | 69.33% | 2.59% | 0.0005 |
| **Fix 2 (normalized PolyAttn)** | **80.47%** | **0.59%** | **0.0245** |
| Fix 3 (RMSNorm) | 71.35% | 0.84% | 0.0033 |

Fix 2 **statistically significantly beats the teacher** (p = 0.0245) and is **6.5× more stable**.

### CKKS Encrypted Inference (Classification Head)
| Model | Plaintext | Encrypted | Match | Max Error | Time/sample |
|-------|-----------|-----------|-------|-----------|-------------|
| Fix 2 | 80.31% | 80.31% | 10000/10000 | 1.14×10⁻⁵ | 175.3 ms |
| Fix 3 | 70.50% | 70.50% | 10000/10000 | 4.77×10⁻⁶ | 173.8 ms |

**100% prediction match** across all 20,000 test samples. Zero accuracy degradation under CKKS encryption.

### PolyGELU Diagnostic
The trained polynomial coefficients drift significantly from the GELU initialization - the linear term `b` collapses from 0.500 to ~0.01, resulting in a nearly symmetric quadratic. The model discovers its own optimal nonlinearity rather than approximating GELU. The [-3, 3] fitting interval covers 99.4–99.9% of values depending on the layer.

## Repository Structure

```
DEIT_TINY/
├── verify_fixes.py              # 5-seed verification (main result)
├── substitution_ablation.py     # 5-config ablation (A–E)
├── investigate_collapse.py      # Root cause + 3 fixes
├── simple_kd_baseline.py        # Teacher vs poly+KD vs poly-noKD
├── ckks_classification_head.py  # CKKS encrypted inference experiment
├── polygelu_diagnostic.py       # Activation input distribution analysis
├── verify_poly_replacements.py  # Polynomial replacement verification
├── cold_vs_warm_experiment.py   # Cold-start vs warm-start KD comparison
├── baseline_deit_improved.py    # Improved DeiT baseline
├── step6_encrypted_inference.py # Earlier encrypted inference prototype
├── experiment_results/          # JSON results from all experiments
├── results_*/                   # Per-experiment result directories
└── ckks_models/                 # Saved model checkpoints (gitignored)
```

## Setup

```bash
# Environment
conda create -n deit_baseline python=3.10
conda activate deit_baseline
pip install torch torchvision tenseal

# Run the main verification (5 seeds × 5 configs, ~7 hours on A6000)
python verify_fixes.py

# Run CKKS encrypted inference (~1 hour)
python ckks_classification_head.py

# Run PolyGELU diagnostic (~25 min)
python polygelu_diagnostic.py
```

**Hardware:** NVIDIA RTX A6000 (51 GB VRAM)  
**Dataset:** CIFAR-10 (auto-downloaded)

## KD Configuration

- **Type:** Response-based logit-level KD (Hinton et al., 2015)
- **Temperature:** T = 4.0
- **Alpha:** 0.1 (loss = 0.1 × KL + 0.9 × CE)
- **No T² scaling** (removed after observing training collapse)
- KD is training-only — the teacher is discarded at inference

## CKKS Parameters

- Ring dimension: N = 8192
- Scaling factor: 2⁴⁰
- Multiplicative levels: 1 (classification head)
- Library: TenSEAL (Microsoft SEAL wrapper)

## Current Limitations

- All results on CIFAR-10 with 6-layer model (standard DeiT-Tiny is 12 layers)
- Only classification head encrypted (full backbone is Phase 2)
- PolyAttn is a generic polynomial, not a principled softmax approximation
- KD hyperparameters (T=4, α=0.1) not ablated
- No medical dataset validation yet

## Next Steps

1. Medical dataset validation (APTOS-2019 fundus images)
2. Full backbone CKKS encrypted inference
3. Comparison with Power-Softmax (Zimerman et al., ICLR 2025)
4. Workshop paper submission (PPAI@AAAI or SiMLA@ACNS)

## Key References

- Hinton et al., "Distilling the Knowledge in Neural Networks," 2015
- Baruch et al., "Sensitive Tuning of Large Scale CNNs for E2E Secure Prediction," SiMLA 2022
- Zimerman et al., "Converting Transformers to Polynomial Form for Secure Inference Over HE," ICML 2024
- Zimerman et al., "Power-Softmax: Towards Secure LLM Inference over Encrypted Data," ICLR 2025
- Tastan & Nandakumar, "CaPriDe Learning," CVPR 2023
- Lee et al., "HETAL: Efficient Privacy-preserving Transfer Learning with HE," ICML 2023
- Touvron et al., "Training Data-Efficient Image Transformers & Distillation through Attention," ICML 2021

## License

This is private research code. Not for redistribution.
