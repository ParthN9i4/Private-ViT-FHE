# Knowledge Distillation + FHE + ViT + Medical Imaging: Literature Review

Systematic survey of the intersection of KD, FHE, Vision Transformers, and medical image
analysis. Organized into 6 parts, from foundational KD+FHE work to open novelty gaps.

---

## Part 1: Knowledge Distillation + FHE — Direct Prior Work

Papers that explicitly combine KD with FHE or that train HE-friendly models using distillation.

| Year | Venue | Paper | KD Role | FHE Scheme | Key Result |
|------|-------|-------|---------|-----------|-----------|
| 2019 | CVPRW | Nandakumar et al., "Towards DNN Training on Encrypted Data" | None (training baseline) | BGV | 96% on MNIST; 1.5 days per 60-image batch |
| 2021 | arXiv | Baruch et al., "Methodology for Training HE-Friendly NNs" | ReLU teacher → poly student | CKKS-compatible | ≤2% accuracy drop on COVID-19 X-ray/CT |
| 2023 | CVPR | CaPriDe (Tastan & Nandakumar) | Encrypted KL/L2 distillation across federated clients | CKKS | Federated KD without sharing weights or data |
| 2023 | ICML (Oral) | HETAL (Lee et al.) | No KD — transfer via encrypted head training | CKKS | Encrypted classification head; 567–3442s; competitive accuracy |
| 2024 | AAAI-W | MedBlindTuner (Panzade et al.) | No KD — fine-tune DeiT head encrypted | CKKS | First DeiT+CKKS on MedMNIST |
| 2024 | ICML | Zimerman et al., "Converting Transformers to Polynomial Form" | No KD — direct polynomial conversion | CKKS-compatible | First pure-polynomial ViT for FHE inference |
| 2024 | arXiv | Power-Softmax (Zimerman et al.) | No KD — architectural replacement | CKKS-compatible | Polynomial LLMs at 32 layers / 1B+ params |

### Key Observations

1. **KD and FHE are rarely combined**: Most FHE inference papers design a model directly
   for FHE constraints. Baruch et al. (2021) and CaPriDe are the primary exceptions.
2. **KD as accuracy bridge**: Baruch et al. show that KD from a ReLU teacher closes most
   of the accuracy gap introduced by replacing ReLU with low-degree polynomials.
3. **Medical imaging is underexplored**: Only MedBlindTuner and HETAL evaluate on medical
   datasets; neither uses KD.
4. **The encrypted head approach is the current practical baseline**: HETAL and MedBlindTuner
   encrypt only the classification head. Fully encrypted ViT inference remains slow (hours).

---

## Part 2: HE-Friendly Network Design (Chronological)

Papers focused on designing or adapting neural networks to be compatible with FHE constraints
(polynomial activations, bounded depth, SIMD-friendly computation).

| Year | Paper | Key Innovation |
|------|-------|----------------|
| 2016 | **CryptoNets** (Gilad-Bachrach et al.) | First NN on FHE; square activation (x²); no softmax |
| 2019 | **Nandakumar et al.** | DNN training on BGV-encrypted data; sigmoid + quadratic loss |
| 2021 | **Baruch et al.** | Trainable polynomial activations; KD from ReLU teacher |
| 2021 | **HEAR** (Song et al.) | Hybrid: CKKS for linear layers, garbled circuits for ReLU |
| 2022 | **Iron** (Hao et al.) | Degree-27 GELU approx; composite softmax for BERT |
| 2022 | **AESPA** | Attention with polynomial expansion and sparse approximation |
| 2022 | **HEMET** | HE-friendly CNN with co-designed activations and packing |
| 2023 | **BOLT** (Xu et al.) | Bootstrapping-free BoBa-ViT; linear attention + poly GELU |
| 2023 | **AutoFHE** (Ao & Boddeti) | EvoReLU — evolutionary search for optimal polynomial activations |
| 2023 | **THE-X / AutoPrivacy** | NAS with FHE depth as architecture search objective |
| 2024 | **Zimerman et al.** | First fully polynomial ViT (ICML); RangeLoss + LinearNorm |
| 2024 | **LIME** | Low-depth image model with FHE co-design |

### Design Patterns

- **Activation replacement**: x² (degree-1 mult), degree-4 poly, degree-27 minimax poly
- **Attention replacement**: linear attention (no exp), Power-Softmax (x^p), L2Q
- **Normalization replacement**: LinearNorm (affine only), ScalarNorm (single scale)
- **Training co-design**: RangeLoss (Zimerman), trainable coefficients (Baruch), EvoReLU (AutoFHE)
- **Encoding**: diagonal matrix encoding (BOLT, Iron), BSGS rotation reduction, slot packing

---

## Part 3: Vision Transformers under FHE

### 3a. Pure FHE Inference (all layers encrypted)

| Paper | Year | Model | Dataset | Latency | Notes |
|-------|------|-------|---------|---------|-------|
| THE-X | 2022 | ViT-B | CIFAR-10 | ~1h | First private ViT; high latency |
| NEXUS | 2022 | ViT-S | ImageNet | ~10min | Better depth management |
| THOR | 2023 | ViT-S | CIFAR-100 | ~20min | Packed attention |
| MOAI | 2023 | DeiT-S | ImageNet | ~5min | Optimized packing |
| **Zimerman et al.** | 2024 | Custom ViT | ImageNet subset | TBD | First polynomial-only ViT |
| LHE-Transformer | 2024 | ViT-B | ImageNet | TBD | No bootstrapping |

### 3b. MPC-Based Private ViT Inference (2PC — not pure FHE)

| Paper | Year | Model | Key Technique | Latency |
|-------|------|-------|---------------|---------|
| MPCViT | 2022 | DeiT-S | Mixed OT+HE; attention in GC | ~2min |
| **SAL-ViT** | 2023 | DeiT-S | Selective head linearization | ~1min |
| PriViT | 2023 | ViT-B | Attention via SS; GELU via GC | ~3min |

### 3c. Attention Approximation Methods

| Method | Polynomial Form | Degree | Accuracy Impact | FHE-ready |
|--------|----------------|--------|-----------------|-----------|
| MGF-Softmax | Gaussian Mixture | ~8 | ~1% drop | ⚠️ partial |
| Power-Softmax | x^p / Σx^p | p×(depth) | ~2% drop | ✅ |
| L2Q | L2 normalization variant | low | ~1.5% drop | ✅ |
| Linear Attention | Q·K^T·V (no exp) | 1 | ~3–5% drop | ✅ |
| RangeLoss + Taylor | Bounded Taylor series | 4–8 | ~2% drop | ✅ |

---

## Part 4: Compact Distilled ViTs in Medical Imaging

### 4a. Compact ViT Architectures (Teachers and Students)

| Model | Params | Top-1 (IN-1k) | KD Source | Notes |
|-------|--------|--------------|-----------|-------|
| ViT-B/16 | 86M | 81.8% | JFT-300M pretrain | Standard teacher |
| DeiT-Small | 22M | 79.8% | ViT-B teacher | Best KD student (small) |
| DeiT-Tiny | 5.7M | 72.2% | ViT-B teacher | Fast baseline |
| TinyViT-21M | 21M | 83.1% | ViT-L IN-21k teacher | Sparse KD |
| TinyViT-11M | 11M | 81.5% | ViT-L IN-21k teacher | Good accuracy/efficiency |
| TinyViT-5M | 5M | 79.1% | ViT-L IN-21k teacher | Compact baseline |
| MobileViT-S | 5.6M | 78.4% | None | Hybrid CNN-ViT |

### 4b. Medical Imaging Applications

| Paper | Dataset | Architecture | Privacy | Notes |
|-------|---------|-------------|---------|-------|
| HETAL (Lee et al., ICML'23) | PathMNIST, DermaMNIST, AG News | ViT-B + encrypted linear head | CKKS (head only) | Practical encrypted transfer learning |
| MedBlindTuner (Panzade et al., AAAI-W'24) | MedMNIST (multiple) | DeiT-Small + CKKS head | CKKS (head only) | First DeiT+FHE on biomedical images |
| MICCAI 2025 paper 0621 | TBD | TBD | TBD | Pending indexing |

### 4c. MedMNIST Benchmark Reference

| Dataset | Task | Train | Test | Image Size | Metric |
|---------|------|-------|------|-----------|--------|
| PathMNIST | 9-class colon pathology | 89996 | 7180 | 28×28 RGB | ACC/AUC |
| DermaMNIST | 7-class skin lesion | 7007 | 2005 | 28×28 RGB | ACC/AUC |
| RetinaMNIST | 5-class retinal disease | 1080 | 400 | 28×28 RGB | ACC/AUC |
| BreastMNIST | Binary breast ultrasound | 546 | 156 | 28×28 Gray | ACC/AUC |
| ChestMNIST | Multi-label chest X-ray | 78468 | 22433 | 28×28 Gray | AUC |

---

## Part 5: Novelty Gap Analysis

The following table maps existing work to the key axes of our research.
A ✅ means the paper addresses that axis; ❌ means it does not.

| Paper | KD | Poly-ViT | Pure FHE | Medical | Notes |
|-------|:--:|:--------:|:--------:|:-------:|-------|
| CryptoNets (2016) | ❌ | ❌ | ✅ | ❌ | CNN only, toy task |
| Nandakumar (2019) | ❌ | ❌ | ✅ | ❌ | Training on encrypted data |
| HEAR (2021) | ❌ | ❌ | ❌ | ❌ | Hybrid 2PC |
| Iron (2022) | ❌ | ❌ | ❌ | ❌ | BERT, hybrid |
| Baruch et al. (2021) | ✅ | ⚠️ CNN | ❌ | ✅ | Best KD+HE-friendly baseline |
| BOLT (2023) | ❌ | ✅ | ✅ | ❌ | No KD, no medical |
| SAL-ViT (2023) | ❌ | ⚠️ partial | ❌ | ❌ | MPC-based |
| AutoFHE (2023) | ❌ | ❌ | ❌ | ❌ | CNN, activation search |
| CaPriDe (2023) | ✅ | ❌ | ⚠️ partial | ❌ | Federated, not ViT |
| HETAL (2023) | ❌ | ❌ | ⚠️ partial | ✅ | Encrypted head only |
| TinyViT (2022) | ✅ | ❌ | ❌ | ❌ | No FHE |
| DeiT (2021) | ✅ | ❌ | ❌ | ❌ | No FHE |
| MedBlindTuner (2024) | ❌ | ❌ | ⚠️ partial | ✅ | Encrypted head only |
| Zimerman et al. (2024) | ❌ | ✅ | ✅ | ❌ | No KD, no medical |
| Power-Softmax (2024) | ❌ | ✅ | ✅ | ❌ | LLM focus |
| **Our Target** | **✅** | **✅** | **✅** | **✅** | **All four axes** |

### The 5 Novelty Gaps

1. **No paper applies KD to train a fully polynomial ViT for pure FHE inference.**
   - Baruch et al. apply KD to CNNs; Zimerman et al. convert ViTs without KD.
   - Combining them (KD bridges the accuracy gap from polynomial conversion) is open.

2. **No paper evaluates a fully encrypted ViT on medical imaging datasets.**
   - HETAL and MedBlindTuner use encrypted heads; BOLT uses toy datasets.
   - Medical imaging under pure FHE is completely unexplored.

3. **No paper uses TinyViT or DeiT-Tiny as the FHE-friendly student.**
   - TinyViT achieves 83.1% top-1 with 21M params — compact enough for FHE.
   - Its IN-21k pretrained features may transfer to medical tasks.

4. **No paper studies the KD temperature / polynomial degree tradeoff under FHE.**
   - Higher KD temperature → better soft-label transfer → more FHE-friendly student.
   - The interaction between temperature, polynomial degree, and FHE accuracy is unstudied.

5. **No paper optimizes KD specifically for FHE constraints (depth budget, SIMD packing).**
   - CaPriDe avoids exp in the encrypted KL loss but doesn't jointly optimize depth budget.
   - A depth-aware KD loss (penalizing multiplicative depth in student activations) is novel.

---

## Part 6: GPU-Accelerated FHE Feasibility

### Current Accelerators

| System | Target HW | FHE Scheme | Speedup vs CPU | Status |
|--------|----------|-----------|---------------|--------|
| Cheddar | GPU (V100) | CKKS | ~40–100× | Published (USENIX 2024) |
| Cerium | GPU (A100) | CKKS+BFV | ~50–200× | arXiv 2024 |
| FIDESlib | GPU (H100) | CKKS | ~100–400× | arXiv 2024 |

### Latency Estimates for TinyViT-21M on H100 (Extrapolated)

Assumptions:
- TinyViT-21M: 12 transformer layers, embed_dim=448, 7 heads
- Each layer: ~8 CKKS multiplication levels (after polynomial conversion)
- Total: ~96 levels → needs n=2^16 at 128-bit security
- Bootstrapping every 30 levels → 3 bootstrapping operations

| Operation | CPU Estimate | FIDESlib GPU Estimate (100× speedup) |
|-----------|-------------|--------------------------------------|
| Single attention block | ~120s | ~1.2s |
| Full 12-layer ViT | ~1440s (24min) | ~14.4s |
| With bootstrapping (×3) | ~1500s | ~15s |
| Classification head | ~30s | ~0.3s |
| **Total inference** | **~1530s (25.5min)** | **~15.3s** |

At 15s per inference on H100, private ViT inference becomes **feasible for batch medical diagnostics**.
The H100 is available via AWS (p4de.24xlarge), Azure (NDmv4), and GCP (A3 instances).

### FHE Library Comparison

| Library | Language | GPU | CKKS | Bootstrap | Ease of Use |
|---------|----------|-----|------|-----------|-------------|
| TenSEAL | Python/C++ | ❌ | ✅ | ❌ | ⭐⭐⭐⭐ |
| OpenFHE | C++ | ⚠️ partial | ✅ | ✅ | ⭐⭐⭐ |
| HEAAN/HELib | C++ | ❌ | ✅ | ✅ | ⭐⭐ |
| FIDESlib | CUDA | ✅ | ✅ | ✅ | ⭐⭐ |
| Concrete-ML | Python | ❌ | ⚠️ | ❌ | ⭐⭐⭐⭐⭐ |

**Recommendation for this project**: Start with TenSEAL (prototyping), move to OpenFHE
(bootstrapping), target FIDESlib (benchmarking) for the final GPU experiments.
