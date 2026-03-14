# Paper Tiers: Implementation Priority

All papers ranked by implementation priority. Each entry includes: arxiv/DOI link,
code repo (✅ = working, ⚠️ = partial/requires work, ❌ = no public code),
FHE library used, and one-line relevance summary.

---

## Tier 1: Implement Now

These papers are directly relevant to the core pipeline. Read, reproduce key results,
and integrate into experiments.

| Paper | Year | Venue | Code | FHE Library | Relevance |
|-------|------|-------|------|-------------|-----------|
| [CryptoNets](https://proceedings.mlr.press/v48/gilad-bachrach16.html) | 2016 | ICML | ✅ (SEAL) | BFV/CKKS | First NN on FHE; baseline depth budget |
| [BOLT](https://arxiv.org/abs/2307.07645) | 2023 | IEEE S&P | ✅ [repo](https://github.com/inpluslab/bolt) | CKKS (TenSEAL) | Bootstrapping-free BoBa-ViT; main architecture target |
| [DeiT](https://arxiv.org/abs/2012.12877) | 2021 | ICML | ✅ [repo](https://github.com/facebookresearch/deit) | N/A (plaintext) | Distillation token; primary student architecture |
| [TinyViT](https://arxiv.org/abs/2207.10666) | 2022 | ECCV | ✅ [repo](https://github.com/microsoft/Cream/tree/main/TinyViT) | N/A (plaintext) | Sparse KD from IN-21k; compact student for FHE |
| [Baruch et al.](https://arxiv.org/abs/2111.03362) | 2021 | arXiv | ⚠️ partial | CKKS-compatible | KD from ReLU teacher → poly student on medical data; closest prior work |
| [CaPriDe](https://openaccess.thecvf.com/content/CVPR2023/papers/Tastan_CaPriDe_Learning_Confidential_and_Private_Decentralized_Learning_Based_on_Encryption-Friendly_CVPR_2023_paper.pdf) | 2023 | CVPR | ✅ [repo](https://github.com/tnurbek/capride-learning) | CKKS | Encrypted KD loss; L2 vs. KL in FHE domain |
| [HETAL](https://proceedings.mlr.press/v202/lee23m.html) | 2023 | ICML (Oral) | ✅ [repo](https://github.com/CryptoLabInc/HETAL) | CKKS (HEaaN) | Encrypted classification head on MedMNIST; primary baseline |
| [MedBlindTuner](https://arxiv.org/abs/2401.09604) | 2024 | AAAI-W | ⚠️ partial | CKKS (TenSEAL) | DeiT + CKKS on biomedical images; closest to our target |
| [Zimerman et al.](https://proceedings.mlr.press/v235/zimerman24a.html) | 2024 | ICML | ⚠️ partial | CKKS-compatible | First polynomial ViT; RangeLoss + LinearNorm blueprint |
| [PolyTransformer](papers/poly_transformer/) | 2024 | Workshop | ❌ | CKKS (custom) | IBM polynomial ViT; depth budget management |
| [SAL-ViT](https://arxiv.org/abs/2310.04604) | 2023 | ICCV | ⚠️ partial | 2PC (not FHE) | Selective linearization heuristic for attention heads |
| [AutoFHE](https://arxiv.org/abs/2307.11815) | 2023 | arXiv | ✅ [repo](https://github.com/HungYiHo/AutoFHE) | CKKS-compatible | EvoReLU — evolutionary polynomial activation search |

---

## Tier 2: Read Carefully

These papers inform the approach but do not need full reproduction. Understand the key
techniques and extract relevant ideas.

| Paper | Year | Venue | Code | FHE Library | Relevance |
|-------|------|-------|------|-------------|-----------|
| [HEAR](https://arxiv.org/abs/2104.09164) | 2021 | arXiv | ❌ | CKKS + GC | Hybrid FHE+GC baseline; understand trade-offs |
| [Iron](https://arxiv.org/abs/2207.04872) | 2022 | NeurIPS | ❌ | CKKS + GC | First private BERT; degree-27 GELU; BSGS encoding |
| [Power-Softmax](https://arxiv.org/abs/2410.09457) | 2024 | arXiv | ⚠️ partial | CKKS-compatible | Polynomial softmax for LLMs; directly in `papers/power_softmax/` |
| [MGF-Softmax](https://arxiv.org/abs/2302.01817) | 2023 | arXiv | ❌ | CKKS-compatible | Gaussian mixture approximation of softmax |
| [NEXUS](https://arxiv.org/abs/2205.09001) | 2022 | arXiv | ❌ | CKKS | Private ViT with improved depth management |
| [Nandakumar et al.](https://openaccess.thecvf.com/content_CVPRW_2019/papers/CV-COPS/Nandakumar_Towards_Deep_Neural_Network_Training_on_Encrypted_Data_CVPRW_2019_paper.pdf) | 2019 | CVPRW | ❌ | BGV | First DNN training on encrypted data; historical baseline |
| [MedMNIST v2](https://arxiv.org/abs/2110.14795) | 2021 | NeurIPS-D | ✅ [repo](https://github.com/MedMNIST/MedMNIST) | N/A | Standard medical imaging benchmark |

---

## Tier 3: Skim

Read abstract + introduction + results table only. Extract any single applicable technique.

| Paper | Year | Venue | Code | Relevance |
|-------|------|-------|------|-----------|
| [THE-X](https://arxiv.org/abs/2204.03459) | 2022 | arXiv | ❌ | First pure FHE ViT; superseded by Zimerman'24 |
| THOR | 2023 | arXiv | ❌ | Packed attention for private ViT |
| MOAI | 2023 | arXiv | ❌ | DeiT-S under FHE; packing strategies |
| [MPCViT](https://arxiv.org/abs/2211.13955) | 2022 | arXiv | ❌ | MPC-based private ViT; NAS for MPC cost |
| PriViT | 2023 | arXiv | ❌ | Private ViT via secret sharing + GC |
| [LHE-Transformer](papers/lhe_transformer/) | 2024 | Workshop | ❌ | Leveled HE for transformers |
| [AutoPrivacy](https://arxiv.org/abs/2204.03459) | 2024 | arXiv | ❌ | NAS with FHE depth as constraint |
| AESPA | 2022 | arXiv | ❌ | Polynomial attention expansion |
| HEMET | 2022 | arXiv | ❌ | HE-friendly CNN; packed activation |
| LIME | 2024 | arXiv | ❌ | Low-depth image model for FHE |

---

## Tier 4: Survey Only

Read title and abstract. Do not implement.

| Paper | Year | Relevance |
|-------|------|-----------|
| [Cheddar](https://arxiv.org/abs/2407.12019) | 2024 | GPU FHE accelerator (V100); ~100× speedup |
| Cerium | 2024 | GPU FHE accelerator (A100); CKKS+BFV |
| FIDESlib | 2024 | GPU FHE library (H100); bootstrapping on GPU |
| [HETAL Softmax](https://arxiv.org/abs/2410.11184) | 2024 | Efficient homomorphic softmax (separate from HETAL) |
| Gou et al. KD Survey | 2021 | Survey: 40+ KD methods; use as reference |

---

## Novelty Gap Table

Maps each paper to the 4 key axes of our research.
✅ = addressed, ⚠️ = partially, ❌ = not addressed.

| Paper | KD | Poly-ViT | Pure FHE | Medical Imaging |
|-------|:--:|:--------:|:--------:|:---------------:|
| CryptoNets (2016) | ❌ | ❌ | ✅ | ❌ |
| Nandakumar (2019) | ❌ | ❌ | ✅ | ❌ |
| HEAR (2021) | ❌ | ❌ | ❌ | ❌ |
| DeiT (2021) | ✅ | ❌ | ❌ | ❌ |
| NEXUS (2022) | ❌ | ❌ | ✅ | ❌ |
| THE-X (2022) | ❌ | ❌ | ✅ | ❌ |
| TinyViT (2022) | ✅ | ❌ | ❌ | ❌ |
| Iron (2022) | ❌ | ❌ | ❌ | ❌ |
| MPCViT (2022) | ❌ | ⚠️ | ❌ | ❌ |
| AutoFHE (2023) | ❌ | ❌ | ❌ | ❌ |
| BOLT (2023) | ❌ | ✅ | ✅ | ❌ |
| SAL-ViT (2023) | ❌ | ⚠️ | ❌ | ❌ |
| CaPriDe (2023) | ✅ | ❌ | ⚠️ | ❌ |
| HETAL (2023) | ❌ | ❌ | ⚠️ | ✅ |
| Baruch et al. (2021) | ✅ | ⚠️ CNN | ❌ | ✅ |
| MedBlindTuner (2024) | ❌ | ❌ | ⚠️ | ✅ |
| Zimerman et al. (2024) | ❌ | ✅ | ✅ | ❌ |
| Power-Softmax (2024) | ❌ | ✅ | ✅ | ❌ |
| **Our Target** | **✅** | **✅** | **✅** | **✅** |

**No existing paper hits all four axes simultaneously.**

---

## Papers with Confirmed Working Code Repos

12 papers with confirmed usable repositories (as of 2026-03):

| # | Paper | Repo | FHE Library | Notes |
|---|-------|------|-------------|-------|
| 1 | BOLT | https://github.com/inpluslab/bolt | TenSEAL | Full inference pipeline |
| 2 | CaPriDe | https://github.com/tnurbek/capride-learning | TenSEAL | Full training + inference |
| 3 | HETAL | https://github.com/CryptoLabInc/HETAL | HEaaN | Encrypted head training |
| 4 | DeiT | https://github.com/facebookresearch/deit | PyTorch | Plaintext KD |
| 5 | TinyViT | https://github.com/microsoft/Cream/tree/main/TinyViT | PyTorch | Plaintext KD |
| 6 | AutoFHE | https://github.com/HungYiHo/AutoFHE | PyTorch | Evolutionary search |
| 7 | MedMNIST | https://github.com/MedMNIST/MedMNIST | N/A | Dataset only |
| 8 | lucidrains ViT | https://github.com/lucidrains/vit-pytorch | PyTorch | ViT + DeiT implementations |
| 9 | timm | https://github.com/huggingface/pytorch-image-models | PyTorch | Pretrained ViT zoo |
| 10 | torchdistill | https://github.com/yoshitomo-matsubara/torchdistill | PyTorch | Unified KD framework |
| 11 | TenSEAL | https://github.com/OpenMined/TenSEAL | CKKS (SEAL) | FHE Python bindings |
| 12 | OpenFHE | https://github.com/openfheorg/openfhe-development | CKKS/BFV | Full FHE with bootstrapping |
