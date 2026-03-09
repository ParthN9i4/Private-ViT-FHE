# Reading Log

Track every paper read — key ideas, open questions, and implementation priority.

---

## Template

```
## [Paper Title] (Year)
**Authors**:
**Link**:
**Read date**:

### Core Idea
1-3 sentences on what the paper does.

### Key Techniques
- ...

### Limitations / Open Problems
- ...

### Implementation Priority
[ ] Low / [ ] Medium / [x] High

### Notes
...
```

---

## Papers

### CryptoNets (Gilad-Bachrach et al., 2016)
**Authors**: Ran Gilad-Bachrach, Nathan Dowlin, Kim Laine, Kristin Lauter, Michael Naehrig, John Wernsing
**Link**: https://proceedings.mlr.press/v48/gilad-bachrach16.html
**Read date**: [ ]

#### Core Idea
First demonstration of running a neural network on FHE-encrypted data using SEAL (BFV at the time). Used a simple CNN on MNIST.

#### Key Techniques
- Square activation (x²) instead of ReLU — polynomial, depth 1
- Scaled weights to reduce noise
- No division — avoided softmax, used argmax post-decryption

#### Limitations / Open Problems
- Only works on toy models (MNIST CNN)
- ~250s per inference at the time
- No attention mechanism — predates transformers

#### Implementation Priority
- [x] High — start here, understand baseline noise budget

---

### HEAR: Human Action Recognition via Neural Networks on Homomorphically Encrypted Data (2021)
**Authors**: Yongsoo Song et al.
**Link**: https://arxiv.org/abs/2104.09164
**Read date**: [ ]

#### Core Idea
Hybrid protocol: use FHE for linear layers, garbled circuits for non-linearities (ReLU). Avoids polynomial approximation overhead.

#### Key Techniques
- Linear layers (matmul) in CKKS ciphertext domain
- ReLU evaluation using garbled circuits (2PC)
- Communication-efficient protocol for switching between FHE and GC

#### Limitations / Open Problems
- Requires online communication rounds (not non-interactive)
- Garbled circuit latency adds up for deep networks
- Not pure FHE — 2-party protocol

#### Implementation Priority
- [ ] Medium — useful for understanding hybrid approaches

---

### Iron: Private Inference on Transformers (NeurIPS 2022)
**Authors**: Meng Hao, Hanxiao Chen, Tianwei Zhang et al.
**Link**: https://arxiv.org/abs/2207.04872
**Read date**: [ ]

#### Core Idea
First private inference system for BERT-scale transformers. Handles attention (softmax), LayerNorm, and GELU via polynomial approximation + 2PC hybrid.

#### Key Techniques
- Softmax: composite polynomial approximation via domain decomposition
- GELU: degree-27 minimax polynomial approximation
- LayerNorm: approximate via scale-invariant formulation
- Packs multiple attention heads into single CKKS ciphertext

#### Limitations / Open Problems
- BERT, not ViT — sequence over tokens, not image patches
- Still requires bootstrapping for deep models
- 14.4s per inference for BERT-base (promising but not real-time)

#### Implementation Priority
- [x] High — directly applicable to ViT attention

---

### BOLT: Privacy-Preserving, Accurate and Efficient Inference for Transformers (IEEE S&P 2023)
**Authors**: Yichen Xu, Mengze Li, Xunyuan Yin, Junzuo Lai, Jian Weng
**Link**: https://arxiv.org/abs/2307.07645
**Read date**: [ ]

#### Core Idea
Bootstrapping-free private inference for vision transformers. Uses a modified ViT architecture (BoBa-ViT) that is FHE-friendly, achieving practical latency on CIFAR-10 and Tiny-ImageNet.

#### Key Techniques
- Removes softmax: replaces with scaled dot-product without exp (linear attention variant)
- Removes LayerNorm: replaces with a learnable scaling factor
- GELU replaced with degree-4 polynomial during training (co-design)
- No bootstrapping required — fits in a single leveled CKKS computation
- Packed attention via diagonal matrix encoding

#### Limitations / Open Problems
- Accuracy drops vs standard ViT (ImageNet is hard)
- Only tested on small-scale datasets
- Linear attention loses global context modeling

#### Implementation Priority
- [x] High — main target for this repo

---

### THE-X / AutoPrivacy (2022-2024)
**Authors**: Various
**Link**: https://arxiv.org/abs/2204.03459
**Read date**: [ ]

#### Core Idea
Automated search for FHE-friendly model architectures. Treats FHE constraints (depth, polynomial degree) as architecture search objectives alongside accuracy.

#### Key Techniques
- NAS (Neural Architecture Search) with FHE-cost as a constraint
- Polynomial activation search
- Joint optimization of accuracy and multiplicative depth

#### Limitations / Open Problems
- NAS is expensive — not easy to reproduce without large compute
- Results are model-family specific

#### Implementation Priority
- [ ] Low — understand the approach, don't implement full NAS

---

### LHE-Transformer (2024)
**Authors**: To be added
**Link**: [ ]
**Read date**: [ ]

#### Core Idea
Leveled HE (no bootstrapping) applied to transformer inference. Carefully manages noise budget across all 12 layers.

#### Key Techniques
- TBD after reading

#### Implementation Priority
- [ ] Medium
