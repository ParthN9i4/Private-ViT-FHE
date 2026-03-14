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

---

### DeiT: Training Data-Efficient Image Transformers & Distillation Through Attention (2021)
**Authors**: Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou
**Link**: https://arxiv.org/abs/2012.12877
**Read date**: [ ]

#### Core Idea
Trains a competitive ViT on ImageNet-1k alone (no extra data) using knowledge distillation. Introduces a dedicated distillation token that learns to match a CNN teacher's output, complementing the class token.

#### Key Techniques
- Distillation token: separate learnable token whose prediction is supervised by the teacher (hard label) rather than soft logits
- Hard distillation (argmax of teacher) outperforms soft distillation in practice
- Data augmentation (RandAugment, Mixup, CutMix) compensates for less data than JFT-300M
- DeiT-Tiny (5.7M params) and DeiT-Small (22M) reach 72.2% and 79.8% top-1

#### Limitations / Open Problems
- Distillation token is specific to a fixed teacher architecture — not trivially transferable to other teacher types
- Still requires ImageNet-scale training for best results
- Attention maps use standard softmax — not FHE-friendly

#### Implementation Priority
- [x] High — DeiT is the baseline student architecture for Phase B KD experiments

#### Notes
Code: https://github.com/facebookresearch/deit. The distillation token is the key design choice — forward pass produces two outputs (cls token + distill token), each with its own loss term.

---

### A Methodology for Training Homomorphic Encryption Friendly Neural Networks (2021)
**Authors**: Moran Baruch, Nir Drucker, Lev Greenberg, Guy Moshkowich
**Link**: https://arxiv.org/abs/2111.03362
**Read date**: [ ]

#### Core Idea
A systematic KD pipeline to train HE-friendly neural networks. A standard (ReLU) teacher trains on plaintext; a polynomial-activation student learns from the teacher's soft logits. Applied to COVID-19 chest X-ray classification and CT-scan analysis.

#### Key Techniques
- Teacher: standard ResNet/CNN with ReLU trained on plaintext data
- Student: same architecture with ReLU replaced by trainable polynomial activations (degree 2 or 4)
- KD loss: KL divergence on teacher soft logits (temperature τ=4)
- Trainable activations: coefficients of the polynomial are learned jointly with weights
- Results: student within 1–2% of teacher on COVID-19 X-ray dataset

#### Limitations / Open Problems
- Only applied to CNNs, not transformers
- Polynomial degree kept low (≤4) — may not capture complex non-linearities in deeper models
- No actual FHE inference reported — only demonstrates FHE-friendliness of the trained model

#### Implementation Priority
- [x] High — direct inspiration for our KD pipeline; trainable polynomial activations are a key technique

#### Notes
This is the most directly relevant KD+FHE paper for our approach. The "trainable polynomial activation" idea is the critical contribution — instead of fixing the approximation polynomial, the student learns it.

---

### CaPriDe Learning: Confidential and Private Decentralized Learning Based on Encryption-Friendly Distillation Loss (CVPR 2023)
**Authors**: Nurbek Tastan, Karthik Nandakumar
**Link**: https://openaccess.thecvf.com/content/CVPR2023/papers/Tastan_CaPriDe_Learning_Confidential_and_Private_Decentralized_Learning_Based_on_Encryption-Friendly_CVPR_2023_paper.pdf
**Read date**: [ ]

#### Core Idea
Federated learning framework where clients train models locally and share encrypted soft labels (logits) instead of gradients. KD is used as the aggregation mechanism — a server-side student learns from encrypted client logits without ever seeing client data or model weights.

#### Key Techniques
- Clients encrypt their output logits using CKKS before sending to server
- Server-side KD: student trained on encrypted ensemble of client soft labels
- Encryption-friendly distillation loss: avoids exp/log operations in the encrypted domain (uses L2 distance instead of KL divergence)
- Supports heterogeneous client architectures (each client can have a different model)

#### Limitations / Open Problems
- Assumes honest-but-curious server (not fully malicious)
- Communication overhead: encrypted logits are much larger than gradients
- Only tested on CIFAR-10 and MNIST — not medical imaging

#### Implementation Priority
- [x] High — code available: https://github.com/tnurbek/capride-learning. The encrypted KD loss formulation is directly relevant.

#### Notes
The insight that KL divergence (requires log/exp) is incompatible with FHE, and L2 loss is a practical substitute, is critical for our encrypted training experiments.

---

### HETAL: Efficient Privacy-Preserving Transfer Learning with Homomorphic Encryption (ICML 2023, Oral)
**Authors**: Seewoo Lee, Garam Lee, Jung Woo Kim, Junbum Shin, Mun-Kyu Lee
**Link**: https://proceedings.mlr.press/v202/lee23m.html
**Read date**: [ ]

#### Core Idea
Encrypted transfer learning: a large pretrained model (ViT, BERT) runs as a plaintext feature extractor; only the final classification head is trained under CKKS encryption. Achieves competitive accuracy with practical latency (567–3442s) on medical and NLP datasets.

#### Key Techniques
- CKKS-encrypted training of a linear classification head on top of frozen plaintext features
- Encrypted gradient computation: forward pass in encrypted domain, backward pass approximated
- Amortized per-sample cost via batching (SIMD slots in CKKS)
- Evaluated on: MedMNIST (PathMNIST, DermaMNIST), AG News, 20 Newsgroups

#### Limitations / Open Problems
- Only the linear head is encrypted — the backbone runs in plaintext (weaker privacy guarantee)
- Latency 567–3442s makes real-time inference impractical
- No distillation — requires the full pretrained backbone at inference time

#### Implementation Priority
- [x] High — this is our primary baseline for the "encrypted head" approach. Code: https://github.com/CryptoLabInc/HETAL

#### Notes
HETAL is ICML 2023 Oral — significant credibility. The main limitation we aim to address: by distilling a compact fully-polynomial student, we can encrypt *all* layers (not just the head) with lower latency.

---

### MedBlindTuner: Towards Privacy-Preserving Fine-Tuning on Biomedical Images with Transformers and Fully Homomorphic Encryption (AAAI Workshop 2024)
**Authors**: Prajwal Panzade, Daniel Takabi, Zhipeng Cai
**Link**: https://arxiv.org/abs/2401.09604
**Read date**: [ ]

#### Core Idea
First work combining DeiT (data-efficient image transformer) with CKKS FHE for biomedical image classification. Uses a plaintext DeiT backbone with an encrypted classification head, fine-tuned on MedMNIST datasets. Achieves comparable accuracy to plaintext fine-tuning.

#### Key Techniques
- DeiT-Small pretrained on ImageNet-1k as backbone (plaintext)
- Classification head encrypted with CKKS (TenSEAL)
- Fine-tuning on MedMNIST: PathMNIST, DermaMNIST, RetinaMNIST
- Gradient updates computed in encrypted domain for the classification head only

#### Limitations / Open Problems
- Same fundamental limitation as HETAL: backbone is plaintext (not fully private)
- DeiT still uses standard softmax in attention — not FHE-friendly for full encryption
- Limited to small MedMNIST images (28×28)

#### Implementation Priority
- [x] High — directly replicable baseline. Code dir: `papers/medblindtuner/`

#### Notes
Our proposed contribution extends this: distill DeiT → compact polynomial ViT, then encrypt *all* layers. The accuracy gap introduced by distillation + polynomial activations is the key research question.

---

### Converting Transformers to Polynomial Form for Secure Inference Over Homomorphic Encryption (ICML 2024)
**Authors**: Itamar Zimerman, Moran Baruch, Nir Drucker, Gilad Ezov, Omri Soceanu, Lior Wolf
**Link**: https://proceedings.mlr.press/v235/zimerman24a.html
**Read date**: [ ]

#### Core Idea
First work to convert a full Vision Transformer to a purely polynomial form suitable for FHE inference. Systematically replaces all non-polynomial operations (softmax, LayerNorm, GELU) with polynomial approximations and introduces a training procedure that preserves accuracy.

#### Key Techniques
- Polynomial softmax: uses RangeLoss during training to constrain attention logit range, enabling low-degree approximation
- LinearNorm: replaces LayerNorm with a learnable linear affine transform (no sqrt/division)
- Polynomial GELU: degree-4 minimax approximation
- RangeLoss: auxiliary loss term penalizing large attention logit values → enables Taylor approximation in a bounded range
- First pure-FHE ViT inference on ImageNet-scale tasks

#### Limitations / Open Problems
- Accuracy drop vs. standard ViT (~3-5% on ImageNet)
- High multiplicative depth — requires bootstrapping for deep models
- Only tested on image classification, not medical imaging specifically

#### Implementation Priority
- [x] High — this is our main architectural blueprint. Code dir: `papers/poly_transformer/`

#### Notes
The RangeLoss idea is the critical innovation: by training the model to use a small range of attention logits, the softmax becomes well-approximated by a low-degree polynomial. This is what makes pure-FHE ViT inference tractable.

---

### PolyTransformer (IBM Research, 2024)
**Authors**: Moran Baruch, Nir Drucker, Lev Greenberg, Guy Moshkowich et al. (IBM Research)
**Link**: (IBM internal / workshop proceedings — TBD)
**Read date**: [ ]

#### Core Idea
IBM Research's polynomial transformer implementation, building on the Zimerman et al. ICML 2024 framework. Focuses on practical deployment aspects: CKKS parameter selection, depth budget management, and HW-software co-design.

#### Key Techniques
- Same polynomial operator conversions as Zimerman'24
- CKKS-specific optimizations: optimal packing strategies for attention matrices
- Depth budget analysis across different ViT scales

#### Limitations / Open Problems
- Limited public documentation (workshop paper)
- Implementation details not fully public

#### Implementation Priority
- [x] High — `papers/poly_transformer/` implements this approach

---

### AutoFHE: Automated Adaption of CNNs for Efficient Evaluation over FHE (2023)
**Authors**: Wei Ao, Vishnu Naresh Boddeti
**Link**: https://arxiv.org/abs/2307.11815
**Read date**: [ ]

#### Core Idea
Automated search for HE-friendly activation functions using evolutionary algorithms (EvoReLU). Searches over parameterized polynomial families to find activations that jointly optimize accuracy and multiplicative depth.

#### Key Techniques
- EvoReLU: evolutionary search over polynomial activation parameters
- Fitness function: accuracy × (1 / depth_cost)
- Applied to ResNets and VGGs — reduces depth by ~30% with <1% accuracy drop
- Can be combined with KD (evolutionary search on student model)

#### Limitations / Open Problems
- Search is expensive — requires many GPU hours
- Focused on CNNs, not transformers
- Polynomial degree fixed at search time

#### Implementation Priority
- [x] High — EvoReLU concept directly applicable to ViT activation search. Code dir: `papers/autofhe/`

---

### TinyViT: Fast Pretraining Distillation for Small Vision Transformers (2022)
**Authors**: Kan Wu, Jinnian Zhang, Huiyu Wang, Yanwei Fu, Wei Liu, Gui-Song Xia, Xinggang Wang
**Link**: https://arxiv.org/abs/2207.10666
**Read date**: [ ]

#### Core Idea
Distills large ViT teachers pretrained on ImageNet-21k to small student ViTs using sparse soft labels. Key insight: store only the top-k (k=1000) teacher logits per image rather than full 21k-class distributions, making large-scale KD tractable.

#### Key Techniques
- Sparse soft label storage: top-1000 logits saved per image × 13M images → 50GB instead of 1.2TB
- Teacher: ViT-L/16 pretrained on ImageNet-21k
- Student: TinyViT-5M, 11M, 21M (varying depth and width)
- TinyViT-21M: 83.1% top-1 on ImageNet — competitive with ViT-Base (81.8%)

#### Limitations / Open Problems
- Requires access to ImageNet-21k teacher logits (large preprocessing step)
- Not evaluated on medical imaging datasets
- Standard softmax in attention — not FHE-friendly as-is

#### Implementation Priority
- [x] High — TinyViT-21M is our target student architecture for Phase B. Code dir: `papers/tinyvit/`

#### Notes
Code: https://github.com/microsoft/Cream/tree/main/TinyViT. The sparse label storage trick makes it feasible to distill from IN-21k teachers without 1TB+ storage.

---

### SAL-ViT: Towards Latency Efficient Private Inference on ViT Using Selective Attention Head and Linearization (2023)
**Authors**: Yuke Zhang, Dake Chen, Souvik Kundu, Chenghao Li, Peter A. Beerel
**Link**: https://arxiv.org/abs/2310.04604
**Read date**: [ ]

#### Core Idea
MPC-based private ViT inference that selectively linearizes attention heads to reduce non-linear operations (and thus communication rounds in 2PC). Uses a saliency-based criterion to decide which heads can be linearized with minimal accuracy impact.

#### Key Techniques
- Head saliency scoring: gradient-based importance metric per attention head
- Selective linearization: low-salience heads use linear attention; high-salience heads keep softmax (evaluated via GC in 2PC)
- Structured pruning of attention heads for additional speedup
- Latency: 2.4× faster than baseline private ViT on DeiT-Small

#### Limitations / Open Problems
- MPC-based (2PC) — not pure FHE; requires online communication
- Linearization heuristic may not transfer across datasets
- Evaluated on ImageNet, not medical imaging

#### Implementation Priority
- [ ] Medium — useful for understanding selective linearization as a design principle. Code dir: `papers/sal_vit/`

---

### Power-Softmax: Towards Secure LLM Inference over Encrypted Data (2024)
**Authors**: Itamar Zimerman, Allon Adir, Ehud Aharoni, Nir Drucker, Gilad Ezov, Ariel Farkash, Lior Wolf
**Link**: https://arxiv.org/abs/2410.09457
**Read date**: [ ]

#### Core Idea
Replaces softmax with a power function (x^p / sum(x^p)) in transformers for FHE-compatible inference. Extends the polynomial transformer approach to large language models (up to 1B+ parameters, 32 layers) using GPT-2 and OPT model families.

#### Key Techniques
- Power-Softmax: p(x) = x^p / Σ(x^p) — avoids exp, computable with low-degree polynomials
- Trained end-to-end with Power-Softmax from scratch (no softmax pre-training needed)
- Evaluated on language modeling (perplexity) and classification benchmarks
- First polynomial LLMs at scale: 32-layer, 1.3B-parameter GPT-2-XL equivalent

#### Limitations / Open Problems
- Perplexity increase vs. standard softmax (~10-20% relative)
- Language modeling focus — not directly applicable to vision without adaptation
- Still requires RangeLoss or equivalent to bound logit magnitude

#### Implementation Priority
- [ ] Medium — Power-Softmax directly implemented in `papers/power_softmax/`. Key reference for attention approximation.

---

### Towards Deep Neural Network Training on Encrypted Data (CVPRW 2019)
**Authors**: Karthik Nandakumar, Nalini Ratha, Sharath Pankanti, Shai Halevi
**Link**: https://openaccess.thecvf.com/content_CVPRW_2019/papers/CV-COPS/Nandakumar_Towards_Deep_Neural_Network_Training_on_Encrypted_Data_CVPRW_2019_paper.pdf
**Read date**: [ ]

#### Core Idea
First demonstration of training (not just inference) a neural network on BGV-encrypted data. Simple fully-connected network on MNIST achieves 96% accuracy; 50× speed-up over naive implementation via smart data packing.

#### Key Techniques
- BGV scheme (bitwise encryption) — different from CKKS used in modern work
- Same-dimension data from multiple images packed into a single ciphertext (SIMD parallelism)
- Sigmoid activation (polynomial-approximable); quadratic loss (avoids cross-entropy log)
- Lookup tables for non-linear operations on bitwise-encrypted inputs

#### Limitations / Open Problems
- Training a 3-layer FC net on a minibatch of 60 images takes 1.5 days — not practical
- BGV is less efficient than CKKS for floating-point computations
- No transformers, no medical imaging

#### Implementation Priority
- [ ] Low — historical baseline only. Understand the conceptual framework; do not replicate.

#### Notes
Important for understanding the historical progression: CryptoNets'16 (inference) → Nandakumar'19 (training attempt) → HETAL'23 (practical encrypted transfer learning). The 1.5-day training time motivates why we focus on inference + encrypted head, not full encrypted training.

---

### MICCAI 2025 Paper 0621 (Pending)
**Authors**: TBD
**Link**: https://papers.miccai.org/miccai-2025/paper/0621_paper.pdf
**Read date**: [ ]

#### Core Idea
TBD — not yet indexed (as of 2026-03). Likely related to privacy-preserving medical image analysis or FHE-based inference on medical ViTs.

#### Key Techniques
- TBD after reading

#### Implementation Priority
- [ ] TBD — assess after reading

#### Notes
URL noted for future reference. Re-check indexing quarterly.
