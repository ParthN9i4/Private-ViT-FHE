# Learning Roadmap: Private ViT Inference

6-phase structured path from transformer fundamentals to a working KD+FHE+ViT+Medical pipeline.
For each phase: what to read (with links), what to implement, key concepts to internalize.

---

## Phase 1: Transformer Foundations

**Goal**: Understand attention, positional encoding, and the transformer block from first principles.

### Read

| Resource | Link | Format |
|----------|------|--------|
| Jay Alammar, "The Illustrated Transformer" | https://jalammar.github.io/illustrated-transformer/ | Blog post |
| Lilian Weng, "Attention? Attention!" | https://lilianweng.github.io/posts/2018-06-24-attention/ | Blog post |
| Andrej Karpathy, "Let's build GPT from scratch" | https://youtu.be/kCc8FmEb1nY | YouTube video |
| d2l.ai Ch. 11: Attention Mechanisms | https://d2l.ai/chapter_attention-mechanisms-and-transformers/ | Textbook |
| Umar Jamil, "Attention is all you need" annotated | https://youtu.be/bCz4OMemCcA | YouTube video |

### Implement

- [ ] Multi-head attention from scratch in NumPy (no autograd): Q, K, V projections, softmax, output projection
- [ ] Add positional encoding (sinusoidal); verify attention patterns on a toy sequence
- [ ] Transformer encoder block: MHA + residual + LayerNorm + FFN
- [ ] Train a 2-layer transformer to classify simple sequences (e.g., bracket matching)

### Key Concepts to Gain

- Why softmax is used (normalize attention weights to sum to 1; prevent gradient explosion)
- The multiplicative depth cost of each operation when evaluated under CKKS
- What "masked" vs. "unmasked" attention means and why only encoder-style matters for classification

---

## Phase 2: Vision Transformer (ViT) Architecture

**Goal**: Understand how images are processed as sequences of patches in ViT.

### Read

| Resource | Link | Format |
|----------|------|--------|
| Dosovitskiy et al., "An Image is Worth 16×16 Words" | https://arxiv.org/abs/2010.11929 | Paper |
| Yannic Kilcher, ViT paper walkthrough | https://youtu.be/TrdevFK_am4 | YouTube video |
| labml.ai annotated ViT | https://nn.labml.ai/transformers/vit/index.html | Annotated code |
| lucidrains ViT PyTorch | https://github.com/lucidrains/vit-pytorch | Code |
| timm library (used for pretrained ViTs) | https://github.com/huggingface/pytorch-image-models | Code |

### Implement

- [ ] ViT-Tiny from scratch in PyTorch: patch embedding, CLS token, positional encoding, 6-layer encoder, MLP head
- [ ] Train on CIFAR-10 from scratch (~80% accuracy expected)
- [ ] Load `vit_base_patch16_224` from `timm` and fine-tune on CIFAR-10
- [ ] Visualize attention maps with `timm`'s `vit_base` on a sample image

### Key Concepts to Gain

- Patch embedding: why 16×16 patches? How does patch size affect sequence length and compute?
- Class token vs. global average pooling — which is more FHE-friendly and why
- Positional encoding: learned vs. sinusoidal — impact on FHE (learned is fine; sinusoidal needs range analysis)
- ViT depth vs. accuracy vs. FHE depth budget tradeoff

---

## Phase 3: Knowledge Distillation Theory

**Goal**: Understand the theory and practice of model compression via teacher-student KD.

### Read

| Resource | Link | Format |
|----------|------|--------|
| Hinton et al., "Distilling the Knowledge in a Neural Network" (2015) | https://arxiv.org/abs/1503.02531 | Paper |
| Intel Distiller documentation | https://nervanasystems.github.io/distiller/knowledge_distillation.html | Docs |
| Jake Tae, "A Mathematical Derivation of KD Loss" | https://jaketae.github.io/study/knowledge-distillation/ | Blog post |
| Neptune.ai, "Knowledge Distillation Survey" | https://neptune.ai/blog/knowledge-distillation | Survey blog |
| Gou et al., "Knowledge Distillation: A Survey" (2021) | https://arxiv.org/abs/2006.05525 | Survey paper |

### Implement

- [ ] KD loss function: αL_CE + (1-α)τ²·KL(σ(z_t/τ), σ(z_s/τ))
- [ ] Train: ResNet-50 teacher → ResNet-18 student on CIFAR-10; measure accuracy gap
- [ ] Sweep τ ∈ {2, 4, 8} and α ∈ {0.3, 0.5, 0.9}; plot accuracy vs. temperature
- [ ] Feature distillation variant: L2 loss on intermediate representations

### Key Concepts to Gain

- Temperature softens the teacher's distribution → richer gradient signal for the student
- Why soft labels carry more information than hard one-hot labels (dark knowledge)
- The KL divergence is the "right" loss for distribution matching — but it requires log/exp
- For FHE: KL divergence in the encrypted domain requires polynomial approximation of log; L2 is safer (CaPriDe insight)

---

## Phase 4: DeiT and TinyViT

**Goal**: Understand the specific KD mechanisms used for efficient ViT training.

### Read

| Resource | Link | Format |
|----------|------|--------|
| Touvron et al., "DeiT: Training Data-Efficient Image Transformers" | https://arxiv.org/abs/2012.12877 | Paper |
| DeiT official code | https://github.com/facebookresearch/deit | Code |
| Wu et al., "TinyViT: Fast Pretraining Distillation for Small ViTs" | https://arxiv.org/abs/2207.10666 | Paper |
| TinyViT official code | https://github.com/microsoft/Cream/tree/main/TinyViT | Code |
| lucidrains DistillWrapper | https://github.com/lucidrains/vit-pytorch#distillation | Code |

### Implement

- [ ] Add distillation token to ViT-Tiny: extra learnable token, separate prediction head
- [ ] Implement DeiT hard distillation loss: cross-entropy on distillation head output against teacher argmax
- [ ] Train DeiT-Tiny (from scratch) with ViT-Base teacher on CIFAR-10; compare to non-distilled DeiT-Tiny
- [ ] Implement sparse label storage: for each training image, store top-k (k=100) teacher logits
- [ ] TinyViT-style pretraining: use sparse labels for KD instead of full soft distributions

### Key Concepts to Gain

- The distillation token attends to image patches separately from the class token — what does it learn?
- Hard KD (argmax) vs. soft KD (full distribution) — which is more FHE-compatible?
- Sparse label storage: practical trick that makes large-scale KD feasible without TB of storage
- Why TinyViT-21M at 21M params reaches 83.1% top-1 — better than DeiT-Small (22M) at 79.8%

---

## Phase 5: Transfer Learning + Medical Imaging

**Goal**: Fine-tune compact ViTs on MedMNIST; understand domain adaptation challenges.

### Read

| Resource | Link | Format |
|----------|------|--------|
| HuggingFace, "Fine-tuning ViT for image classification" | https://huggingface.co/docs/transformers/tasks/image_classification | Tutorial |
| MedMNIST v2 paper | https://arxiv.org/abs/2110.14795 | Paper |
| MedMNIST official code + benchmarks | https://github.com/MedMNIST/MedMNIST | Code |
| timm fine-tuning guide | https://huggingface.co/docs/timm/training_script | Docs |
| HETAL paper (ICML'23) | https://proceedings.mlr.press/v202/lee23m.html | Paper |

### Implement

- [ ] Install MedMNIST: `pip install medmnist`; load RetinaMNIST, DermaMNIST, BreastMNIST
- [ ] Fine-tune `vit_base_patch16_224` (timm, ImageNet-21k pretrained) on RetinaMNIST; log ACC + AUC per class
- [ ] Fine-tune DeiT-Tiny (from Phase 4) on RetinaMNIST; compare to ViT-Base
- [ ] Distill ViT-Base teacher → DeiT-Tiny student on RetinaMNIST using Phase 4 DeiT KD pipeline
- [ ] Replicate MedBlindTuner: plaintext DeiT-Small backbone + encrypted linear head (TenSEAL)

### Key Concepts to Gain

- Domain gap: ImageNet pretraining → medical images; how much does the gap hurt?
- MedMNIST class imbalance: AUC is a better metric than accuracy for rare classes
- Why the classification head is the easiest to encrypt: it's a linear layer (matmul + bias)
- What accuracy drop to expect: plaintext ViT-Base → DeiT-Tiny → polynomial DeiT-Tiny

---

## Phase 6: Integrated KD + FHE Toolkit

**Goal**: Build a reproducible pipeline: polynomial ViT trained with KD, evaluated under CKKS.

### Read

| Resource | Link | Format |
|----------|------|--------|
| torchdistill (unified KD framework) | https://github.com/yoshitomo-matsubara/torchdistill | Code |
| KD_Lib (KD collection) | https://github.com/SforAiDl/KD_Lib | Code |
| Cheddar GPU FHE paper | https://arxiv.org/abs/2407.12019 | Paper |
| TenSEAL tutorials | https://github.com/OpenMined/TenSEAL/tree/main/tutorials | Notebooks |
| OpenFHE bootstrapping tutorial | https://github.com/openfheorg/openfhe-development/tree/main/src/pke/examples | Code |

### Implement

- [ ] Polynomial ViT pipeline: load DeiT-Tiny → replace all non-polynomial ops (softmax → L2Q, GELU → degree-4 poly, LayerNorm → LinearNorm)
- [ ] Train polynomial DeiT-Tiny with KD from ViT-Base teacher on RetinaMNIST using RangeLoss
- [ ] Depth budget analysis: count multiplicative levels needed for each layer (`utils/depth_counter.py`)
- [ ] FHE inference: encrypt a test batch; run forward pass under TenSEAL CKKS; decrypt predictions
- [ ] Benchmark: plaintext time | encrypted head time | full encrypted time on BreastMNIST

### Key Concepts to Gain

- Why polynomial conversion reduces accuracy and how KD mitigates this
- The RangeLoss mechanism: constraining attention logit magnitude to enable Taylor approximation
- CKKS depth budget: how many levels remain after each transformer block
- The latency hierarchy: plaintext (ms) → encrypted head (seconds) → full FHE (minutes/hours)
- Path to practical private inference: what needs to improve (GPU FHE, better approximations, model architecture)
