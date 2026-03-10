# TinyViT (2022)

> Wu et al., "TinyViT: Fast Pretraining Distillation for Small Vision Transformers", ECCV 2022

## Why This is the Target FHE Student

TinyViT is the ideal candidate for FHE deployment because:
1. **Compact architecture**: 5M, 11M, 21M parameter variants
2. **Distillation-trained**: uses pre-stored teacher logits for efficient KD at scale
3. **Hierarchical design**: CNN-style patch merging reduces spatial resolution progressively
4. **Strong accuracy per parameter**: 79.1% top-1 (ImageNet) with only 5.4M params

For our FHE use case: TinyViT-5M adapted for CIFAR-10/MedMNIST with polynomial activations.

## Key Ideas

### 1. Sparsified Logit Distillation

Training large ViTs (teacher) for distillation requires storing their logits. TinyViT pre-stores:
```
teacher_logits[i] = teacher(image_i).topk(k)   # only top-k logits
```

At training time, the student loads logits from disk — no teacher inference needed online:
```python
logits = load_precomputed_logits(image_idx)   # fast lookup
loss = KL(student(image) || logits)
```

This makes distillation **training cheap** even with huge teachers (e.g., ViT-22B as teacher).

### 2. Hierarchical Architecture (4 Stages)

```
Input: 224×224
Stage 1: 56×56 feature map  (patch_size=4, dim=64)
Stage 2: 28×28 feature map  (patch merge 2×2, dim=128)
Stage 3: 14×14 feature map  (patch merge 2×2, dim=192)
Stage 4: 7×7 feature map    (patch merge 2×2, dim=384)
Head: global avg pool → classifier
```

Each stage is a stack of transformer blocks with window attention.

### 3. Window Attention

Instead of full attention over all patches (quadratic), each block attends within local windows:
```
Window size: 7×7 (for 14×14 feature map)
Each patch attends to 49 neighbors instead of 196 patches
```

Cost: O(N · w²) instead of O(N²). Important for FHE where attention ciphertext packing scales with N.

### 4. Convolutional FFN

TinyViT replaces the MLP FFN with a depthwise conv:
```
ConvFFN: FC (expand) → DWConv 3×3 → GELU → FC (compress)
```

For FHE: DWConv becomes a local polynomial operation. The 3×3 kernel is handled as a constant-weight plaintext convolution.

## FHE Adaptation Plan

Replace in each TinyViT block:
1. **Softmax** → Power-Softmax (degree-2) or L2Q (SAL-ViT)
2. **GELU** → PolyGELU degree-4 (BOLT) or degree-2 (PolyTransformer)
3. **Window size** → reduce if needed to fit ciphertext slot count

```
FHE depth estimate (TinyViT-5M adapted, 4 stages × 2 blocks each):
  Per block: attn(4) + GELU(2) = 6 levels
  8 blocks: 48 levels
  + patch embed + head: 2 levels
  Total: ~50 levels → fits in n=2^15
```

## Results

| Model | Params | ImageNet Top-1 | Notes |
|-------|--------|---------------|-------|
| TinyViT-5M | 5.4M | 79.1% | Target FHE student |
| TinyViT-11M | 11M | 81.4% | Borderline |
| TinyViT-21M | 21.2M | 83.1% | Teacher for 5M |
| DeiT-Tiny | 5.7M | 72.2% | Baseline comparison |

## Medical Imaging Relevance

TinyViT variants have been applied to:
- **DD-TinyViT**: pathology slide classification
- **TinyViT-Batten**: ophthalmology screening

For MedMNIST experiments, TinyViT-5M with PolyGELU is the target model.

## Implementation

See [`implementation.py`](implementation.py) for:
- `PatchMerging` — 2×2 spatial downsampling
- `WindowAttention` — local window attention with polynomial replacement
- `ConvFFN` — depthwise conv FFN
- `TinyViTBlock` — full block with configurable activation
- `TinyViT5M` — 4-stage hierarchical model
- `LogitDistillation` — pre-stored logit loading for offline KD
- `adapt_for_fhe()` — swap activations for polynomial alternatives

## Exercises

1. Train TinyViT-5M on CIFAR-10 with standard GELU — establish accuracy baseline
2. Run `adapt_for_fhe()` and fine-tune — measure accuracy drop from polynomial swap
3. Apply soft KD from a larger TinyViT-21M teacher after polynomial adaptation
4. Compare window attention packing vs. full attention for FHE ciphertext efficiency
