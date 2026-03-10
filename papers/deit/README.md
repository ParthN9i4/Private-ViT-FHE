# DeiT (2020)

> Touvron et al., "Training data-efficient image transformers & distillation through attention", ICML 2021

## Why This Matters for FHE

DeiT makes ViT training data-efficient by using **knowledge distillation (KD) from a CNN teacher**. The student learns from both labels and teacher predictions, producing compact models (DeiT-Tiny: 5.7M params) that are the natural candidates for FHE deployment. Smaller model → fewer FHE levels needed.

## Key Ideas

### 1. Distillation Token

DeiT adds a second learnable `[distill]` token alongside `[class]`:
```
Input sequence: [cls] [distill] patch_1 ... patch_N
```

- `[cls]` token: trained with standard cross-entropy loss against ground-truth labels
- `[distill]` token: trained to match the **teacher's predictions** (hard or soft)

At inference, only `[cls]` is used — the distillation token is discarded.

### 2. Hard vs. Soft Distillation

**Hard distillation**: `[distill]` matches the teacher's argmax (hard label).
```
L_hard = (1 - α) · CE(cls_out, y) + α · CE(distill_out, teacher_argmax(x))
```

**Soft distillation**: `[distill]` matches the full teacher probability distribution (KL divergence at temperature τ).
```
L_soft = (1 - α) · CE(cls_out, y) + α · τ² · KL(distill_out/τ || teacher_softmax(x)/τ)
```

Hard distillation is simpler and often works as well or better.

### 3. CNN Teacher Choice

The teacher is a RegNet or EfficientNet — a CNN, not a ViT. CNNs provide a complementary inductive bias (spatial locality) that helps the ViT student learn more efficiently.

## DeiT Model Sizes

| Model | Params | Top-1 (ImageNet) | FHE relevance |
|-------|--------|-----------------|---------------|
| DeiT-Tiny | 5.7M | 72.2% | Primary FHE target |
| DeiT-Small | 22M | 79.8% | Borderline for FHE |
| DeiT-Base | 86M | 81.8% | Teacher role only |

For FHE on CIFAR-10/MedMNIST, DeiT-Tiny is the target student.

## FHE Connection

1. **Model compression**: KD-trained DeiT-Tiny is compact enough to fit in CKKS noise budget
2. **Accuracy recovery**: after replacing GELU/softmax with polynomials, use KD to recover lost accuracy
3. **Distillation from poly-student teacher**: chain — ViT-Base teacher → DeiT-Tiny student → PolyStudent FHE

## Architecture (DeiT-Tiny)

```
Image: 224×224 (or 32×32 CIFAR) → 16×16 patches → 196 tokens
dim = 192, depth = 12, heads = 3, mlp_ratio = 4

[cls] + [distill] + patch embeddings → 12 transformer blocks → two heads
  → cls_head (CE loss)
  → distill_head (KD loss against teacher)
```

## Implementation

See [`implementation.py`](implementation.py) for:
- `DistillationToken` — adds distillation token to standard ViT
- `DeiTStudent` — DeiT-Tiny wrapper with two heads
- `DistillationLoss` — hard and soft distillation losses
- `train_with_distillation()` — full training loop on CIFAR-10

## Results on CIFAR-10

| Model | Accuracy | Notes |
|-------|----------|-------|
| DeiT-Tiny (scratch) | ? | Baseline |
| DeiT-Tiny (+ KD from ResNet-50) | ? | Target |
| DeiT-Tiny + PolyGELU (+ KD) | ? | FHE-ready |

## Exercises

1. Train DeiT-Tiny with and without KD — measure accuracy gap
2. Compare hard vs. soft distillation on CIFAR-10
3. Replace teacher with BoBa-ViT (already trained) — does FHE-teacher help?
4. Add PolyGELU to the student and retrain with KD — how much accuracy is lost vs. recovered?
