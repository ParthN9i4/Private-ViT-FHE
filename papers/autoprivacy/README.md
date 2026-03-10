# AutoPrivacy / THE-X (2022–2024)

> Various. "THE-X: Privacy-Preserving Transformer Inference with Homomorphic Encryption" and related AutoPrivacy work (2022–2024)

## Why This Matters

Rather than manually designing FHE-friendly architectures (like BOLT's BoBa-ViT), AutoPrivacy / THE-X uses **Neural Architecture Search (NAS)** with FHE constraints as optimization objectives. This automates the co-design process.

## Key Ideas

### 1. FHE-Aware NAS

Standard NAS optimizes:
- Accuracy on target task
- Latency / FLOPs on hardware

FHE-aware NAS adds:
- **Multiplicative depth** as a constraint
- **Polynomial-friendly activation** selection
- **Ciphertext packing efficiency** as an objective

### 2. Polynomial Activation Search

Instead of manually choosing GELU vs ReLU vs square:
- Search over polynomial degrees: `{2, 4, 8, 16}`
- Jointly optimize accuracy + depth budget
- Learn per-layer polynomial coefficients during training

### 3. THE-X Approach

THE-X focuses on making pre-trained transformers (BERT, ViT) FHE-compatible post-hoc:
1. Start with a pre-trained model
2. Replace non-polynomial ops with approximations
3. Fine-tune to recover accuracy
4. Minimize depth increase from approximations

Key insight: **fine-tuning after approximation substitution recovers most accuracy** without full retraining.

## Comparison to BOLT

| Aspect | BOLT | AutoPrivacy/THE-X |
|--------|------|--------------------|
| Design approach | Manual co-design | Automated NAS / post-hoc |
| Starting point | Train from scratch | Pre-trained model |
| Flexibility | Fixed architecture | Adaptable to existing models |
| Compute cost | Low | High (NAS is expensive) |

## Relevance to ViT

For this repo, AutoPrivacy/THE-X is mainly a reference for:
- Understanding the trade-off space between depth budget and accuracy
- Seeing which activation functions NAS actually selects
- Post-hoc approximation as an alternative to training from scratch

## Implementation Notes

Full NAS is expensive and not the focus of this repo. Relevant pieces to implement:
1. **Depth-constrained training**: add depth penalty to training loss
2. **Polynomial activation fine-tuning**: start from pre-trained ViT, fine-tune with poly activations
3. **Activation sweep**: benchmark accuracy for degree-2, 4, 8 activations

## Open Questions

1. What polynomial degree does NAS converge to for ViT attention on CIFAR-10?
2. Can post-hoc approximation (fine-tune only) match from-scratch co-design (BOLT)?
3. Is there a depth budget below which accuracy degrades sharply (phase transition)?

## Status

- [ ] Read THE-X paper
- [ ] Read AutoPrivacy NAS paper
- [ ] Implement activation sweep experiment (see `experiments/approximations/`)
- [ ] Run depth-constrained fine-tuning on pre-trained ViT
