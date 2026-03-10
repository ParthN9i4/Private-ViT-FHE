# AutoFHE (2023)

> "AutoFHE: Automated Adaption of CNNs for Efficient Evaluation over FHE" — and related EvoReLU work

## Why This Matters

AutoFHE frames the problem of building FHE-friendly networks as an **evolutionary search** over per-layer polynomial activation functions. Instead of manually choosing degree-2 or degree-4 for every layer, EvoReLU searches for the best polynomial degree per layer jointly with accuracy — and the search converges to different degrees in different layers.

The key insight: **not all layers need the same polynomial degree**. Early layers can tolerate higher depth; later layers near the noise budget limit need lower degree.

## Key Ideas

### 1. EvoReLU — Per-Layer Polynomial Search

Each layer gets its own polynomial activation of learnable degree:
```
EvoReLU_l(x) = Σ_{k=0}^{d_l} c_{l,k} · x^k
```

The degree `d_l` is searched per layer. The search objective:
```
minimize  FHE_depth(d_1, ..., d_L)   subject to  accuracy ≥ target
```

Search algorithm: evolutionary strategy (CMA-ES or similar) over degree assignments.

### 2. Polynomial-Aware Training (implicit KD)

After the degree schedule is found:
1. Train a **ReLU teacher** to full accuracy
2. Train a **polynomial student** matching the degree schedule, using soft KD from the ReLU teacher

The KD loss:
```
L = CE(student, labels) + τ² · KL(student/τ || teacher/τ)
```

This is the **same as DeiT's soft distillation** — the FHE polynomial network is the student, the ReLU network is the teacher.

### 3. Mixed-Degree Depth Budget

The depth of a mixed-degree network:
```
FHE_depth = Σ_l ceil(log2(d_l))
```

Example mixed assignment for 6-layer ViT:
| Layer | Degree | Depth |
|-------|--------|-------|
| 1 | 4 | 2 |
| 2 | 4 | 2 |
| 3 | 2 | 1 |
| 4 | 2 | 1 |
| 5 | 2 | 1 |
| 6 | 2 | 1 |
| **Total** | | **8** |

vs. uniform degree-4 (depth 12) or uniform degree-2 (depth 6 but lower accuracy).

## Connection to This Repo

AutoFHE's polynomial-aware training is a drop-in complement to:
- **DeiT**: the ReLU DeiT-Tiny is the teacher; the polynomial DeiT-Tiny is the student
- **BoBa-ViT (BOLT)**: already uses degree-4 — AutoFHE would ask: is degree-2 enough for some layers?
- **PolyTransformer**: range-loss tightens input range; AutoFHE finds optimal degree given that range

## Implementation

See [`implementation.py`](implementation.py) for:
- `EvoReLU` — parameterized polynomial activation with learnable coefficients
- `MixedDegreeViT` — assign different polynomial degrees per block
- `polynomial_aware_training()` — KD loop: ReLU teacher → polynomial student

## Exercises

1. Train a 6-layer ViT on CIFAR-10 with degrees [4, 4, 4, 2, 2, 2] — compare to uniform-4 and uniform-2
2. Run the greedy degree search: start from all degree-4, greedily reduce each layer's degree if accuracy stays above threshold
3. Compare KD recovery: polynomial student with and without ReLU teacher
4. Plot accuracy vs. total FHE depth across degree assignments

## Open Questions

1. Do the same degree assignments that work for CIFAR-10 transfer to medical imaging datasets?
2. Can we replace evolutionary search with differentiable degree selection?
3. Is there a natural ordering (first vs. last layers) that explains which layers tolerate lower degree?
