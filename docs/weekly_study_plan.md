# Weekly Study Plan: Private ViT Inference

Day-by-day schedule for Weeks 1 and 2. Structure: **Morning → Afternoon → Evening reflection**.
Each day connects theory to FHE-specific constraints.

---

## Week 1: KD Fundamentals + Polynomial ViT Training

### Day 1 (Monday): Transformers + FHE Incompatibilities

**Morning Reading (2h)**
- Alammar, "The Illustrated Transformer": https://jalammar.github.io/illustrated-transformer/
- Skim CryptoNets (ICML'16): https://proceedings.mlr.press/v48/gilad-bachrach16.html (focus on why they use x² instead of ReLU)

**Afternoon Implementation (3h)**
- Implement 2-head attention in NumPy from scratch (no autograd)
- Write `utils/depth_counter.py::attention_depth_cost()` — count mult levels for QKV + attn + out
- Extend to a full transformer block: MHA + LayerNorm + FFN

**Evening Reflection**
1. The standard softmax has an `exp` and a division. In CKKS, division is done by multiplying by the inverse. Why does `exp` not have a polynomial representation that works for all real numbers?
2. CryptoNets used x² (one multiplication). If we replace softmax with a degree-4 polynomial, how many extra multiplicative levels does that add per attention block?
3. What is the minimum number of slots needed (in CKKS SIMD packing) to encode a ViT-Tiny's full attention matrix (196 tokens × 192 dims)?

---

### Day 2 (Tuesday): ViT Architecture + Patch Encoding

**Morning Reading (2h)**
- Kilcher ViT walkthrough: https://youtu.be/TrdevFK_am4 (1h video)
- labml.ai annotated ViT: https://nn.labml.ai/transformers/vit/index.html (read alongside video)

**Afternoon Implementation (3h)**
- Fine-tune `vit_base_patch16_224` (timm) on CIFAR-10 with frozen backbone; log top-1 accuracy
- Visualize attention maps on 5 CIFAR-10 test images; save to `results/week1/attention_maps/`
- Run `experiments/baseline/vit_plaintext_baseline.py` and verify accuracy numbers

**Evening Reflection**
1. ViT uses a CLS token (appended to the patch sequence) for classification. Is this the only option? What would happen if we used global average pooling over all patch tokens instead? Which is more FHE-friendly?
2. Patch size 16×16 on 224×224 images gives 196 tokens. If we use 28×28 MedMNIST images with patch size 4×4, we get 49 tokens. How does this change the depth budget?
3. Positional encoding in ViT is learned (not sinusoidal). Is this a problem under FHE? Where do positional encodings get added, and does it cost multiplicative levels?

---

### Day 3 (Wednesday): Knowledge Distillation Theory + Loss Functions

**Morning Reading (2h)**
- Hinton et al. 2015: https://arxiv.org/abs/1503.02531 (foundational; read all)
- Jake Tae's mathematical derivation: https://jaketae.github.io/study/knowledge-distillation/
- CaPriDe paper (abstract + Sections 1-3): https://arxiv.org/abs/2304.10909

**Afternoon Implementation (3h)**
- Run `experiments/week1_kd_basics/02_deit_kd_pipeline.py`
- Try τ ∈ {2, 4, 8}; log accuracy gap between KD and non-KD DeiT-Tiny
- Add a function to compute L2 distillation loss (as an alternative to KL) and compare

**Evening Reflection**
1. Hinton's KD loss is: αL_CE + (1-α)·τ²·KL(σ(z_t/τ), σ(z_s/τ)). In FHE, the KL term requires log(p/q). CaPriDe replaces KL with L2. Does this change what "information" the student learns from the teacher?
2. Higher temperature τ softens the distribution. If τ→∞, the distribution becomes uniform. At what temperature does the teacher's distribution no longer provide useful information?
3. In our setting, the teacher runs in plaintext and the student will eventually run under FHE. Should we apply KD on the full task (final logits) or on intermediate representations? What are the FHE implications of feature-level KD?

---

### Day 4 (Thursday): DeiT + Distillation Token

**Morning Reading (2h)**
- DeiT paper: https://arxiv.org/abs/2012.12877 (Sections 1-4)
- lucidrains DistillWrapper: https://github.com/lucidrains/vit-pytorch#distillation (read the code)

**Afternoon Implementation (3h)**
- Run `experiments/week1_kd_basics/02_deit_kd_pipeline.py` fully
- Manually inspect the distillation token's attention pattern on 3 test images
- Add a variant that uses hard KD (argmax of teacher) instead of soft KD; compare accuracy

**Evening Reflection**
1. DeiT's distillation token is a separate learnable vector that attends to the same patch tokens as the CLS token. At test time, DeiT averages the CLS and distillation token predictions. Would averaging work under FHE? (Hint: averaging two ciphertexts is free — addition is cheap in CKKS.)
2. Hard KD (teacher argmax as a one-hot label) is simpler than soft KD. Is hard KD more or less FHE-friendly? (Consider: do we need to compute any non-polynomial operations on the teacher's output?)
3. In the encrypted head approach (HETAL, MedBlindTuner), only the final linear layer is encrypted. Can we apply KD specifically to the encrypted head? What would the training loop look like?

---

### Day 5 (Friday): Polynomial Activations + Polynomial ViT

**Morning Reading (2h)**
- BOLT paper (Sections 2-3 on BoBa-ViT): https://arxiv.org/abs/2307.07645
- Baruch et al. 2021 (full): https://arxiv.org/abs/2111.03362

**Afternoon Implementation (3h)**
- Run `experiments/week1_kd_basics/03_poly_gelu_kd.py`
- Train DeiT-Tiny with PolyGELU (degree 4); compare to standard GELU (with and without KD)
- Plot training curves: accuracy vs. epoch for all 4 variants in `03_poly_gelu_kd.py`

**Evening Reflection**
1. The degree-4 polynomial GELU approximation introduces approximation error. Does the error act like noise (additive, random) or like bias (systematic)? What happens to this error when it passes through the next layer?
2. Baruch et al. train the polynomial coefficients jointly with the model weights. Why might learned coefficients outperform fixed minimax approximations? What is the risk of jointly training coefficients? (Hint: can the model "cheat" by making the input range small?)
3. BOLT removes LayerNorm entirely and replaces it with a scalar normalization. What does LayerNorm compute, and why is division hard in CKKS? If we remove LayerNorm completely, what happens to training stability?

---

### Day 6 (Saturday): Polynomial Softmax + Attention Approximations

**Morning Reading (2h)**
- Power-Softmax paper: https://arxiv.org/abs/2410.09457 (Sections 1-3)
- Zimerman et al. ICML'24: https://proceedings.mlr.press/v235/zimerman24a.html (Section 3: RangeLoss)

**Afternoon Implementation (3h)**
- Run `experiments/week1_kd_basics/04_poly_softmax_study.py`
- Compare PowerSoftmax, MGFSoftmax, L2Q on a 1-block ViT; plot error vs. standard softmax
- Run `experiments/week1_kd_basics/05_layernorm_ablation.py`

**Evening Reflection**
1. Power-Softmax uses x^p / Σx^p. For p=2, this is x² / Σx² — always non-negative and sums to 1. But for large sequences (196 tokens), does this suffer from numerical issues? What if some x values are negative?
2. RangeLoss penalizes large attention logit values during training. This forces the attention logits into a small range, making them approximable by a low-degree polynomial. But attention logits encode semantic similarity. Could RangeLoss hurt the model's ability to attend to important tokens?
3. L2Q replaces softmax with L2 normalization (x / ||x||). This removes the "competition" between tokens (softmax makes scores sum to 1 via normalization across the sequence). How does this change the interpretation of attention? Under FHE, L2 norm requires a square root — how do we compute it polynomially?

---

### Day 7 (Sunday): Synthesis + Problem Statement Revision

**Morning Reading (1h)**
- Reread the 5 novelty gaps in `docs/kd_fhe_literature_review.md` (Part 5)
- Skim HETAL abstract + results table: https://proceedings.mlr.press/v202/lee23m.html

**Afternoon (3h)**
- Update `docs/paper_tiers.md` with any newly discovered papers
- Write a 2-paragraph "problem statement" in your research notes:
  - Para 1: What is the gap? (No paper combines KD + poly ViT + FHE + medical imaging)
  - Para 2: What will you do? (KD from ReLU ViT teacher → polynomial ViT student; evaluate under CKKS on MedMNIST)
- Identify the 3 biggest open questions from Week 1 experiments

**Evening Reflection**
1. After Week 1, what is your estimate of the accuracy gap: standard DeiT-Tiny → polynomial DeiT-Tiny (with KD)? Is it < 5%? < 3%? What experiments would give you a more precise estimate?
2. What is the bottleneck for making private ViT inference practical on medical images? Is it accuracy, latency, ease of implementation, or regulatory/compliance issues?
3. If you had to choose one experiment from Week 1 to present as a conference paper contribution, which would it be and why?

---

## Week 2: FHE Inference + Encrypted Evaluation

### Day 8 (Monday): CKKS Fundamentals + TenSEAL

**Morning Reading (2h)**
- `docs/ckks_parameter_guide.md` (read all)
- TenSEAL Tutorial 0 (basic operations): https://github.com/OpenMined/TenSEAL/blob/main/tutorials/Tutorial%200%20-%20Getting%20Started.ipynb
- TenSEAL Tutorial 1 (CKKS): https://github.com/OpenMined/TenSEAL/blob/main/tutorials/Tutorial%201%20-%20Training%20and%20Evaluation%20of%20Logistic%20Regression%20on%20Encrypted%20Data.ipynb

**Afternoon Implementation (3h)**
- Install TenSEAL: `pip install tenseal`
- Run `utils/ckks_helpers.py`'s `make_ckks_context()` at depth 8, 16, 32; time a matmul at each depth
- Encrypt a 192-dim vector; apply a linear layer (plaintext weights × ciphertext); decrypt; verify result

**Evening Reflection**
1. CKKS encodes floating point numbers with some error (scale factor Δ). If Δ = 2^40, approximately how many decimal places of precision does a CKKS ciphertext carry? Is this sufficient for ViT attention logits that range from, say, -10 to +10?
2. A CKKS multiplication consumes one "level" (reduces the modulus). After L multiplications, we run out of levels and need to bootstrap. What is bootstrapping, and why does it cost a lot of additional levels?
3. The diagonal matrix encoding allows matrix-vector multiply in O(n) ciphertext multiplications instead of O(n²). For a 192×192 attention projection, how many rotations are needed with BSGS vs. naive diagonal encoding?

---

### Day 9 (Tuesday): Encrypted Linear Layer + Depth Counting

**Morning Reading (1.5h)**
- Iron paper Section 3 (encoding strategies): https://arxiv.org/abs/2207.04872
- BOLT paper Section 4 (depth budget analysis): https://arxiv.org/abs/2307.07645

**Afternoon Implementation (3h)**
- Run `experiments/week2_fhe_inference/01_single_block_encryption.py`
- Vary polynomial degree for GELU (2, 4, 8) and measure noise after 1 block
- Use `utils/depth_counter.py::vit_depth_budget()` on TinyViT-5M, 11M, 21M

**Evening Reflection**
1. After running `01_single_block_encryption.py`, what is the wall-clock time for a single PolyTransformerBlock forward pass under CKKS? How does this scale with the number of layers?
2. The depth budget analysis shows TinyViT-21M needs ~96 levels. With n=2^16 and 128-bit security, what is the maximum log(q) available? Is there enough room for 96 levels?
3. Bootstrapping refreshes the noise budget but consumes ~20 additional levels (implementation-dependent). For a 12-layer model that needs 96 levels, where would you place bootstrapping operations to minimize total level consumption?

---

### Day 10 (Wednesday): MedMNIST Encrypted Head Baseline

**Morning Reading (1.5h)**
- MedBlindTuner paper: https://arxiv.org/abs/2401.09604 (all)
- HETAL results table (Table 2): https://proceedings.mlr.press/v202/lee23m.html

**Afternoon Implementation (3h)**
- Run `experiments/week2_fhe_inference/02_medmnist_fhe_baseline.py`
- Try DermaMNIST and BreastMNIST; log accuracy and latency
- Compare to MedBlindTuner's reported numbers (expect 1-3% deviation due to setup differences)

**Evening Reflection**
1. MedBlindTuner reports comparable accuracy to plaintext fine-tuning. Why is the accuracy so well preserved when only the head is encrypted? (Hint: the backbone runs in plaintext — all the hard computation is done unencrypted.)
2. The encrypted head approach gives a weaker privacy guarantee than fully encrypted inference. What specifically is the server able to learn about the client's data in the encrypted head setting? Is this acceptable for medical applications?
3. What would need to change to move from "encrypted head only" to "fully encrypted ViT"? List the 3 biggest technical obstacles.

---

### Day 11 (Thursday): KD + Polynomial ViT Integration

**Morning Reading (1.5h)**
- Baruch et al. 2021 Sections 3-4 (trainable polynomial activations): https://arxiv.org/abs/2111.03362
- Zimerman et al. ICML'24 Sections 4-5 (full pipeline): https://proceedings.mlr.press/v235/zimerman24a.html

**Afternoon Implementation (3h)**
- Combine Week 1 and Week 2 work: train polynomial DeiT-Tiny with KD, then run FHE head inference
- Run `experiments/week2_fhe_inference/03_kd_fhe_comparison.py`
- Document the 3-way comparison numbers in `benchmarks/three_way_comparison.json`

**Evening Reflection**
1. After running the 3-way comparison, what is the accuracy hierarchy? Expected: plaintext ViT-Base > encrypted head DeiT-Tiny > polynomial DeiT-Tiny + KD > polynomial DeiT-Tiny (no KD). Does the ordering hold?
2. The gap between plaintext ViT-Base and polynomial student+KD — is it primarily due to (a) the smaller student architecture, (b) the polynomial approximation of non-linearities, or (c) the FHE noise? How would you design an experiment to separate these contributions?
3. If you were to apply RangeLoss (from Zimerman et al.) during KD training, would you add it to the teacher loss, the student loss, or both? What would happen to the teacher's accuracy if you applied RangeLoss to the teacher?

---

### Day 12 (Friday): Depth Budget Optimization

**Morning Reading (1.5h)**
- CKKS parameter guide (reread): `docs/ckks_parameter_guide.md`
- BOLT supplementary material on depth optimization: https://arxiv.org/abs/2307.07645

**Afternoon Implementation (3h)**
- Profile polynomial DeiT-Tiny layer-by-layer: which layers consume the most levels?
- Try replacing the highest-depth operation with a cheaper approximation; retrain with KD
- Attempt to get total depth below 48 levels (fits in n=2^15, faster than n=2^16)

**Evening Reflection**
1. After profiling, which operation is the depth bottleneck: GELU approximation, L2Q attention, or LinearNorm? How many levels does each consume?
2. Reducing the polynomial degree of GELU from 4 to 2 saves levels but hurts accuracy. With KD, how much accuracy can you recover by using a degree-2 GELU + KD vs. degree-4 GELU without KD?
3. If you had access to a GPU FHE accelerator (like FIDESlib at 400× CPU speedup), at what total depth (levels) would the latency become < 1 minute? < 10 seconds?

---

### Day 13 (Saturday): FHE Noise Analysis

**Morning Reading (1.5h)**
- OpenFHE tutorial on noise growth: https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/advanced-ckks-bootstrapping.cpp
- CryptoNets Section 4 (noise budget analysis): https://proceedings.mlr.press/v48/gilad-bachrach16.html

**Afternoon Implementation (3h)**
- Add noise measurement to `01_single_block_encryption.py`: after each operation, decrypt and compare to plaintext; compute MSE
- Plot: MSE vs. number of operations, for different n (16384, 32768)
- Verify that the approximation error (from polynomial ops) dominates over CKKS noise

**Evening Reflection**
1. From the noise analysis, which contributes more to final prediction error: CKKS encryption noise or polynomial approximation error? Does this match your expectation?
2. At what depth does the CKKS noise become larger than the approximation error? Is this the point where we need to bootstrap?
3. If approximation error dominates, should we prioritize (a) reducing polynomial approximation error (higher degree → more depth) or (b) reducing CKKS noise (larger n → more levels)?

---

### Day 14 (Sunday): Week 2 Synthesis + Research Direction

**Morning (2h)**
- Reread all entries in `docs/reading_log.md`
- Review all experiment results in `results/week1/` and `results/week2/`

**Afternoon (3h)**
- Update the novelty gap table in `docs/kd_fhe_literature_review.md` with Week 2 findings
- Draft a 1-page "project summary" with:
  - Baseline numbers (3-way comparison)
  - Identified bottlenecks (depth, accuracy, latency)
  - Next experiments (Week 3: RangeLoss ablation, bootstrapping integration)
- Write 3 concrete research hypotheses for the next phase

**Evening Reflection**
1. After 2 weeks, what is your confidence that the proposed approach (KD → poly ViT → FHE inference on medical images) can work at a level competitive enough for a conference paper? What is the biggest remaining uncertainty?
2. The encrypted head approach (HETAL, MedBlindTuner) achieves high accuracy but weak privacy. Our fully encrypted approach achieves strong privacy but lower accuracy and higher latency. What threshold of accuracy drop and latency increase would a radiologist (or hospital) accept for a medical diagnostic tool?
3. Write down the title of the paper you would submit if everything works as planned. What would the main experimental comparison table contain?
