# FHE Inference Experiments

End-to-end private inference experiments — running ViT forward passes on encrypted inputs using CKKS.

## Planned Experiments

### 1. Single-Layer FHE Forward Pass

Start with one transformer block encrypted end-to-end:
- Encrypt a single patch sequence
- Run linear attention in CKKS
- Measure noise growth and remaining levels
- Decrypt and check against plaintext output

### 2. Full BoBa-ViT FHE Inference

Port the BOLT BoBa-ViT architecture to TenSEAL:
- Use degree-4 GELU approximation
- Linear attention (no softmax)
- Scalar norm (no LayerNorm)
- Target: reproduce BOLT's 6.7s on CIFAR-10

### 3. Depth Budget Validation

Empirically verify the theoretical depth budget:
- Add level probes after each layer
- Compare to `depth_counter.py` static analysis
- Identify if bootstrapping is actually needed

## Files

- `single_layer_fhe.py` — single transformer block on encrypted input (to be created)
- `boba_vit_fhe.py` — full BoBa-ViT in FHE (to be created)
- `depth_probe.py` — level tracking during forward pass (to be created)

## Dependencies

```python
import tenseal as ts
# See requirements.txt for full dependencies
```

## Usage

```bash
# Run single-layer experiment
python experiments/fhe_inference/single_layer_fhe.py

# Run full inference benchmark
python experiments/fhe_inference/boba_vit_fhe.py --dataset cifar10 --n_samples 10
```

## Expected Results

| Experiment | Expected Latency | Levels Used |
|------------|-----------------|-------------|
| Single layer (FHE) | ~1.1s | ~8 |
| Full BoBa-ViT (FHE) | ~6-8s | ~50 |
| Full BoBa-ViT (plaintext) | ~50ms | N/A |
