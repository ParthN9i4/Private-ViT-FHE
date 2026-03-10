# Benchmark Experiments

Scripts for systematic latency/throughput benchmarking across FHE schemes and model configurations.

## Benchmark Dimensions

### 1. Scheme Comparison

| Scheme | Library | Use Case |
|--------|---------|----------|
| CKKS | TenSEAL | Approximate arithmetic (our main target) |
| CKKS | OpenFHE | Higher performance, more control |
| Concrete | Concrete-ML | sklearn/PyTorch compatible |

### 2. Model Size Sweep

Vary ViT depth and width while tracking:
- FHE inference latency
- Multiplicative depth used
- Accuracy on CIFAR-10

### 3. CKKS Parameter Sweep

For fixed BoBa-ViT:
- Vary `poly_modulus_degree`: 2^14, 2^15, 2^16
- Vary scale: 2^30, 2^40, 2^50
- Measure: latency, noise error, security level

## Files

- `run_benchmarks.py` — main benchmark runner (to be created)
- `scheme_comparison.py` — compare TenSEAL vs OpenFHE (to be created)
- `model_size_sweep.py` — vary ViT dimensions (to be created)

## Usage

```bash
# Run all benchmarks and save to benchmarks/ directory
python experiments/benchmarks/run_benchmarks.py --output benchmarks/results.json

# Quick smoke test (small model, few samples)
python experiments/benchmarks/run_benchmarks.py --quick
```

## Output Format

Results are saved to `benchmarks/` as JSON:

```json
{
  "timestamp": "2024-01-01T00:00:00",
  "hardware": {
    "cpu": "...",
    "ram_gb": 32
  },
  "results": [
    {
      "model": "boba_vit_small",
      "scheme": "ckks_tenseal",
      "poly_degree": 32768,
      "scale_bits": 40,
      "latency_s": 6.7,
      "levels_used": 50,
      "accuracy": 0.883
    }
  ]
}
```
