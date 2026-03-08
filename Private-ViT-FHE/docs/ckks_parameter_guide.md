# CKKS Parameter Selection Guide

Quick reference for choosing CKKS parameters for ViT inference experiments.

---

## Key Parameters

| Parameter | Symbol | What it controls |
|-----------|--------|-----------------|
| Polynomial degree | `n` | Ring dimension — security and ciphertext size |
| Coefficient modulus | `q` | Noise budget = multiplicative depth capacity |
| Scale | `Δ` | Precision of encoded floats |
| Security level | λ | Typically 128-bit |

---

## Depth Budget Formula

For a CKKS computation with `L` multiplication levels:

```
q = q_0 * q_1 * ... * q_L
```

Each `q_i` is roughly `log2(Δ) + log2(Δ)` bits for a rescaling level. In practice:
- Each multiplication + rescaling costs ~40-60 bits
- `q_0` (special modulus) is ~60 bits
- Target: `log2(q) = 60 + L * 40`

---

## ViT Depth Budget (Rough Estimate)

For a 6-layer ViT with:
- QKV projection: 1 level
- Attention (linear approximation): 2 levels
- MLP (GELU approx, degree 4): 3 levels
- LayerNorm approx: 2 levels
- Per layer subtotal: ~8 levels
- 6 layers: ~48 levels
- Final linear + softmax: +5 levels
- **Total: ~53 multiplicative levels**

This requires `n = 2^15` (32768) or `n = 2^16` (65536) depending on security.

---

## Practical Settings (TenSEAL)

```python
import tenseal as ts

# For moderate depth (≤ 30 levels), small model
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=16384,
    coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 40, 60]  # 7 levels
)
context.global_scale = 2**40

# For deep ViT (≤ 50 levels)
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=32768,
    coeff_mod_bit_sizes=[60] + [40]*49 + [60]  # 49 levels
)
context.global_scale = 2**40
```

---

## Encoding Strategies for ViT

### Batch Encoding (SIMD)
- CKKS supports `n/2` slots per ciphertext
- For `n=16384`: 8192 slots
- Pack a full 196-token sequence (14×14 patches, dim=64) in one ciphertext if `196 * 64 ≤ 8192`

### Diagonal Matrix Encoding (for MatMul)
- Encodes matrix-vector products using diagonal rotations
- Cost: `d` rotations for a `d×d` matrix
- Used in BOLT and Iron for attention

### Baby-step Giant-step (BSGS)
- Reduces rotation count from `d` to `O(√d)`
- Important for large attention dimensions

---

## Security Reference

From the HomomorphicEncryption.org standard:

| n | Max log(q) | Security (bits) |
|---|-----------|-----------------|
| 2^13 (8192) | 218 | 128 |
| 2^14 (16384) | 438 | 128 |
| 2^15 (32768) | 881 | 128 |
| 2^16 (65536) | 1761 | 128 |

For ViT experiments: `n = 2^15` is the practical minimum for deep models.
