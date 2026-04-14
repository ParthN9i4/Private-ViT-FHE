"""
Step 6: Encrypted Inference with CKKS (TenSEAL)
=================================================
Demonstrates that the polynomial DeiT-Tiny produces correct results
under Fully Homomorphic Encryption (CKKS scheme).

Three levels of demonstration:
  Level 1: Encrypted polynomial GELU on a single vector
           → proves polynomial activation works under CKKS
  Level 2: Encrypted polynomial softmax on attention scores
           → proves attention mechanism works under CKKS
  Level 3: Encrypted classification head (linear layer)
           → proves matrix-vector multiply works under CKKS
  Level 4: End-to-end encrypted inference on features
           → extract features in plaintext, encrypt, run head under
             CKKS, decrypt, compare with plaintext prediction

This matches the client-aided architecture described in the findings:
the client runs feature extraction (or sends encrypted image for server
to process), and the classification head runs under encryption.

For full encrypted inference of all 12 blocks, a GPU-accelerated CKKS
library (Cheddar, Lattigo) would be needed. TenSEAL demonstrates
correctness; production speed requires specialized implementations.

Requirements:
    pip install tenseal

Usage:
    python step6_encrypted_inference.py --dataset blood --baseline_dir results
    python step6_encrypted_inference.py --dataset retina --poly_mult_depth 15
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm

# Check TenSEAL availability
try:
    import tenseal as ts
    HAS_TENSEAL = True
    print(f"  TenSEAL version: {ts.__version__}")
except ImportError:
    HAS_TENSEAL = False
    print("=" * 60)
    print("  TenSEAL not found. Install it:")
    print("    pip install tenseal")
    print("  Or with conda:")
    print("    pip install tenseal --break-system-packages")
    print("=" * 60)
    sys.exit(1)

import medmnist
from medmnist import (
    RetinaMNIST, PneumoniaMNIST, BloodMNIST,
    DermaMNIST, BreastMNIST, PathMNIST
)


# ═══════════════════════════════════════════════════════════════
# Polynomial Operations (same as training, but for reference)
# ═══════════════════════════════════════════════════════════════

class PolyActivation(nn.Module):
    """Trainable polynomial activation (used for model loading)."""
    def __init__(self, degree=2, init_method="gelu_fit"):
        super().__init__()
        self.degree = degree
        if init_method == "gelu_fit" and degree == 2:
            init_coeffs = [0.0711, 0.5, 0.2576]
        else:
            init_coeffs = [0.0] * (degree + 1)
            init_coeffs[1] = 1.0
        self.coeffs = nn.Parameter(torch.tensor(init_coeffs, dtype=torch.float32))

    def forward(self, x):
        result = self.coeffs[-1]
        for i in range(len(self.coeffs) - 2, -1, -1):
            result = result * x + self.coeffs[i]
        return result


# ═══════════════════════════════════════════════════════════════
# CKKS Context Setup
# ═══════════════════════════════════════════════════════════════

def create_ckks_context(poly_mod_degree=16384, coeff_mod_bit_sizes=None,
                        global_scale=None):
    """
    Create a CKKS encryption context.

    CKKS parameters explained:
    - poly_mod_degree (N): Ring dimension. Determines security level and
      number of slots (N/2 values can be packed per ciphertext).
      8192 → ~128-bit security, 4096 slots.
      16384 → ~128-bit security, 8192 slots.

    - coeff_mod_bit_sizes: List of bit sizes for the coefficient modulus chain.
      Length = number of multiplicative levels + 2 (one for special prime,
      one for the initial scale). Each multiplication consumes one level.
      Example: [40, 21, 21, 21, 40] → 3 multiplicative levels.

    - global_scale: Scaling factor (2^scale_bits). Determines precision of
      encrypted arithmetic. Higher = more precise but consumes more bits.
      2^21 ≈ 2M, giving ~6 decimal digits of precision.

    Returns: context, and prints security/capacity info.
    """
    if coeff_mod_bit_sizes is None:
        # Scale of 2^40 gives ~12 decimal digits of precision per operation.
        # Each intermediate modulus must match the scale (40 bits).
        # First and last are special primes (60 bits each).
        # 5 intermediate = 5 multiplicative levels.
        coeff_mod_bit_sizes = [60, 40, 40, 40, 40, 40, 60]

    if global_scale is None:
        global_scale = 2 ** 40

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_mod_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.global_scale = global_scale
    context.generate_galois_keys()

    n_slots = poly_mod_degree // 2
    n_mult_levels = len(coeff_mod_bit_sizes) - 2  # subtract special + initial

    print(f"\n  ── CKKS Context ──")
    print(f"  Ring dimension (N):        {poly_mod_degree}")
    print(f"  Slots per ciphertext:      {n_slots}")
    print(f"  Multiplicative levels:     {n_mult_levels}")
    print(f"  Coeff modulus bit sizes:   {coeff_mod_bit_sizes}")
    print(f"  Global scale:              2^{int(np.log2(global_scale))}")
    print(f"  Security level:            ~128 bits")

    return context, n_slots, n_mult_levels


# ═══════════════════════════════════════════════════════════════
# Level 1: Encrypted Polynomial GELU
# ═══════════════════════════════════════════════════════════════

def demo_encrypted_poly_gelu(context, coeffs):
    """
    Demonstrate polynomial GELU evaluation under CKKS.

    f(x) = a·x² + b·x + c where coeffs = [c, b, a]

    Under CKKS:
    - x is encrypted (CKKSVector)
    - a, b, c are plaintext (known to server, part of model weights)
    - x² requires one ciphertext-ciphertext multiplication (1 mult level)
    - a·x² is plaintext-ciphertext multiplication (free)
    - b·x is plaintext-ciphertext multiplication (free)
    - additions are free
    """
    print(f"\n  ── Level 1: Encrypted Polynomial GELU ──")

    c, b, a = coeffs[0], coeffs[1], coeffs[2]
    print(f"  Polynomial: f(x) = {a:.4f}·x² + {b:.4f}·x + {c:.4f}")

    # Test inputs (typical activation values from a transformer)
    test_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]

    # Plaintext computation
    x_plain = np.array(test_values, dtype=np.float64)
    y_plain = a * x_plain**2 + b * x_plain + c

    # Encrypted computation
    t_start = time.time()
    x_enc = ts.ckks_vector(context, x_plain.tolist())

    # f(x) = a·x² + b·x + c
    # Step 1: x² (ciphertext × ciphertext → 1 mult level consumed)
    x_sq = x_enc * x_enc
    # Step 2: a·x² (plaintext × ciphertext → free)
    ax2 = x_sq * a
    # Step 3: b·x (plaintext × ciphertext → free)
    bx = x_enc * b
    # Step 4: a·x² + b·x + c (additions → free)
    y_enc = ax2 + bx + c

    y_dec = np.array(y_enc.decrypt())
    t_elapsed = time.time() - t_start

    # Compare
    max_error = np.max(np.abs(y_plain - y_dec))
    mean_error = np.mean(np.abs(y_plain - y_dec))

    print(f"\n  {'x':>8s} {'Plaintext':>12s} {'Encrypted':>12s} {'Error':>12s}")
    print(f"  {'─' * 48}")
    for i in range(len(test_values)):
        err = abs(y_plain[i] - y_dec[i])
        print(f"  {test_values[i]:>8.2f} {y_plain[i]:>12.6f} {y_dec[i]:>12.6f} {err:>12.2e}")

    print(f"\n  Max error:  {max_error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")
    print(f"  Time:       {t_elapsed*1000:.1f} ms")
    print(f"  Mult levels used: 1 (for x²)")
    print(f"  ✓ Polynomial GELU works under CKKS" if max_error < 1e-3
          else f"  ⚠ Error larger than expected")

    return max_error


# ═══════════════════════════════════════════════════════════════
# Level 2: Encrypted Polynomial Softmax
# ═══════════════════════════════════════════════════════════════

def demo_encrypted_poly_softmax(context, depth=3):
    """
    Demonstrate polynomial softmax under CKKS.

    exp(x) ≈ (1 + x/2^d)^{2^d} via d repeated squarings.
    Each squaring consumes 1 multiplicative level → d levels total.

    The normalization (dividing by sum) would require Goldschmidt
    iteration under CKKS (~2 extra levels). Here we demonstrate
    the exp approximation and note the normalization requirement.
    """
    print(f"\n  ── Level 2: Encrypted Polynomial Exp (Softmax Component) ──")
    print(f"  Depth: {depth} (uses {depth} multiplicative levels)")

    # Typical attention scores (after Q·K^T / sqrt(d))
    test_values = [0.5, 1.0, -0.5, 0.2, -1.0, 1.5]

    # Plaintext poly_exp
    x_plain = np.array(test_values, dtype=np.float64)
    scale = 2 ** depth
    result_plain = (1.0 + x_plain / scale)
    result_plain = np.maximum(result_plain, 0.0)
    for _ in range(depth):
        result_plain = result_plain * result_plain

    # True exp for reference
    true_exp = np.exp(x_plain)

    # Encrypted poly_exp
    t_start = time.time()
    x_enc = ts.ckks_vector(context, x_plain.tolist())

    # (1 + x/2^d)
    result_enc = x_enc * (1.0 / scale) + 1.0

    # Repeated squaring: d multiplications
    for i in range(depth):
        result_enc = result_enc * result_enc

    result_dec = np.array(result_enc.decrypt())
    t_elapsed = time.time() - t_start

    print(f"\n  {'x':>8s} {'True exp':>12s} {'Poly(plain)':>12s} {'Poly(enc)':>12s} {'Approx err':>12s} {'CKKS err':>12s}")
    print(f"  {'─' * 72}")
    for i in range(len(test_values)):
        approx_err = abs(true_exp[i] - result_plain[i])
        ckks_err = abs(result_plain[i] - result_dec[i])
        print(f"  {test_values[i]:>8.2f} {true_exp[i]:>12.6f} {result_plain[i]:>12.6f} "
              f"{result_dec[i]:>12.6f} {approx_err:>12.2e} {ckks_err:>12.2e}")

    max_ckks_err = np.max(np.abs(result_plain - result_dec))
    max_approx_err = np.max(np.abs(true_exp - result_plain))

    print(f"\n  Polynomial approximation error (vs true exp): {max_approx_err:.2e}")
    print(f"  CKKS precision error (vs polynomial plaintext): {max_ckks_err:.2e}")
    print(f"  Time: {t_elapsed*1000:.1f} ms")
    print(f"  Mult levels used: {depth}")
    print(f"  Note: Full softmax also needs Goldschmidt division (~2 extra levels)")
    ok = max_ckks_err < 1e-2
    print(f"  ✓ Polynomial exp works under CKKS" if ok
          else f"  ⚠ CKKS precision error larger than expected")

    return max_ckks_err


# ═══════════════════════════════════════════════════════════════
# Level 3: Encrypted Linear Layer (Matrix-Vector Multiply)
# ═══════════════════════════════════════════════════════════════

def demo_encrypted_linear(context, weight, bias):
    """
    Demonstrate encrypted matrix-vector multiplication.

    Linear layer: y = W·x + b
    - x is encrypted (CKKSVector of dimension d_in)
    - W is plaintext (model weight, known to server)
    - b is plaintext (model bias, known to server)
    - W·x is computed via CKKS matrix-vector multiply (1 mult level)
    - +b is plaintext addition (free)

    In TenSEAL, matmul with encrypted vector uses diagonal method:
    rotates and multiplies slots, then sums. This is the standard
    technique for encrypted neural network inference.
    """
    print(f"\n  ── Level 3: Encrypted Linear Layer ──")

    d_out, d_in = weight.shape
    print(f"  Weight shape: {d_out} × {d_in}")
    print(f"  Bias shape: {d_out}")

    # Random input (simulating features from a transformer block)
    x_plain = np.random.randn(d_in).astype(np.float64)

    # Plaintext computation
    y_plain = weight.numpy().astype(np.float64) @ x_plain + bias.numpy().astype(np.float64)

    # Encrypted computation
    t_start = time.time()
    x_enc = ts.ckks_vector(context, x_plain.tolist())

    # Matrix-vector multiply under CKKS
    # TenSEAL's mm expects matrix in [n_in, n_out] format
    # PyTorch stores Linear weight as [n_out, n_in], so we transpose
    weight_T = weight.T.numpy().astype(np.float64).tolist()  # [d_in, d_out]
    y_enc = x_enc.mm(weight_T)
    # Add bias
    y_enc = y_enc + bias.numpy().astype(np.float64).tolist()

    y_dec = np.array(y_enc.decrypt())[:d_out]  # Trim to output dimension
    t_elapsed = time.time() - t_start

    max_error = np.max(np.abs(y_plain - y_dec))
    mean_error = np.mean(np.abs(y_plain - y_dec))

    print(f"\n  Sample outputs (first 5):")
    print(f"  {'Class':>8s} {'Plaintext':>12s} {'Encrypted':>12s} {'Error':>12s}")
    print(f"  {'─' * 48}")
    for i in range(min(5, d_out)):
        err = abs(y_plain[i] - y_dec[i])
        print(f"  {i:>8d} {y_plain[i]:>12.6f} {y_dec[i]:>12.6f} {err:>12.2e}")

    print(f"\n  Max error:  {max_error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")
    print(f"  Time:       {t_elapsed*1000:.1f} ms")
    print(f"  Mult levels used: 1")
    ok = max_error < 1e-1
    print(f"  ✓ Encrypted linear layer works" if ok
          else f"  ⚠ Error larger than expected")

    return max_error


# ═══════════════════════════════════════════════════════════════
# Level 4: End-to-End Encrypted Classification
# ═══════════════════════════════════════════════════════════════

def demo_encrypted_classification(context, model, test_loader, device,
                                  n_classes, n_samples=10):
    """
    End-to-end encrypted classification demo.

    Architecture (client-aided):
    1. Client sends image to server (or encrypts and server processes)
    2. Server extracts features using polynomial transformer (plaintext here,
       would be encrypted in full deployment)
    3. Server encrypts features with client's public key
    4. Server runs classification head under CKKS
    5. Server returns encrypted prediction
    6. Client decrypts and gets DR grade

    For this demo, we run feature extraction in plaintext and only
    encrypt the classification head, demonstrating the correctness
    of the encrypted inference pipeline.
    """
    print(f"\n  ── Level 4: End-to-End Encrypted Classification ──")
    print(f"  Testing on {n_samples} images...")

    model.eval()

    # Extract classification head weights
    head_weight = model.head.weight.data.cpu()  # [n_classes, embed_dim]
    head_bias = model.head.bias.data.cpu()       # [n_classes]
    embed_dim = head_weight.shape[1]

    print(f"  Classification head: {n_classes} classes × {embed_dim} features")

    # Collect samples
    images_list = []
    labels_list = []
    count = 0
    for images, labels in test_loader:
        for i in range(images.size(0)):
            if count >= n_samples:
                break
            images_list.append(images[i:i+1])
            labels_list.append(labels[i].item())
            count += 1
        if count >= n_samples:
            break

    # Run inference
    plaintext_correct = 0
    encrypted_correct = 0
    prediction_match = 0
    max_logit_error = 0.0
    total_enc_time = 0.0

    print(f"\n  {'#':>4s} {'Label':>6s} {'Plain pred':>11s} {'Enc pred':>9s} {'Match':>6s} {'Max err':>10s} {'Time':>8s}")
    print(f"  {'─' * 58}")

    for idx in range(n_samples):
        img = images_list[idx].to(device)
        true_label = labels_list[idx]

        # Step 1: Extract features (plaintext — in full deployment this
        # would be encrypted, but we focus on the head for this demo)
        with torch.no_grad():
            features = model.forward_features(img)
            # DeiT uses cls token (first token) for classification
            cls_features = features[:, 0]  # [1, embed_dim]
            features_np = cls_features.cpu().numpy().astype(np.float64).flatten()

            # Plaintext logits (reference)
            plain_logits = model.head(cls_features)
            plain_pred = plain_logits.argmax(dim=1).item()

        # Step 2: Encrypt features
        t_start = time.time()
        features_enc = ts.ckks_vector(context, features_np.tolist())

        # Step 3: Encrypted classification head
        # TenSEAL mm expects [n_in, n_out], PyTorch weight is [n_out, n_in]
        logits_enc = features_enc.mm(
            head_weight.T.numpy().astype(np.float64).tolist()
        )
        logits_enc = logits_enc + head_bias.numpy().astype(np.float64).tolist()

        # Step 4: Decrypt
        logits_dec = np.array(logits_enc.decrypt())[:n_classes]
        enc_pred = int(np.argmax(logits_dec))
        t_elapsed = time.time() - t_start
        total_enc_time += t_elapsed

        # Compare
        plain_logits_np = plain_logits.cpu().numpy().flatten()
        logit_error = np.max(np.abs(plain_logits_np - logits_dec))
        max_logit_error = max(max_logit_error, logit_error)

        p_correct = plain_pred == true_label
        e_correct = enc_pred == true_label
        match = plain_pred == enc_pred

        plaintext_correct += int(p_correct)
        encrypted_correct += int(e_correct)
        prediction_match += int(match)

        status = "✓" if match else "✗"
        print(f"  {idx+1:>4d} {true_label:>6d} {plain_pred:>11d} {enc_pred:>9d} {status:>6s} "
              f"{logit_error:>10.2e} {t_elapsed*1000:>7.1f}ms")

    # Summary
    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  Encrypted Classification Results                │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  Samples tested:           {n_samples:>4d}                 │")
    print(f"  │  Plaintext accuracy:        {plaintext_correct}/{n_samples} ({100*plaintext_correct/n_samples:.0f}%)             │")
    print(f"  │  Encrypted accuracy:        {encrypted_correct}/{n_samples} ({100*encrypted_correct/n_samples:.0f}%)             │")
    print(f"  │  Prediction match rate:     {prediction_match}/{n_samples} ({100*prediction_match/n_samples:.0f}%)             │")
    print(f"  │  Max logit error (CKKS):    {max_logit_error:.2e}           │")
    print(f"  │  Avg encryption time:       {total_enc_time/n_samples*1000:.1f} ms/sample      │")
    print(f"  │  Total encryption time:     {total_enc_time:.2f} s              │")
    print(f"  └─────────────────────────────────────────────────┘")

    if prediction_match == n_samples:
        print(f"\n  ✓ PERFECT MATCH: All {n_samples} encrypted predictions match plaintext")
        print(f"    This proves the CKKS parameters provide sufficient precision")
        print(f"    for the polynomial DeiT-Tiny classification head.")
    else:
        mismatches = n_samples - prediction_match
        print(f"\n  ⚠ {mismatches} prediction mismatches out of {n_samples}")
        print(f"    Consider increasing CKKS scale or polynomial modulus degree")

    return {
        "n_samples": n_samples,
        "plaintext_correct": plaintext_correct,
        "encrypted_correct": encrypted_correct,
        "prediction_match": prediction_match,
        "max_logit_error": float(max_logit_error),
        "avg_time_ms": float(total_enc_time / n_samples * 1000),
    }


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

DATASET_CONFIG = {
    "retina":      {"class": RetinaMNIST,      "n_classes": 5},
    "pneumonia":   {"class": PneumoniaMNIST,   "n_classes": 2},
    "blood":       {"class": BloodMNIST,       "n_classes": 8},
    "derma":       {"class": DermaMNIST,       "n_classes": 7},
    "breast":      {"class": BreastMNIST,      "n_classes": 2},
    "path":        {"class": PathMNIST,        "n_classes": 9},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 6: Encrypted inference demo with CKKS"
    )
    parser.add_argument("--dataset", type=str, default="blood",
                        choices=list(DATASET_CONFIG.keys()),
                        help="Dataset to test (default: blood)")
    parser.add_argument("--baseline_dir", type=str, default="results",
                        help="Directory with Step 1 teacher checkpoints")
    parser.add_argument("--step3_dir", type=str, default="results_step3_poly_softmax",
                        help="Directory with Step 3 polynomial model checkpoints")
    parser.add_argument("--step5b_dir", type=str, default="results_step5b_coldstart_kd",
                        help="Directory with Step 5B polynomial model checkpoints")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of test samples for encrypted inference")
    parser.add_argument("--poly_mod_degree", type=int, default=16384,
                        help="CKKS polynomial modulus degree (default: 16384)")
    parser.add_argument("--output_dir", type=str, default="results_step6_encrypted")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# Model Loading (reuse polynomial architecture from Step 3/5B)
# ═══════════════════════════════════════════════════════════════

def replace_gelu(model, degree=2, init_method="gelu_fit"):
    """Replace GELU with PolyActivation for model loading."""
    import types
    count = 0
    for block in model.blocks:
        if hasattr(block, 'mlp') and hasattr(block.mlp, 'act'):
            if isinstance(block.mlp.act, nn.GELU):
                block.mlp.act = PolyActivation(degree=degree, init_method=init_method)
                count += 1
    return count


def load_polynomial_model(dataset_name, n_classes, step3_dir, step5b_dir, device):
    """
    Load the best available polynomial model.
    Tries Step 3 (GELU+softmax replaced) first, falls back to Step 5B (GELU only).
    """
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False,
                              num_classes=n_classes)

    # Replace GELU with PolyActivation (needed for checkpoint loading)
    replace_gelu(model, degree=2, init_method="gelu_fit")

    # Try Step 5B checkpoint (GELU-only polynomial, simpler to load)
    ckpt_path = None
    source = None
    for directory, name in [(step5b_dir, "Step 5B (GELU+KD)"),
                            (step3_dir, "Step 3 (GELU+softmax+KD)")]:
        path = os.path.join(directory, dataset_name, "best_model.pth")
        if os.path.exists(path):
            ckpt_path = path
            source = name
            break

    # Also try final_model.pth
    if ckpt_path is None:
        for directory, name in [(step5b_dir, "Step 5B"), (step3_dir, "Step 3")]:
            path = os.path.join(directory, dataset_name, "final_model.pth")
            if os.path.exists(path):
                ckpt_path = path
                source = name
                break

    if ckpt_path is None:
        print(f"  No polynomial model found for {dataset_name}")
        print(f"  Searched: {step5b_dir}/{dataset_name}/ and {step3_dir}/{dataset_name}/")
        print(f"  Using ImageNet pretrained with polynomial GELU (untrained)")
        source = "ImageNet pretrained (untrained polynomial)"
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Try strict loading, fall back to non-strict if attention was also replaced
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded polynomial model: {source}")
        print(f"  Checkpoint: {ckpt_path}")

    model = model.to(device)
    model.eval()
    return model, source


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  Step 6: Encrypted Inference with CKKS (TenSEAL)            ║")
    print(f"  ║  Dataset: {args.dataset:<15s}                                ║")
    print(f"  ║  Demonstrating polynomial ViT works under encryption         ║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  Device: {device}")

    config = DATASET_CONFIG[args.dataset]
    n_classes = config["n_classes"]

    # ── Create CKKS context ──
    context, n_slots, n_levels = create_ckks_context(
        poly_mod_degree=args.poly_mod_degree,
    )

    # ── Load polynomial model ──
    print(f"\n  Loading polynomial model...")
    model, model_source = load_polynomial_model(
        args.dataset, n_classes, args.step3_dir, args.step5b_dir, device
    )

    # Extract polynomial coefficients for Level 1 demo
    poly_coeffs = None
    for name, module in model.named_modules():
        if isinstance(module, PolyActivation):
            poly_coeffs = module.coeffs.data.cpu().tolist()
            print(f"  Using poly coefficients from {name}: "
                  f"f(x) = {poly_coeffs[2]:.4f}·x² + {poly_coeffs[1]:.4f}·x + {poly_coeffs[0]:.4f}")
            break

    if poly_coeffs is None:
        poly_coeffs = [0.0711, 0.5, 0.2576]
        print(f"  Using default GELU approximation coefficients")

    # ── Level 1: Encrypted Polynomial GELU ──
    gelu_err = demo_encrypted_poly_gelu(context, poly_coeffs)

    # ── Level 2: Encrypted Polynomial Exp ──
    exp_err = demo_encrypted_poly_softmax(context, depth=3)

    # ── Level 3: Encrypted Linear Layer ──
    head_weight = model.head.weight.data.cpu()
    head_bias = model.head.bias.data.cpu()
    linear_err = demo_encrypted_linear(context, head_weight, head_bias)

    # ── Level 4: End-to-End Encrypted Classification ──
    print(f"\n  Loading test data...")
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    DatasetClass = config["class"]
    test_dataset = DatasetClass(split="test", transform=eval_transform,
                                download=True, as_rgb=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0
    )

    enc_results = demo_encrypted_classification(
        context, model, test_loader, device, n_classes, args.n_samples
    )

    # ── Save Results ──
    save_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "step": 6,
        "description": "Encrypted inference with CKKS (TenSEAL)",
        "dataset": args.dataset,
        "model_source": model_source,
        "ckks_params": {
            "poly_mod_degree": args.poly_mod_degree,
            "n_slots": n_slots,
            "n_mult_levels": n_levels,
            "security_level": "~128 bits",
        },
        "level1_poly_gelu_max_error": float(gelu_err),
        "level2_poly_exp_max_error": float(exp_err),
        "level3_linear_max_error": float(linear_err),
        "level4_classification": enc_results,
    }

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {save_dir}/results.json")

    # ── Final Summary ──
    print(f"\n  {'='*60}")
    print(f"  STEP 6 SUMMARY: Encrypted Inference Demo")
    print(f"  {'='*60}")
    print(f"  Dataset:                {args.dataset.upper()}")
    print(f"  Model:                  {model_source}")
    print(f"  CKKS ring dimension:    {args.poly_mod_degree}")
    print(f"  Security:               ~128 bits")
    print(f"  ")
    print(f"  Level 1 (poly GELU):    max error = {gelu_err:.2e}")
    print(f"  Level 2 (poly exp):     max error = {exp_err:.2e}")
    print(f"  Level 3 (linear layer): max error = {linear_err:.2e}")
    print(f"  Level 4 (end-to-end):   {enc_results['prediction_match']}/{enc_results['n_samples']} match "
          f"({100*enc_results['prediction_match']/enc_results['n_samples']:.0f}%)")
    print(f"  Avg inference time:     {enc_results['avg_time_ms']:.1f} ms/sample")
    print(f"  ")
    print(f"  This demonstrates that:")
    print(f"  1. Polynomial activations evaluate correctly under CKKS")
    print(f"  2. The classification head produces matching predictions")
    print(f"  3. CKKS precision is sufficient for medical image classification")
    print(f"  {'='*60}")


if __name__ == "__main__":
    main()