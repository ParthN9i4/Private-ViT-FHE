"""
Polynomial softmax replacements for FHE-compatible attention.

Implements and benchmarks:
  - PowerSoftmax: (Q@K.T)^p / sum — depth 1
  - MGFSoftmax: Taylor-series exp approximation — depth ceil(log2(degree))
  - L2Q: Learnable quadratic per head — depth 1

Paper references:
  Power-Softmax: various FHE transformer papers
  MGF-Softmax: "MGF-Softmax: Privacy-Preserving Attention via MGF Approximation"
  L2Q: SAL-ViT (see papers/sal_vit/)
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Power-Softmax
# ---------------------------------------------------------------------------

class PowerSoftmax(nn.Module):
    """
    Attention weights via element-wise power of dot-products.

        A_ij = (q_i · k_j)^p / Σ_k (q_i · k_k)^p

    FHE depth: ceil(log2(p))
      p=2 → 1 level
      p=4 → 2 levels

    Args:
        p: Power (must be a positive even integer for FHE-friendliness)
        temperature: Scale applied to logits before powering (tightens range)
    """

    def __init__(self, p: int = 2, temperature: float = 1.0):
        super().__init__()
        assert p >= 1 and p % 2 == 0, "p must be a positive even integer"
        self.p = p
        self.temperature = temperature

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                scale: float = 1.0) -> torch.Tensor:
        """
        Args:
            q, k, v: (B, H, N, head_dim)
            scale: sqrt(head_dim) scaling factor
        Returns:
            attended values: (B, H, N, head_dim)
        """
        logits = (q @ k.transpose(-2, -1)) * scale / self.temperature  # (B, H, N, N)
        # Power: element-wise logits^p
        attn = logits ** self.p
        # Normalize per row
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
        return attn @ v

    def fhe_depth(self) -> int:
        return math.ceil(math.log2(self.p))


# ---------------------------------------------------------------------------
# MGF-Softmax
# ---------------------------------------------------------------------------

class MGFSoftmax(nn.Module):
    """
    Softmax approximation via truncated Taylor series of exp.

        exp(x) ≈ Σ_{k=0}^{degree} x^k / k!

    Then softmax(x_i) = exp(x_i) / Σ_j exp(x_j)  using the approximation.

    FHE depth: ceil(log2(degree))
      degree=7 → 3 levels
      degree=3 → 2 levels

    For best accuracy: apply to logits in a bounded range [-R, R].
    Combine with range-loss training (papers/poly_transformer/) for optimal results.

    Args:
        degree: Truncation degree of Taylor series
    """

    def __init__(self, degree: int = 7):
        super().__init__()
        self.degree = degree
        # Precompute factorials
        factorials = [1.0]
        for i in range(1, degree + 1):
            factorials.append(factorials[-1] * i)
        self.register_buffer("inv_factorials",
                             torch.tensor([1.0 / f for f in factorials]))

    def poly_exp(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate truncated exp(x) via Taylor series."""
        result = torch.zeros_like(x)
        x_power = torch.ones_like(x)  # x^0
        for k in range(self.degree + 1):
            result = result + self.inv_factorials[k] * x_power
            if k < self.degree:
                x_power = x_power * x
        return result

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                scale: float = 1.0) -> torch.Tensor:
        """
        Args:
            q, k, v: (B, H, N, head_dim)
        Returns:
            attended values: (B, H, N, head_dim)
        """
        logits = (q @ k.transpose(-2, -1)) * scale  # (B, H, N, N)
        exp_logits = self.poly_exp(logits)
        attn = exp_logits / (exp_logits.sum(dim=-1, keepdim=True) + 1e-6)
        return attn @ v

    def fhe_depth(self) -> int:
        return math.ceil(math.log2(self.degree))


# ---------------------------------------------------------------------------
# L2Q (Learnable 2-Quad)
# ---------------------------------------------------------------------------

class L2Q(nn.Module):
    """
    Learnable quadratic attention: per-head coefficients {alpha, beta, gamma}.

        A_ij = (α * logit_ij^2 + β * logit_ij + γ) / normalization

    FHE depth: 1 level (same as PowerSoftmax p=2 but with more expressivity).

    Coefficients are initialized so the quadratic approximates standard softmax
    near logit=0: gamma ~ 1/N (uniform attention as baseline).

    Args:
        n_heads: Number of attention heads
        n_tokens: Sequence length (used for gamma initialization)
    """

    def __init__(self, n_heads: int, n_tokens: int = 64):
        super().__init__()
        # Per-head learnable coefficients
        self.alpha = nn.Parameter(torch.ones(n_heads, 1, 1) * 0.125)
        self.beta  = nn.Parameter(torch.ones(n_heads, 1, 1) * 0.5)
        self.gamma = nn.Parameter(torch.ones(n_heads, 1, 1) * (1.0 / n_tokens))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                scale: float = 1.0) -> torch.Tensor:
        """
        Args:
            q, k, v: (B, H, N, head_dim)
        Returns:
            attended values: (B, H, N, head_dim)
        """
        logits = (q @ k.transpose(-2, -1)) * scale  # (B, H, N, N)
        # Learnable quadratic per head
        attn = self.alpha * logits ** 2 + self.beta * logits + self.gamma
        # Normalize
        attn = F.relu(attn)  # ensure non-negative (GELU approx of ReLU for range)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
        return attn @ v

    def fhe_depth(self) -> int:
        return 1  # degree-2 polynomial


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_softmax_replacements(
    dim: int = 64,
    n_heads: int = 4,
    n_tokens: int = 64,
    batch_size: int = 4,
    n_runs: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Compare accuracy and speed of softmax replacement strategies on a toy ViT attention block.

    Metrics:
      - Max absolute error vs. standard softmax output
      - Forward pass latency
      - FHE depth cost
    """
    head_dim = dim // n_heads
    scale = head_dim ** -0.5

    # Random Q, K, V
    torch.manual_seed(42)
    q = torch.randn(batch_size, n_heads, n_tokens, head_dim).to(device)
    k = torch.randn(batch_size, n_heads, n_tokens, head_dim).to(device)
    v = torch.randn(batch_size, n_heads, n_tokens, head_dim).to(device)

    # Reference: standard softmax attention
    with torch.no_grad():
        logits = (q @ k.transpose(-2, -1)) * scale
        attn_std = logits.softmax(dim=-1)
        ref_out = attn_std @ v

    methods = {
        "Standard Softmax": ("exact", lambda: (q @ k.transpose(-2, -1) * scale).softmax(-1) @ v),
        "PowerSoftmax (p=2)": (1, PowerSoftmax(p=2).to(device)),
        "PowerSoftmax (p=4)": (2, PowerSoftmax(p=4).to(device)),
        "MGFSoftmax (deg=3)": (2, MGFSoftmax(degree=3).to(device)),
        "MGFSoftmax (deg=7)": (3, MGFSoftmax(degree=7).to(device)),
        "L2Q": (1, L2Q(n_heads=n_heads, n_tokens=n_tokens).to(device)),
    }

    print(f"\n{'Method':<25} {'FHE Depth':>10} {'Max Error':>12} {'Latency (ms)':>14}")
    print("-" * 65)

    with torch.no_grad():
        for name, (depth, method) in methods.items():
            # Warm up
            for _ in range(3):
                if callable(method) and not isinstance(method, nn.Module):
                    _ = method()
                else:
                    _ = method(q, k, v, scale)

            # Timed runs
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                if callable(method) and not isinstance(method, nn.Module):
                    out = method()
                else:
                    out = method(q, k, v, scale)
                times.append(time.perf_counter() - t0)

            max_err = (out - ref_out).abs().max().item()
            latency_ms = 1000 * sum(times) / len(times)
            depth_str = str(depth) if depth != "exact" else "N/A"

            print(f"{name:<25} {depth_str:>10} {max_err:>12.4f} {latency_ms:>14.3f}")

    print(f"\nNote: Max error is vs. standard softmax output (B={batch_size}, N={n_tokens}, H={n_heads})")
    print("FHE depth is multiplicative levels consumed by the attention scoring.")


if __name__ == "__main__":
    benchmark_softmax_replacements(
        dim=64, n_heads=4, n_tokens=64, batch_size=4, n_runs=50
    )
