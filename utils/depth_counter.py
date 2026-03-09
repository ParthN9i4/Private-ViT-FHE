"""
Multiplicative depth analysis for transformer architectures.

Before implementing FHE inference, know exactly how many CKKS levels
your model consumes. This tool lets you annotate a model and get a
depth budget breakdown.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DepthNode:
    """Represents one operation in the computation graph."""
    name: str
    depth_cost: int
    description: str = ""
    children: List["DepthNode"] = field(default_factory=list)

    @property
    def total_depth(self) -> int:
        """Total depth including all children."""
        if not self.children:
            return self.depth_cost
        return self.depth_cost + max(c.total_depth for c in self.children)


def vit_depth_budget(
    n_layers: int = 6,
    n_heads: int = 8,
    hidden_dim: int = 512,
    mlp_dim: int = 2048,
    gelu_degree: int = 4,
    attention_type: str = "linear",  # "linear" | "softmax_approx"
    layernorm_type: str = "approx",  # "approx" | "removed"
) -> dict:
    """
    Estimate multiplicative depth for a ViT with given configuration.

    Args:
        n_layers: Number of transformer blocks
        n_heads: Number of attention heads
        hidden_dim: Token embedding dimension
        mlp_dim: MLP hidden dimension
        gelu_degree: Polynomial degree for GELU approximation
        attention_type: How to handle softmax in attention
        layernorm_type: How to handle LayerNorm

    Returns:
        Dictionary with per-component and total depth breakdown
    """
    from .poly_approx import poly_depth

    # --- Attention block ---
    qkv_proj = 1  # linear projection (1 mult for weight application)

    if attention_type == "linear":
        # BOLT-style: replace softmax with scaled dot product, no exp
        attention_softmax = 2  # just the dot product QK^T and scaling
    elif attention_type == "softmax_approx":
        # Iron-style: polynomial approximation of softmax
        # exp approx (degree ~7) + division approx
        attention_softmax = poly_depth(7) + poly_depth(5)
    else:
        raise ValueError(f"Unknown attention_type: {attention_type}")

    attn_v_proj = 1  # attention * V
    out_proj = 1  # output projection

    attention_total = qkv_proj + attention_softmax + attn_v_proj + out_proj

    # --- LayerNorm ---
    if layernorm_type == "approx":
        # Approximate: compute mean (free), subtract (free),
        # approximate 1/sqrt(var + eps) with degree-3 poly
        layernorm_cost = poly_depth(3)
    elif layernorm_type == "removed":
        # BOLT-style: replace with learnable scalar (free)
        layernorm_cost = 0
    else:
        raise ValueError(f"Unknown layernorm_type: {layernorm_type}")

    # --- MLP block ---
    fc1 = 1
    activation = poly_depth(gelu_degree)
    fc2 = 1
    mlp_total = fc1 + activation + fc2

    # --- Per transformer block ---
    block_depth = (
        layernorm_cost  # pre-attn norm
        + attention_total
        + layernorm_cost  # pre-mlp norm
        + mlp_total
    )

    # --- Patch embedding (initial) ---
    patch_embed = 1

    # --- Classification head ---
    cls_head = 1

    # --- Total ---
    total = patch_embed + n_layers * block_depth + cls_head

    return {
        "patch_embedding": patch_embed,
        "per_block": {
            "layernorm_pre_attn": layernorm_cost,
            "qkv_projection": qkv_proj,
            "attention_softmax": attention_softmax,
            "attn_v_projection": attn_v_proj,
            "output_projection": out_proj,
            "attention_total": attention_total,
            "layernorm_pre_mlp": layernorm_cost,
            "mlp_fc1": fc1,
            "mlp_activation": activation,
            "mlp_fc2": fc2,
            "mlp_total": mlp_total,
            "block_total": block_depth,
        },
        "n_layers": n_layers,
        "all_blocks_total": n_layers * block_depth,
        "classification_head": cls_head,
        "grand_total": total,
        "config": {
            "gelu_degree": gelu_degree,
            "attention_type": attention_type,
            "layernorm_type": layernorm_type,
        },
    }


def print_depth_report(config: Optional[dict] = None):
    """Print a formatted depth budget report for common ViT configurations."""
    configs = [
        # (name, n_layers, gelu_degree, attn_type, ln_type)
        ("Vanilla ViT-S (softmax, layernorm, gelu-4)", 6, 4, "softmax_approx", "approx"),
        ("BOLT-style ViT (linear attn, no LN, gelu-4)", 6, 4, "linear", "removed"),
        ("Iron-style ViT (softmax approx, LN approx, gelu-27)", 6, 27, "softmax_approx", "approx"),
        ("Tiny FHE-ViT (linear attn, no LN, gelu-4, 4 layers)", 4, 4, "linear", "removed"),
    ]

    print("\nMultiplicative Depth Budget for ViT Configurations")
    print("=" * 70)
    print(f"{'Configuration':<50} {'Total Depth':>12}")
    print("-" * 70)

    for name, n_layers, gelu_deg, attn, ln in configs:
        result = vit_depth_budget(
            n_layers=n_layers,
            gelu_degree=gelu_deg,
            attention_type=attn,
            layernorm_type=ln,
        )
        print(f"{name:<50} {result['grand_total']:>12}")

    print("=" * 70)
    print("\nNote: Each level requires ~40-60 bits in the modulus chain.")
    print("For 128-bit security with n=2^15: max ~50 levels.")
    print("For 128-bit security with n=2^16: max ~100 levels.\n")


if __name__ == "__main__":
    print_depth_report()

    # Detailed breakdown for BOLT-style
    print("\nDetailed breakdown: BOLT-style ViT-S (6 layers)")
    print("=" * 50)
    result = vit_depth_budget(
        n_layers=6, gelu_degree=4, attention_type="linear", layernorm_type="removed"
    )
    for k, v in result["per_block"].items():
        print(f"  {k:<30} {v:>4}")
    print(f"\n  {'Grand total':<30} {result['grand_total']:>4}")
