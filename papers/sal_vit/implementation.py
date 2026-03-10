"""
SAL-ViT: Selective Attention with Learnable Quadratic Polynomial.

Implements:
  - L2QAttention: multi-head attention with per-head learnable quadratic
  - SelectiveAttentionViT: ViT with configurable attention type per block
  - search_attention_schedule(): greedy SAS search

Paper: "SAL-ViT: Towards Latency Efficient Private Inference on ViT using
        Selective Attention with a Learnable Quadratic Polynomial"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math


# ---------------------------------------------------------------------------
# L2Q Attention
# ---------------------------------------------------------------------------

class L2QAttention(nn.Module):
    """
    Multi-head attention with L2Q (Learnable 2-Quad) scoring per head.

    Score function per head h:
        s_h(q, k) = α_h · (q·k)^2 + β_h · (q·k) + γ_h

    Attention weights:
        A_h[i,j] = ReLU(s_h(q_i, k_j)) / Σ_k ReLU(s_h(q_i, k_k))

    FHE depth: 1 level (degree-2 polynomial).

    Args:
        dim: Model dimension
        n_heads: Number of attention heads
        init_mode: 'linear' (start near BOLT linear attn) or
                   'quadratic' (start near Power-Softmax p=2)
    """

    def __init__(self, dim: int, n_heads: int = 8, init_mode: str = "quadratic"):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        # Per-head learnable coefficients: (H, 1, 1) for broadcasting
        if init_mode == "linear":
            # Near linear attention: β=1, α,γ small
            self.alpha = nn.Parameter(torch.zeros(n_heads, 1, 1))
            self.beta  = nn.Parameter(torch.ones(n_heads, 1, 1))
            self.gamma = nn.Parameter(torch.zeros(n_heads, 1, 1))
        else:  # 'quadratic' — near Power-Softmax p=2
            self.alpha = nn.Parameter(torch.ones(n_heads, 1, 1) * 0.125)
            self.beta  = nn.Parameter(torch.ones(n_heads, 1, 1) * 0.5)
            self.gamma = nn.Parameter(torch.ones(n_heads, 1, 1) * 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, H, N, d)

        logits = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, N, N)

        # L2Q scoring: α·logit² + β·logit + γ
        attn = self.alpha * logits ** 2 + self.beta * logits + self.gamma

        # Non-negative normalization
        attn = F.relu(attn)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class StandardAttention(nn.Module):
    """Standard softmax attention (used for layers not polynomialized in SAS)."""

    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# ---------------------------------------------------------------------------
# Selective Attention Block
# ---------------------------------------------------------------------------

class SelectiveBlock(nn.Module):
    """
    Transformer block that can use either L2Q or standard softmax attention.

    This is the building block for SelectiveAttentionViT.
    """

    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: float = 4.0,
                 use_poly: bool = True):
        super().__init__()
        mlp_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = L2QAttention(dim, n_heads) if use_poly else StandardAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )
        self.use_poly = use_poly

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def convert_to_poly(self) -> None:
        """Switch this block from standard softmax to L2Q in-place."""
        if not self.use_poly:
            old = self.attn
            new = L2QAttention(old.qkv.in_features, old.n_heads)
            # Copy QKV and projection weights
            new.qkv.weight.data.copy_(old.qkv.weight.data)
            new.proj.weight.data.copy_(old.proj.weight.data)
            new.proj.bias.data.copy_(old.proj.bias.data)
            self.attn = new
            self.use_poly = True


# ---------------------------------------------------------------------------
# SelectiveAttentionViT
# ---------------------------------------------------------------------------

class SelectiveAttentionViT(nn.Module):
    """
    ViT where each block independently uses L2Q or standard softmax.

    The attention_mask list controls which blocks are polynomial:
        attention_mask = [True, True, False, True, False, False]
        → blocks 0,1,3 use L2Q (FHE-friendly)
        → blocks 2,4,5 use standard softmax (expensive in FHE)

    Args:
        attention_mask: List of booleans, one per block.
                        True = L2Q (poly), False = standard softmax
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        n_classes: int = 10,
        dim: int = 256,
        depth: int = 6,
        n_heads: int = 8,
        mlp_ratio: float = 2.0,
        attention_mask: Optional[List[bool]] = None,
    ):
        super().__init__()
        if attention_mask is None:
            attention_mask = [True] * depth  # all poly by default

        assert len(attention_mask) == depth

        n_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        self.patch_size = patch_size
        self.attention_mask = attention_mask

        self.patch_embed = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))

        self.blocks = nn.ModuleList([
            SelectiveBlock(dim, n_heads, mlp_ratio, use_poly=mask)
            for mask in attention_mask
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        return x.reshape(B, C, H//p, p, W//p, p).permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C*p*p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from einops import repeat
        B = x.shape[0]
        x = self.patch_embed(self.patchify(x))
        cls = repeat(self.cls_token, "1 1 d -> b 1 d", b=B)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 0]))

    def fhe_depth_estimate(self) -> int:
        """
        Rough FHE depth estimate given the attention schedule.
        Poly layer: 4 levels (attention) + 4 (MLP with GELU degree-4) = 8
        Std layer: 9 levels (softmax approx) + 4 = 13 — but run separately
        """
        poly_layers = sum(self.attention_mask)
        std_layers = len(self.attention_mask) - poly_layers
        # This assumes std layers use polynomial softmax approximation (degree-27, ~5 levels)
        return 1 + poly_layers * 8 + std_layers * 13 + 1

    def attention_schedule_summary(self) -> str:
        parts = []
        for i, is_poly in enumerate(self.attention_mask):
            parts.append(f"Block {i}: {'L2Q' if is_poly else 'Softmax'}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Selective Attention Search (SAS)
# ---------------------------------------------------------------------------

def search_attention_schedule(
    full_model: SelectiveAttentionViT,
    train_loader,
    test_loader,
    accuracy_threshold: float = 87.0,
    fine_tune_epochs: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[bool]:
    """
    Greedy Selective Attention Search.

    Starting from all-softmax, greedily converts blocks to L2Q if
    accuracy stays above threshold after short fine-tuning.

    Returns:
        attention_mask: List[bool] — True = L2Q, False = keep softmax
    """
    import copy

    n_blocks = len(full_model.blocks)
    mask = [False] * n_blocks  # start: all standard softmax

    print(f"Starting SAS with threshold={accuracy_threshold}%")

    for block_idx in range(n_blocks):
        # Try converting block_idx to L2Q
        trial_mask = mask.copy()
        trial_mask[block_idx] = True

        model = copy.deepcopy(full_model)
        model.attention_mask = trial_mask
        # Replace the block's attention
        model.blocks[block_idx].convert_to_poly()
        model = model.to(device)

        # Short fine-tune
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        for _ in range(fine_tune_epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                loss = criterion(model(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(-1) == y).sum().item()
                total += y.size(0)
        acc = 100 * correct / total
        depth = SelectiveAttentionViT(attention_mask=trial_mask).fhe_depth_estimate()

        print(f"  Block {block_idx}: {'L2Q'} → acc={acc:.1f}% | FHE depth={depth}")

        if acc >= accuracy_threshold:
            mask[block_idx] = True
            full_model = model  # carry forward fine-tuned model
            print(f"  ✓ Accepted (mask={mask})")
        else:
            print(f"  ✗ Rejected")

    n_poly = sum(mask)
    print(f"\nFinal mask: {mask}")
    print(f"L2Q layers: {n_poly}/{n_blocks}")
    print(f"Estimated FHE depth: {SelectiveAttentionViT(attention_mask=mask).fhe_depth_estimate()}")
    return mask


if __name__ == "__main__":
    # Demo: all-L2Q SelectiveAttentionViT
    mask = [True] * 6
    model = SelectiveAttentionViT(attention_mask=mask)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SelectiveAttentionViT (all L2Q): {n_params:,} params")
    print(model.attention_schedule_summary())
    print(f"Estimated FHE depth: {model.fhe_depth_estimate()}")

    x = torch.randn(2, 3, 32, 32)
    print(f"Output: {model(x).shape}")
