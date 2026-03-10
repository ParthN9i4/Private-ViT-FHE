"""
TinyViT: Hierarchical ViT with sparsified logit distillation.

Implements a CIFAR-10/MedMNIST-scale variant of TinyViT-5M:
  - 4 stages with patch merging
  - Window attention (local)
  - ConvFFN with depthwise convolution
  - adapt_for_fhe(): swap GELU/softmax for polynomial alternatives
  - LogitDistillation: pre-stored logit loading

Paper: https://arxiv.org/abs/2207.10666
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import os
import json


# ---------------------------------------------------------------------------
# Patch Merging (spatial downsampling)
# ---------------------------------------------------------------------------

class PatchMerging(nn.Module):
    """
    2×2 spatial downsampling: (H, W, C) → (H/2, W/2, 2C).

    Concatenates 4 neighboring patches and projects to 2C.
    FHE depth: 1 level (linear projection).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * in_dim)
        self.proj = nn.Linear(4 * in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: (B, H*W, C)
        Returns:
            x: (B, H/2 * W/2, out_dim)
            new_H, new_W
        """
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # top-left
        x1 = x[:, 1::2, 0::2, :]  # bottom-left
        x2 = x[:, 0::2, 1::2, :]  # top-right
        x3 = x[:, 1::2, 1::2, :]  # bottom-right
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)
        x = self.proj(self.norm(x))
        return x, H // 2, W // 2


# ---------------------------------------------------------------------------
# Window Attention
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    """
    Local window attention. Each patch attends within a window_size × window_size region.

    For FHE: smaller window = fewer slots needed per ciphertext.

    Args:
        dim: Feature dimension
        window_size: Local window size (e.g. 7 for 14×14 feature map)
        n_heads: Number of attention heads
        use_poly: If True, replace softmax with quadratic normalization
    """

    def __init__(self, dim: int, window_size: int = 7, n_heads: int = 4,
                 use_poly: bool = False):
        super().__init__()
        self.window_size = window_size
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.use_poly = use_poly

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias table
        self.rel_pos_bias = nn.Embedding(
            (2 * window_size - 1) ** 2, n_heads
        )
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"
        ))
        flat = coords.flatten(1)
        rel = flat[:, :, None] - flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rel_pos_index", rel.sum(-1).flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape  # N = window_size^2
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        bias = self.rel_pos_bias(self.rel_pos_index).reshape(N, N, self.n_heads)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)

        if self.use_poly:
            # Quadratic normalization (polynomial softmax)
            attn = attn ** 2
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) → (B*nW, ws*ws, C)"""
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """(B*nW, ws*ws, C) → (B, H, W, C)"""
    B_nW = windows.shape[0]
    nW = (H // window_size) * (W // window_size)
    B = B_nW // nW
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)


# ---------------------------------------------------------------------------
# ConvFFN
# ---------------------------------------------------------------------------

class ConvFFN(nn.Module):
    """
    Feed-forward with depthwise 3×3 convolution.

    Structure: FC (expand) → DWConv 3×3 → activation → FC (compress)

    FHE note: DWConv is a constant-weight plaintext operation.
    For FHE, replace GELU with polynomial alternative via use_poly flag.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, use_poly: bool = False):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.act = _PolyGELU4() if use_poly else nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.use_poly = use_poly

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = self.fc1(x)  # (B, N, hidden)
        # Reshape to spatial for DWConv
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.dwconv(x)
        x = x.reshape(B, -1, N).transpose(1, 2)
        x = self.act(x)
        return self.fc2(x)


class _PolyGELU4(nn.Module):
    def forward(self, x):
        return 0.5 * x + 0.1972 * x ** 3 + 0.0012 * x ** 4


# ---------------------------------------------------------------------------
# TinyViT Block
# ---------------------------------------------------------------------------

class TinyViTBlock(nn.Module):
    """
    Single TinyViT block: WindowAttention + ConvFFN.

    Args:
        use_poly: If True, use polynomial ops (FHE-compatible)
    """

    def __init__(self, dim: int, window_size: int = 7, n_heads: int = 4,
                 mlp_ratio: float = 4.0, use_poly: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim) if not use_poly else _LinearNorm(dim)
        self.attn = WindowAttention(dim, window_size, n_heads, use_poly)
        self.norm2 = nn.LayerNorm(dim) if not use_poly else _LinearNorm(dim)
        self.ffn = ConvFFN(dim, mlp_ratio, use_poly)
        self.window_size = window_size

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        shortcut = x

        x = self.norm1(x)
        x_2d = x.reshape(B, H, W, C)
        x_windows = window_partition(x_2d, self.window_size)  # (B*nW, ws^2, C)
        x_windows = self.attn(x_windows)
        x = window_reverse(x_windows, self.window_size, H, W).reshape(B, N, C)
        x = shortcut + x

        x = x + self.ffn(self.norm2(x), H, W)
        return x


class _LinearNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        return self.gamma * x + self.beta


# ---------------------------------------------------------------------------
# TinyViT-5M (CIFAR-10 scale)
# ---------------------------------------------------------------------------

class TinyViT5M(nn.Module):
    """
    TinyViT-5M adapted for CIFAR-10 (32×32 images).

    4 stages with progressive patch merging.
    Default: standard (non-FHE) mode. Call adapt_for_fhe() to switch.

    Stage dims: [64, 128, 192, 384]
    """

    def __init__(
        self,
        image_size: int = 32,
        n_classes: int = 10,
        dims: List[int] = None,
        depths: List[int] = None,
        n_heads: List[int] = None,
        window_sizes: List[int] = None,
        mlp_ratio: float = 4.0,
        use_poly: bool = False,
    ):
        super().__init__()
        dims = dims or [64, 128, 192, 384]
        depths = depths or [2, 2, 2, 2]
        n_heads = n_heads or [2, 4, 6, 8]
        window_sizes = window_sizes or [8, 4, 2, 1]  # adjust for 32×32 input

        # Stem: 32×32 → 8×8 (patch size 4)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
        )
        self.stem_H = image_size // 4
        self.stem_W = image_size // 4

        # Build stages
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        H, W = self.stem_H, self.stem_W

        for i, (dim, depth, heads, ws) in enumerate(zip(dims, depths, n_heads, window_sizes)):
            stage = nn.ModuleList([
                TinyViTBlock(dim, ws, heads, mlp_ratio, use_poly)
                for _ in range(depth)
            ])
            self.stages.append(stage)
            if i < len(dims) - 1:
                self.merges.append(PatchMerging(dim, dims[i + 1]))

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], n_classes)
        self.dims = dims
        self.depths = depths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.patch_embed(x)  # (B, C, H/4, W/4)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        for i, stage in enumerate(self.stages):
            for block in stage:
                x = block(x, H, W)
            if i < len(self.merges):
                x, H, W = self.merges[i](x, H, W)

        x = self.norm(x)
        x = x.mean(dim=1)  # global average pooling
        return self.head(x)

    def adapt_for_fhe(self) -> "TinyViT5M":
        """
        In-place: swap all GELU and softmax for polynomial alternatives.
        Returns self for chaining.
        """
        for stage in self.stages:
            for block in stage:
                block.attn.use_poly = True
                block.ffn.act = _PolyGELU4()
                block.ffn.use_poly = True
                block.norm1 = _LinearNorm(block.norm1.normalized_shape[0])
                block.norm2 = _LinearNorm(block.norm2.normalized_shape[0])
        return self


# ---------------------------------------------------------------------------
# Logit Distillation (pre-stored teacher logits)
# ---------------------------------------------------------------------------

class LogitDistillation(nn.Module):
    """
    Training loss using pre-stored teacher logits.

    Teacher logits are pre-computed and saved to disk as a JSON/pt file.
    At training time, we load the logits by index — no teacher model needed online.

    Usage:
        # Pre-compute:
        store_teacher_logits(teacher, train_dataset, "data/teacher_logits.pt")

        # Training:
        distill = LogitDistillation("data/teacher_logits.pt")
        loss = distill(student_logits, labels, indices)
    """

    def __init__(self, logit_path: str, tau: float = 1.0, alpha: float = 0.5):
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        if os.path.exists(logit_path):
            self.teacher_logits = torch.load(logit_path, map_location="cpu")
            print(f"Loaded pre-stored logits from {logit_path} "
                  f"({self.teacher_logits.shape[0]} samples)")
        else:
            self.teacher_logits = None
            print(f"Warning: logit file not found at {logit_path}. "
                  "Call store_teacher_logits() first.")

    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        ce = F.cross_entropy(student_logits, labels)
        if self.teacher_logits is None:
            return ce

        t_logits = self.teacher_logits[indices].to(student_logits.device)
        kl = F.kl_div(
            F.log_softmax(student_logits / self.tau, dim=-1),
            F.softmax(t_logits / self.tau, dim=-1),
            reduction="batchmean",
        ) * (self.tau ** 2)
        return (1 - self.alpha) * ce + self.alpha * kl


def store_teacher_logits(
    teacher: nn.Module,
    dataset,
    output_path: str,
    batch_size: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Pre-compute and store teacher logits for all training samples.

    Args:
        teacher: Trained teacher model
        dataset: Training dataset (must return (image, label))
        output_path: Where to save the logits tensor
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    teacher = teacher.to(device).eval()

    all_logits = []
    with torch.no_grad():
        for x, _ in loader:
            logits = teacher(x.to(device)).cpu()
            all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=0)
    torch.save(all_logits, output_path)
    print(f"Saved {all_logits.shape[0]} teacher logits to {output_path}")


if __name__ == "__main__":
    # Quick model check
    model = TinyViT5M(image_size=32, n_classes=10)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyViT-5M (CIFAR-10): {n_params:,} params")

    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"Standard output: {out.shape}")

    model_fhe = model.adapt_for_fhe()
    out_fhe = model_fhe(x)
    print(f"FHE-adapted output: {out_fhe.shape}")
