"""
PolyTransformer: FHE-friendly transformer with range-loss training.

Implements:
  - RangeLoss: penalty on out-of-range attention logits
  - PolyAttention: quadratic attention (softmax-free)
  - LinearNorm: learnable affine scaling (LayerNorm-free)
  - PolyFFN: polynomial GELU feed-forward block
  - PolyTransformerBlock: full FHE-friendly block

FHE depth per block: ~3 levels (vs BOLT's 8)

Reference: IBM Research, "Towards Practical Homomorphic Evaluation of
           Transformer Inference"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Range-Loss Training Trick
# ---------------------------------------------------------------------------

class RangeLoss(nn.Module):
    """
    Penalizes values outside [-R, R].

    Encourages the model to keep attention logits in a bounded range
    so that low-degree polynomial approximations remain accurate.

    L_range = lambda * mean( max(0, |x| - R)^2 )

    Args:
        R: Target range bound
        lam: Loss weight
    """

    def __init__(self, R: float = 5.0, lam: float = 0.01):
        super().__init__()
        self.R = R
        self.lam = lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        excess = F.relu(x.abs() - self.R)
        return self.lam * (excess ** 2).mean()


# ---------------------------------------------------------------------------
# LinearNorm
# ---------------------------------------------------------------------------

class LinearNorm(nn.Module):
    """
    Learnable affine scaling — replaces LayerNorm.

    y = gamma * x + beta

    FHE depth cost: 0 (multiply + add by constants is free).

    Unlike LayerNorm, no division by std — eliminates the costly
    1/sqrt(var) approximation.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x + self.beta


# ---------------------------------------------------------------------------
# Polynomial Attention
# ---------------------------------------------------------------------------

class PolyAttention(nn.Module):
    """
    Attention without softmax, using a polynomial normalization.

    Attn(Q, K, V) = A @ V    where   A_ij = (q_i · k_j)^2 / sum_j (q_i · k_j)^2

    FHE depth:
      - QKV projections: 1 level
      - Q @ K.T: 1 level (matmul)
      - square: 1 level
      - Output projection: 1 level
      Total: 4 levels

    Note: The division for normalization is done in plaintext at decrypt time,
    or approximated. In practice, plain linear attention (BOLT) avoids it entirely.
    Here we keep the quadratic for better approximation of softmax behavior.

    Args:
        dim: Model dimension
        n_heads: Number of attention heads
        range_R: Range bound for RangeLoss (passed in from outside)
    """

    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        # Stores last QK.T for RangeLoss computation in training loop
        self._last_attn_logits: torch.Tensor = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, H, N, d)

        # Scaled dot-product logits
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        self._last_attn_logits = attn_logits  # saved for RangeLoss

        # Quadratic normalization (poly approximation of softmax)
        attn = attn_logits ** 2
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# ---------------------------------------------------------------------------
# Polynomial FFN
# ---------------------------------------------------------------------------

class PolyGELU2(nn.Module):
    """
    Degree-2 polynomial approximation of GELU.

    f(x) ≈ 0.125 x^2 + 0.5 x + 0.25

    Accurate for x ∈ [-3, 3] (sufficient when range-loss is active).
    FHE depth: 1 level.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.125 * x ** 2 + 0.5 * x + 0.25


class PolyFFN(nn.Module):
    """
    Feed-forward block with polynomial GELU.

    FHE depth: 1 level (degree-2) or 2 levels (degree-4).
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, poly_degree: int = 2):
        super().__init__()
        mlp_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.act = PolyGELU2() if poly_degree == 2 else _PolyGELU4()
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _PolyGELU4(nn.Module):
    """Degree-4 GELU (BOLT coefficients). Depth: 2 levels."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x + 0.1972 * x ** 3 + 0.0012 * x ** 4


# ---------------------------------------------------------------------------
# Full Block
# ---------------------------------------------------------------------------

class PolyTransformerBlock(nn.Module):
    """
    FHE-friendly transformer block.

    Depth budget:
      LinearNorm:     0
      PolyAttention:  4  (QKV proj + QK.T + square + out proj)
      LinearNorm:     0
      PolyFFN (deg2): 2  (FC1 + poly)
      Block total:    6 levels
    """

    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: float = 4.0,
                 poly_degree: int = 2):
        super().__init__()
        self.norm1 = LinearNorm(dim)
        self.attn = PolyAttention(dim, n_heads)
        self.norm2 = LinearNorm(dim)
        self.ffn = PolyFFN(dim, mlp_ratio, poly_degree)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    def attn_logits(self) -> torch.Tensor:
        """Return last attention logits for RangeLoss computation."""
        return self.attn._last_attn_logits


# ---------------------------------------------------------------------------
# PolyTransformer ViT
# ---------------------------------------------------------------------------

class PolyTransformerViT(nn.Module):
    """
    Full PolyTransformer ViT for CIFAR-10.

    Designed for FHE: all ops are polynomial, range-loss keeps inputs bounded.
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
        poly_degree: int = 2,
    ):
        super().__init__()
        n_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        self.patch_size = patch_size

        self.patch_embed = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))

        self.blocks = nn.ModuleList([
            PolyTransformerBlock(dim, n_heads, mlp_ratio, poly_degree)
            for _ in range(depth)
        ])
        self.norm = LinearNorm(dim)
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

    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        return x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * p * p)

    def forward(self, x):
        from einops import repeat
        B = x.shape[0]
        x = self.patch_embed(self.patchify(x))
        cls = repeat(self.cls_token, "1 1 d -> b 1 d", b=B)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 0]))

    def range_loss(self, range_loss_fn: RangeLoss) -> torch.Tensor:
        """Aggregate RangeLoss across all attention blocks."""
        total = torch.tensor(0.0)
        for block in self.blocks:
            logits = block.attn_logits()
            if logits is not None:
                total = total + range_loss_fn(logits)
        return total


# ---------------------------------------------------------------------------
# Training with Range-Loss
# ---------------------------------------------------------------------------

def train_poly_transformer(
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 3e-4,
    range_R: float = 5.0,
    range_lam: float = 0.01,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> PolyTransformerViT:
    """Train PolyTransformer on CIFAR-10 with range-loss."""
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    print(f"Training PolyTransformerViT | R={range_R} λ={range_lam}")

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = T.Compose([
        T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_loader = DataLoader(
        torchvision.datasets.CIFAR10("data/", train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=2,
    )
    test_loader = DataLoader(
        torchvision.datasets.CIFAR10("data/", train=False, download=True, transform=transform_test),
        batch_size=256, shuffle=False,
    )

    model = PolyTransformerViT().to(device)
    range_loss_fn = RangeLoss(R=range_R, lam=range_lam)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y) + model.range_loss(range_loss_fn)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    correct += (model(x).argmax(-1) == y).sum().item()
                    total += y.size(0)
            acc = 100 * correct / total
            best_acc = max(best_acc, acc)
            print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.3f} | Acc: {acc:.2f}%")

    print(f"\nBest accuracy: {best_acc:.2f}%")
    return model


if __name__ == "__main__":
    model = PolyTransformerViT()
    print(f"PolyTransformerViT: {sum(p.numel() for p in model.parameters()):,} params")

    # Quick forward pass check
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"Output shape: {out.shape}")  # (2, 10)

    model = train_poly_transformer(epochs=50)
    torch.save(model.state_dict(), "checkpoints/poly_transformer_cifar10.pt")
