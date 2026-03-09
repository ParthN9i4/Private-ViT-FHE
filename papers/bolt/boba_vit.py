"""
BoBa-ViT: BOLT-style FHE-friendly Vision Transformer.

This is the plaintext PyTorch implementation.
The FHE inference layer comes after verifying this achieves target accuracy.

Paper: https://arxiv.org/abs/2307.07645
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


# ---------------------------------------------------------------------------
# FHE-Friendly Components
# ---------------------------------------------------------------------------

class ScalarNorm(nn.Module):
    """
    Replaces LayerNorm with a learnable per-channel scalar.
    Depth cost: 0 (scalar multiply is free in FHE).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class PolyGELU(nn.Module):
    """
    Degree-4 polynomial approximation of GELU.
    Coefficients from BOLT paper.
    Depth cost: ceil(log2(4)) = 2.
    """
    def forward(self, x):
        return 0.5 * x + 0.1972 * x**3 + 0.0012 * x**4


class LinearAttention(nn.Module):
    """
    Linear attention without softmax.
    Attention(Q, K, V) = (Q @ K.T) @ V / n_tokens

    Depth cost:
      - Q, K, V projections: 1 level
      - Q @ K.T: 1 level
      - (QK.T) @ V: 1 level
      - Output projection: 1 level
      Total: 4 levels
    """
    def __init__(self, dim: int, n_heads: int = 8, head_dim: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner_dim = n_heads * head_dim

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim)
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x)  # (B, N, 3*inner_dim)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, N, heads, head_dim)

        q = q.transpose(1, 2)  # (B, heads, N, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Linear attention: skip softmax, use raw dot product
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn / N  # normalize by sequence length

        out = attn @ v  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(out)


class BobaTransformerBlock(nn.Module):
    """
    Single BoBa-ViT transformer block.

    Depth budget per block:
      ScalarNorm:       0
      LinearAttention:  4
      ScalarNorm:       0
      MLP (FC+GELU+FC): 4
      Total:            8
    """
    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        mlp_dim = int(dim * mlp_ratio)

        self.norm1 = ScalarNorm(dim)
        self.attn = LinearAttention(dim, n_heads)
        self.norm2 = ScalarNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            PolyGELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class BobaViT(nn.Module):
    """
    BoBa-ViT: FHE-friendly Vision Transformer for CIFAR-10.

    Default config targets 50 multiplicative levels (fits n=2^15).
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
    ):
        super().__init__()
        assert image_size % patch_size == 0
        n_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size  # RGB patches

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.patch_size = patch_size

        # Transformer
        self.blocks = nn.ModuleList([
            BobaTransformerBlock(dim, n_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Classifier
        self.norm = ScalarNorm(dim)
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

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to sequence of patch embeddings."""
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * p * p)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patchify(x)           # (B, n_patches, patch_dim)
        x = self.patch_embed(x)        # (B, n_patches, dim)

        cls = repeat(self.cls_token, "1 1 d -> b 1 d", b=B)
        x = torch.cat([cls, x], dim=1) # (B, n_patches+1, dim)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        cls_out = x[:, 0]              # use [CLS] token
        return self.head(self.norm(cls_out))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_boba_vit(
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> BobaViT:
    """Train BoBa-ViT on CIFAR-10."""
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    print(f"Training on {device}")
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_loader = DataLoader(
        torchvision.datasets.CIFAR10("data/", train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=2,
    )
    test_loader = DataLoader(
        torchvision.datasets.CIFAR10("data/", train=False, download=True, transform=transform_test),
        batch_size=256, shuffle=False,
    )

    model = BobaViT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Eval
        if (epoch + 1) % 10 == 0:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    preds = model(x).argmax(dim=-1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            acc = 100 * correct / total
            best_acc = max(best_acc, acc)
            print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.3f} | Acc: {acc:.2f}%")

    print(f"\nBest accuracy: {best_acc:.2f}%")
    return model


if __name__ == "__main__":
    # Quick depth check
    from utils import vit_depth_budget, print_depth_report
    print_depth_report()

    # Train
    model = train_boba_vit(epochs=100)
    torch.save(model.state_dict(), "checkpoints/boba_vit_cifar10.pt")
