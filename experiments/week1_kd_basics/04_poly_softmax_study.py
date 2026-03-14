"""
Week 1, Experiment 4: Polynomial softmax comparison in a single-block ViT.

Compares three polynomial attention mechanisms on a 1-block ViT:
  1. Standard softmax (baseline)
  2. PowerSoftmax (x^p / sum(x^p))
  3. MGFSoftmax (Gaussian mixture approximation)
  4. L2Q (L2 normalization variant)

Metrics:
  - Max error vs. standard softmax on random attention logits
  - Training accuracy after 10 epochs on CIFAR-10

Usage:
    python experiments/week1_kd_basics/04_poly_softmax_study.py

Outputs:
    results/week1/softmax_comparison.json
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    from papers.power_softmax.implementation import PowerSoftmax, MGFSoftmax, L2Q
except ImportError:
    # Fallback implementations
    class PowerSoftmax(nn.Module):
        def __init__(self, p: int = 2, dim: int = -1):
            super().__init__()
            self.p = p
            self.dim = dim

        def forward(self, x):
            x_pos = x - x.min(dim=self.dim, keepdim=True).values + 1e-6
            xp = x_pos.pow(self.p)
            return xp / xp.sum(dim=self.dim, keepdim=True)

    class MGFSoftmax(nn.Module):
        """Gaussian mixture approximation of softmax."""
        def forward(self, x):
            # Simplified: normalize using degree-2 approximation of exp
            x2 = 1 + x + 0.5 * x.pow(2)  # Taylor: e^x ≈ 1 + x + x²/2
            x2 = x2.clamp(min=1e-6)
            return x2 / x2.sum(dim=-1, keepdim=True)

    class L2Q(nn.Module):
        """L2 normalization-based attention weight."""
        def forward(self, x):
            x_pos = x - x.min(dim=-1, keepdim=True).values + 1e-6
            return x_pos / (x_pos.norm(dim=-1, keepdim=True) + 1e-6)


RESULTS_DIR = "results/week1"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
EPOCHS = 10
LR = 3e-4
BATCH_SIZE = 128
SEQ_LEN = 196   # 14×14 patches for 224×224 image
HEAD_DIM = 64


class SingleBlockViT(nn.Module):
    """Minimal ViT with a single transformer block and configurable attention."""
    def __init__(self, attention_fn, img_size=32, patch_size=4, embed_dim=256,
                 num_heads=4, num_classes=10):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim)

        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_fn = attention_fn

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, C)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed

        # Self-attention
        x_norm = self.norm1(x)
        Q = self.q(x_norm)
        K = self.k(x_norm)
        V = self.v(x_norm)

        B, N, C = Q.shape
        H, D = self.num_heads, self.head_dim
        Q = Q.view(B, N, H, D).transpose(1, 2)
        K = K.view(B, N, H, D).transpose(1, 2)
        V = V.view(B, N, H, D).transpose(1, 2)

        attn_logits = (Q @ K.transpose(-2, -1)) / (D ** 0.5)  # (B, H, N, N)
        attn = self.attn_fn(attn_logits)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, N, C)
        x = x + self.proj(out)

        # FFN
        x = x + self.ffn(self.norm2(x))
        return self.head(x[:, 0])  # CLS token


def get_cifar10_loaders(batch_size=BATCH_SIZE):
    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train = DataLoader(
        torchvision.datasets.CIFAR10("data/", train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=2,
    )
    test = DataLoader(
        torchvision.datasets.CIFAR10("data/", train=False, transform=transform_test),
        batch_size=256, shuffle=False,
    )
    return train, test


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100 * correct / total


def train_model(model, train_loader, epochs, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            ce(model(x), y).backward()
            optimizer.step()
        scheduler.step()
        print(f"    epoch {epoch}/{epochs}", end="\r")
    print()


def measure_softmax_error(attn_fn, n_samples=1000, seq_len=64, device=DEVICE):
    """Compute max and mean absolute error vs standard softmax on random logits."""
    errors = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = torch.randn(1, 1, seq_len, seq_len, device=device) * 2.0
            ref = F.softmax(logits, dim=-1)
            approx = attn_fn(logits)
            err = (approx - ref).abs()
            errors.append(err.max().item())
    return {
        "max_error":  round(float(np.max(errors)), 5),
        "mean_max_error": round(float(np.mean(errors)), 5),
        "p95_error":  round(float(np.percentile(errors, 95)), 5),
    }


def main():
    print(f"Device: {DEVICE}")
    train_loader, test_loader = get_cifar10_loaders()

    attn_configs = {
        "standard_softmax": lambda x: F.softmax(x, dim=-1),
        "power_softmax_p2": PowerSoftmax(p=2),
        "power_softmax_p4": PowerSoftmax(p=4),
        "mgf_softmax":      MGFSoftmax(),
        "l2q":              L2Q(),
    }

    results = {}
    for name, attn_fn in attn_configs.items():
        print(f"\n--- {name} ---")

        # Error analysis
        print("  Measuring approximation error vs standard softmax...")
        if name == "standard_softmax":
            error_stats = {"max_error": 0.0, "mean_max_error": 0.0, "p95_error": 0.0}
        else:
            attn_module = attn_fn if isinstance(attn_fn, nn.Module) else nn.Module()
            if not isinstance(attn_fn, nn.Module):
                # wrap lambda
                class _Wrap(nn.Module):
                    def forward(self, x): return attn_fn(x)
                attn_module = _Wrap()
            error_stats = measure_softmax_error(attn_module, device=DEVICE)
        print(f"  Error: {error_stats}")

        # Training accuracy
        print(f"  Training SingleBlockViT for {EPOCHS} epochs on CIFAR-10...")
        if isinstance(attn_fn, nn.Module):
            attn_module = attn_fn.to(DEVICE)
        else:
            class _Wrap(nn.Module):
                def __init__(self, fn): super().__init__(); self.fn = fn
                def forward(self, x): return self.fn(x)
            attn_module = _Wrap(attn_fn).to(DEVICE)

        model = SingleBlockViT(attn_module, img_size=32, patch_size=4,
                               embed_dim=256, num_heads=4, num_classes=NUM_CLASSES).to(DEVICE)
        train_model(model, train_loader, EPOCHS, DEVICE)
        test_acc = evaluate(model, test_loader, DEVICE)
        print(f"  Test accuracy: {test_acc:.2f}%")

        results[name] = {
            "test_acc": round(test_acc, 3),
            "approximation_error": error_stats,
        }

    print("\n=== Summary ===")
    print(f"{'Method':<25} {'Test Acc':>10} {'Max Error':>12} {'Mean Max Err':>14}")
    print("-" * 65)
    for name, val in results.items():
        err = val["approximation_error"]
        print(f"  {name:<23} {val['test_acc']:>9.2f}% {err['max_error']:>12.5f} {err['mean_max_error']:>14.5f}")

    out_path = os.path.join(RESULTS_DIR, "softmax_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
