"""
Which polynomial substitution hurts most?
==========================================

We already know:
  - Standard DeiT-Tiny:    79.02% on CIFAR-10
  - All-polynomial + KD:   46.03%
  - All-polynomial no KD:  11.33% (broken)

Now we test ONE substitution at a time to find the culprit:

  Model A: Replace ONLY GELU → PolyGELU       (keep real softmax + LayerNorm)
  Model B: Replace ONLY softmax → PolyAttn     (keep real GELU + LayerNorm)
  Model C: Replace ONLY LayerNorm → BatchNorm  (keep real GELU + softmax)
  Model D: Replace GELU + softmax              (keep real LayerNorm)
  Model E: All three replaced                  (same as previous experiment)

Each model is trained:
  (a) with KD from the teacher
  (b) without KD (CE only)

This tells us:
  1. Which substitution causes the biggest accuracy drop
  2. Which substitution breaks training without KD
  3. Whether the substitutions interact (D vs A+B individually)

Usage: python substitution_ablation.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import timm


# ── Data ─────────────────────────────────────────────────────────────

def get_cifar10(batch_size=128):
    t_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    t_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=t_train)
    test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=t_test)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True))


# ── Polynomial replacements ──────────────────────────────────────────
#
# These are the same ones from simple_kd_baseline.py.
# Each one replaces a nonlinear operation that CKKS cannot compute
# with a polynomial (additions + multiplications only).

class PolyGELU(nn.Module):
    """
    Replaces GELU with ax² + bx + c.

    GELU is the activation function used in every FFN block of the
    transformer. It decides which neurons "fire" and which stay quiet.

    Why polynomial? Under CKKS encryption, you cannot compute
    GELU(x) = x * Φ(x) because the Gaussian CDF Φ involves exp
    and erf — impossible with just add/multiply.

    ax² + bx + c uses only multiply and add → works under CKKS.
    Cost: 1 CKKS multiplicative level.
    """
    def __init__(self):
        super().__init__()
        # Fit a degree-2 polynomial to GELU over [-3, 3]
        x = torch.linspace(-3, 3, 10000)
        y = F.gelu(x)
        X = torch.stack([x**2, x, torch.ones_like(x)], dim=1)
        c = torch.linalg.lstsq(X, y).solution
        self.a = nn.Parameter(c[0].clone())
        self.b = nn.Parameter(c[1].clone())
        self.c = nn.Parameter(c[2].clone())

    def forward(self, x):
        return self.a * x * x + self.b * x + self.c


class PolyAttn(nn.Module):
    """
    Replaces softmax in attention with ax² + bx + c applied element-wise.

    Softmax converts attention scores into probabilities:
      attn_weights = softmax(Q @ K^T / sqrt(d))

    Under CKKS, softmax is the HARDEST operation because it needs:
      - max subtraction (comparison → ~15 CKKS levels)
      - exponentiation (transcendental → impossible exactly)
      - division by sum (inverse → ~20 levels via Goldschmidt)
    Total: ~38 levels, more than the entire CKKS budget.

    Our replacement: just apply a polynomial to each attention score
    independently. No max, no exp, no division.
    Cost: 1 CKKS level.

    The polynomial acts as a "soft gate" — KD trains it so that the
    student's attention patterns produce similar outputs to the teacher.
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.05))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.25))

    def forward(self, x):
        return self.a * x * x + self.b * x + self.c


# ── Configurable DeiT-Tiny ──────────────────────────────────────────
#
# This single model class can use either standard or polynomial
# operations. We control which ones to replace via boolean flags.

class ConfigurableDeiT(nn.Module):
    """
    DeiT-Tiny where you choose which operations to replace.

    Args:
        poly_gelu:    If True, use PolyGELU instead of nn.GELU
        poly_softmax: If True, use PolyAttn instead of softmax
        poly_norm:    If True, use BatchNorm1d instead of LayerNorm
    """
    def __init__(self, num_classes=10, img_size=32, patch_size=4,
                 embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0,
                 poly_gelu=False, poly_softmax=False, poly_norm=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.poly_softmax = poly_softmax
        self.poly_norm = poly_norm

        # Patch embedding — same for all variants
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Build transformer blocks
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(ConfigurableBlock(
                embed_dim, num_heads, mlp_ratio,
                poly_gelu=poly_gelu,
                poly_softmax=poly_softmax,
                poly_norm=poly_norm,
            ))

        # Final norm + head
        if poly_norm:
            self.norm = nn.BatchNorm1d(embed_dim)
        else:
            self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        # Final norm
        if isinstance(self.norm, nn.BatchNorm1d):
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.norm(x)

        return self.head(x[:, 0])


class ConfigurableBlock(nn.Module):
    """One transformer block with configurable operations."""

    def __init__(self, dim, num_heads, mlp_ratio,
                 poly_gelu=False, poly_softmax=False, poly_norm=False):
        super().__init__()

        # Norm layers: LayerNorm (standard) or BatchNorm1d (polynomial)
        if poly_norm:
            self.norm1 = nn.BatchNorm1d(dim)
            self.norm2 = nn.BatchNorm1d(dim)
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        self.use_bn = poly_norm

        # Attention
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.poly_softmax = poly_softmax
        if poly_softmax:
            self.attn_act = PolyAttn()

        # FFN
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        if poly_gelu:
            self.act = PolyGELU()
        else:
            self.act = nn.GELU()

    def _norm(self, norm_layer, x):
        """Apply norm — handles the transpose needed for BatchNorm1d."""
        if self.use_bn:
            return norm_layer(x.transpose(1, 2)).transpose(1, 2)
        else:
            return norm_layer(x)

    def forward(self, x):
        # Attention
        h = self._norm(self.norm1, x)
        B, N, C = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.poly_softmax:
            attn = self.attn_act(attn)
        else:
            attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + self.proj(out)

        # FFN
        h = self._norm(self.norm2, x)
        x = x + self.fc2(self.act(self.fc1(h)))

        return x


# ── Training ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += model(imgs).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def train_model(model, train_loader, test_loader, device, epochs=100,
                teacher=None, kd_alpha=0.1, kd_temp=4.0, label=""):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if teacher is not None:
        teacher = teacher.to(device)
        teacher.eval()

    best_acc = 0.0
    print(f"\n  Training: {label}")
    print(f"  KD: {'yes' if teacher else 'no'}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            ce_loss = F.cross_entropy(logits, labels)

            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(imgs)
                kd_loss = F.kl_div(
                    F.log_softmax(logits / kd_temp, dim=-1),
                    F.softmax(t_logits / kd_temp, dim=-1),
                    reduction='batchmean')
                loss = kd_alpha * kd_loss + (1 - kd_alpha) * ce_loss
            else:
                loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, acc)

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  loss={total_loss/len(train_loader):.4f}"
                  f"  acc={acc:.2f}%  best={best_acc:.2f}%")

    print(f"  ► Best: {best_acc:.2f}%")
    return best_acc


# ── Main ─────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    epochs = 100
    train_loader, test_loader = get_cifar10()

    # ── Step 1: Train teacher ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 1: Train standard teacher")
    print(f"{'='*60}")

    teacher = ConfigurableDeiT(
        poly_gelu=False, poly_softmax=False, poly_norm=False)
    teacher_acc = train_model(
        teacher, train_loader, test_loader, device, epochs,
        label="Teacher: standard GELU + softmax + LayerNorm")

    # ── Step 2: Test each substitution ───────────────────────────────
    #
    # We define 5 student configurations.
    # Each one replaces a different combination of operations.
    # All are trained WITH KD from the teacher trained above.

    configs = [
        # (name,         poly_gelu, poly_softmax, poly_norm)
        ("A: GELU only",    True,      False,        False),
        ("B: softmax only", False,     True,         False),
        ("C: norm only",    False,     False,        True),
        ("D: GELU+softmax", True,      True,         False),
        ("E: all three",    True,      True,         True),
    ]

    print(f"\n{'='*60}")
    print("STEP 2: Test each substitution (with KD)")
    print(f"{'='*60}")

    results_kd = {}
    for name, pg, ps, pn in configs:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        student = ConfigurableDeiT(
            poly_gelu=pg, poly_softmax=ps, poly_norm=pn)

        what_replaced = []
        if pg: what_replaced.append("GELU→PolyGELU")
        if ps: what_replaced.append("softmax→PolyAttn")
        if pn: what_replaced.append("LayerNorm→BatchNorm")

        acc = train_model(
            student, train_loader, test_loader, device, epochs,
            teacher=teacher, label=f"{name}  [{', '.join(what_replaced)}]")
        results_kd[name] = acc

    # ── Step 3: Test without KD (to see which substitution
    #            breaks the model's ability to learn from CE alone)

    print(f"\n{'='*60}")
    print("STEP 3: Test each substitution (WITHOUT KD)")
    print(f"{'='*60}")

    results_no_kd = {}
    for name, pg, ps, pn in configs:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        student = ConfigurableDeiT(
            poly_gelu=pg, poly_softmax=ps, poly_norm=pn)

        acc = train_model(
            student, train_loader, test_loader, device, epochs,
            label=f"{name} (no KD)")
        results_no_kd[name] = acc

    # ── Results table ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Teacher (standard):           {teacher_acc:.2f}%")
    print(f"")
    print(f"  {'Configuration':<22} {'With KD':>8} {'No KD':>8} {'KD helps':>9} {'Drop from teacher':>18}")
    print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*9} {'─'*18}")

    for name, _, _, _ in configs:
        kd = results_kd[name]
        no = results_no_kd[name]
        diff = kd - no
        drop = teacher_acc - kd
        print(f"  {name:<22} {kd:>7.2f}% {no:>7.2f}% {diff:>+8.2f}% {drop:>13.2f}%")

    print(f"\n  Key questions answered:")
    print(f"  1. Which substitution hurts most? (largest drop from teacher)")
    print(f"  2. Which one breaks CE-only training? (no-KD ≈ 10% = random)")
    print(f"  3. Do substitutions interact? (compare D vs A+B drops)")

    # Save
    torch.save({
        'teacher_acc': teacher_acc,
        'results_kd': results_kd,
        'results_no_kd': results_no_kd,
    }, 'substitution_ablation_results.pth')
    print(f"\nSaved to substitution_ablation_results.pth")


if __name__ == '__main__':
    main()