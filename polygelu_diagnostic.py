"""
PolyGELU Activation Input Diagnostic
=====================================

This script answers: "Is [-3, 3] the right fitting interval for our PolyGELU?"

It trains our best model (Fix 2 with KD), then runs all training images
through it with hooks on every PolyGELU layer, recording every single
input value. It then produces:

1. Per-layer histograms of input values
2. Coverage statistics: what % of values fall within [-3, 3]?
3. Percentile analysis: 95th, 99th, 99.9th percentile
4. Per-layer comparison: do early vs late layers differ?
5. Recommendation: is [-3, 3] validated, too narrow, or too wide?

This is the empirical evidence needed to defend the fitting interval
in a paper or to a reviewer.

Usage:
  python polylgelu_diagnostic.py                # Train from scratch + diagnose
  python polylgelu_diagnostic.py --skip-train   # Load saved model, diagnose only
  python polylgelu_diagnostic.py --max-batches 50  # Quick test (50 batches)
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import json
import math


# ══════════════════════════════════════════════════════════════════════
# MODEL (same as verify_fixes.py — Fix 2 configuration)
# ══════════════════════════════════════════════════════════════════════

class PolyGELU(nn.Module):
    def __init__(self):
        super().__init__()
        x = torch.linspace(-3, 3, 10000)
        y = F.gelu(x)
        X = torch.stack([x**2, x, torch.ones_like(x)], dim=1)
        c = torch.linalg.lstsq(X, y).solution
        self.a = nn.Parameter(c[0].clone())
        self.b = nn.Parameter(c[1].clone())
        self.c = nn.Parameter(c[2].clone())

    def forward(self, x):
        return self.a * x * x + self.b * x + self.c


class PolyAttnNormed(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.05))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.25))

    def forward(self, x):
        out = self.a * x * x + self.b * x + self.c
        out = out.clamp(min=1e-6)
        return out / (out.sum(dim=-1, keepdim=True) + 1e-8)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class Block(nn.Module):
    def __init__(self, dim, num_heads=3, mlp_ratio=4.0,
                 norm_type='batchnorm', attn_type='poly_normed',
                 gelu_type='poly'):
        super().__init__()
        self.norm_type = norm_type
        self.norm1 = self._make_norm(norm_type, dim)
        self.norm2 = self._make_norm(norm_type, dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_type = attn_type
        if attn_type == 'poly_normed':
            self.attn_act = PolyAttnNormed()
        self.act = PolyGELU() if gelu_type == 'poly' else nn.GELU()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def _make_norm(self, nt, dim):
        if nt == 'batchnorm': return nn.BatchNorm1d(dim)
        elif nt == 'rmsnorm': return RMSNorm(dim)
        else: return nn.LayerNorm(dim)

    def _apply_norm(self, norm, x):
        if self.norm_type == 'batchnorm':
            return norm(x.transpose(1, 2)).transpose(1, 2)
        return norm(x)

    def forward(self, x):
        h = self._apply_norm(self.norm1, x)
        B, N, C = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.attn_type == 'standard':
            attn = attn.softmax(dim=-1)
        else:
            attn = self.attn_act(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + self.proj(out)
        h = self._apply_norm(self.norm2, x)
        x = x + self.fc2(self.act(self.fc1(h)))
        return x


class DeiTTiny(nn.Module):
    def __init__(self, num_classes=10, img_size=32, patch_size=4,
                 embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0,
                 norm_type='batchnorm', attn_type='poly_normed',
                 gelu_type='poly'):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.norm_type = norm_type
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio,
                  norm_type=norm_type, attn_type=attn_type,
                  gelu_type=gelu_type)
            for _ in range(depth)
        ])
        if norm_type == 'batchnorm':
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
        if self.norm_type == 'batchnorm':
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.norm(x)
        return self.head(x[:, 0])


# ══════════════════════════════════════════════════════════════════════
# TRAINING (same as verify_fixes.py)
# ══════════════════════════════════════════════════════════════════════

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
    train = torchvision.datasets.CIFAR10('./data', train=True,
                                         download=True, transform=t_train)
    test = torchvision.datasets.CIFAR10('./data', train=False,
                                        download=True, transform=t_test)
    return (DataLoader(train, batch_size=batch_size, shuffle=True,
                       num_workers=4, pin_memory=True),
            DataLoader(test, batch_size=batch_size, shuffle=False,
                       num_workers=4, pin_memory=True))


class StandardDeiT(nn.Module):
    """Standard DeiT-Tiny teacher (GELU + softmax + LayerNorm)."""
    def __init__(self, num_classes=10, embed_dim=192, depth=6):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, 4, 4)
        num_patches = 64
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            blk = Block(embed_dim, norm_type='layernorm',
                        attn_type='standard', gelu_type='standard')
            self.blocks.append(blk)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def train_model(model, train_loader, test_loader, device, epochs=100,
                teacher=None, label=""):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if teacher is not None:
        teacher = teacher.to(device)
        teacher.eval()
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            ce = F.cross_entropy(logits, labels)
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(imgs)
                kd = F.kl_div(F.log_softmax(logits / 4.0, dim=-1),
                              F.softmax(t_logits / 4.0, dim=-1),
                              reduction='batchmean')
                loss = 0.1 * kd + 0.9 * ce
            else:
                loss = ce
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += model(imgs).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        acc = 100.0 * correct / total
        best_acc = max(best_acc, acc)
        if (epoch + 1) % 25 == 0:
            print(f"    [{label}] Ep {epoch+1}/{epochs}  acc={acc:.2f}%  best={best_acc:.2f}%")
    return best_acc


# ══════════════════════════════════════════════════════════════════════
# ACTIVATION DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════

def run_diagnostic(model, train_loader, test_loader, device, max_batches=0):
    """
    Hook into every PolyGELU layer and record all input values.
    Run on both train and test sets for complete coverage.
    """
    model = model.to(device)
    model.eval()

    # Find all PolyGELU modules and register hooks
    layer_inputs = {}
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, PolyGELU):
            layer_name = name
            layer_inputs[layer_name] = []

            def make_hook(ln):
                def hook_fn(module, input, output):
                    # input is a tuple, take the first element
                    x = input[0].detach().cpu()
                    layer_inputs[ln].append(x.reshape(-1))
                return hook_fn

            h = module.register_forward_hook(make_hook(layer_name))
            hooks.append(h)

    print(f"  Registered hooks on {len(layer_inputs)} PolyGELU layers")

    # Run forward pass on training set
    print(f"  Running forward pass on training set...")
    n_batches = 0
    with torch.no_grad():
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            _ = model(imgs)
            n_batches += 1
            if max_batches > 0 and n_batches >= max_batches:
                break
            if n_batches % 50 == 0:
                print(f"    Batch {n_batches}...")

    # Run forward pass on test set
    print(f"  Running forward pass on test set...")
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            _ = model(imgs)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Concatenate all values per layer
    for name in layer_inputs:
        layer_inputs[name] = torch.cat(layer_inputs[name], dim=0)

    return layer_inputs


def analyze_layer(name, values, intervals):
    """Analyze activation values for one PolyGELU layer."""
    n = len(values)
    v_min = values.min().item()
    v_max = values.max().item()
    v_mean = values.mean().item()
    v_std = values.std().item()

    # Percentiles
    sorted_abs = values.abs().sort().values
    p95 = sorted_abs[int(0.95 * n)].item()
    p99 = sorted_abs[int(0.99 * n)].item()
    p999 = sorted_abs[int(0.999 * n)].item()
    p9999 = sorted_abs[min(int(0.9999 * n), n - 1)].item()

    # Coverage for each interval
    coverage = {}
    for lo, hi in intervals:
        in_range = ((values >= lo) & (values <= hi)).float().mean().item() * 100
        coverage[f"[{lo},{hi}]"] = round(in_range, 4)

    # Count outliers beyond [-3, 3]
    beyond_3 = ((values < -3) | (values > 3)).float().mean().item() * 100
    beyond_5 = ((values < -5) | (values > 5)).float().mean().item() * 100

    result = {
        "layer": name,
        "n_values": n,
        "min": round(v_min, 4),
        "max": round(v_max, 4),
        "mean": round(v_mean, 4),
        "std": round(v_std, 4),
        "abs_p95": round(p95, 4),
        "abs_p99": round(p99, 4),
        "abs_p999": round(p999, 4),
        "abs_p9999": round(p9999, 4),
        "coverage": coverage,
        "pct_beyond_3": round(beyond_3, 4),
        "pct_beyond_5": round(beyond_5, 4),
    }
    return result


def print_report(results):
    """Print the diagnostic report."""
    intervals = [(-2, 2), (-3, 3), (-4, 4), (-5, 5)]

    print(f"\n{'='*80}")
    print("POLYGELU INPUT DIAGNOSTIC REPORT")
    print(f"{'='*80}")

    for r in results:
        print(f"\n  Layer: {r['layer']}")
        print(f"  Total values: {r['n_values']:,}")
        print(f"  Range: [{r['min']:.4f}, {r['max']:.4f}]")
        print(f"  Mean: {r['mean']:.4f}, Std: {r['std']:.4f}")
        print(f"  |x| percentiles: 95th={r['abs_p95']:.3f}  "
              f"99th={r['abs_p99']:.3f}  "
              f"99.9th={r['abs_p999']:.3f}  "
              f"99.99th={r['abs_p9999']:.3f}")
        print(f"  Coverage:")
        for interval, pct in r['coverage'].items():
            marker = " <-- our fit" if interval == "[-3,3]" else ""
            print(f"    {interval}: {pct:.4f}%{marker}")
        print(f"  Beyond [-3,3]: {r['pct_beyond_3']:.4f}%")
        print(f"  Beyond [-5,5]: {r['pct_beyond_5']:.4f}%")

    # Overall summary
    print(f"\n{'='*80}")
    print("SUMMARY AND RECOMMENDATION")
    print(f"{'='*80}")

    all_beyond_3 = [r['pct_beyond_3'] for r in results]
    max_beyond_3 = max(all_beyond_3)
    worst_layer = results[all_beyond_3.index(max_beyond_3)]['layer']
    all_p999 = [r['abs_p999'] for r in results]
    max_p999 = max(all_p999)

    print(f"\n  Worst-case layer: {worst_layer}")
    print(f"  Max % beyond [-3,3]: {max_beyond_3:.4f}%")
    print(f"  Max 99.9th percentile |x|: {max_p999:.3f}")

    if max_beyond_3 < 0.01:
        print(f"\n  VERDICT: [-3, 3] is STRONGLY VALIDATED.")
        print(f"  Less than 0.01% of values fall outside the fitting interval.")
        print(f"  The polynomial approximation covers virtually all inputs.")
    elif max_beyond_3 < 0.1:
        print(f"\n  VERDICT: [-3, 3] is VALIDATED with minor caveat.")
        print(f"  Less than 0.1% of values fall outside. Acceptable for practical use.")
        print(f"  Consider [-4, 4] for extra safety margin.")
    elif max_beyond_3 < 1.0:
        print(f"\n  VERDICT: [-3, 3] is MARGINAL.")
        print(f"  Up to {max_beyond_3:.2f}% of values fall outside the fitting interval.")
        print(f"  Recommend refitting on [-4, 4] or [-5, 5] and retraining.")
    else:
        print(f"\n  VERDICT: [-3, 3] is INSUFFICIENT.")
        print(f"  {max_beyond_3:.2f}% of values exceed the fitting interval.")
        print(f"  The polynomial approximation is invalid for these inputs.")
        print(f"  Must refit on a wider interval and retrain.")

    # Trained coefficients report
    print(f"\n{'='*80}")
    print("TRAINED POLYGELU COEFFICIENTS (per layer)")
    print(f"{'='*80}")
    print(f"  {'Layer':<35} {'a':>8} {'b':>8} {'c':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")


def print_coefficients(model):
    """Print the trained PolyGELU coefficients per layer."""
    for name, module in model.named_modules():
        if isinstance(module, PolyGELU):
            a = module.a.item()
            b = module.b.item()
            c = module.c.item()
            print(f"  {name:<35} {a:>8.4f} {b:>8.4f} {c:>8.4f}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, load saved model")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Limit training batches for diagnosis (0=all)")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    train_loader, test_loader = get_cifar10()

    save_dir = Path("ckks_models")
    save_dir.mkdir(exist_ok=True)

    # ── Train or load models ─────────────────────────────────────────
    teacher_path = save_dir / "teacher_diag.pth"
    model_path = save_dir / "fix2_diag.pth"

    if args.skip_train and model_path.exists() and teacher_path.exists():
        print("Loading saved models...")
        teacher = StandardDeiT()
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        model = DeiTTiny()
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # Train teacher
        print("\nTraining teacher...")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        teacher = StandardDeiT()
        acc = train_model(teacher, train_loader, test_loader, device,
                          args.epochs, label="Teacher")
        print(f"  Teacher: {acc:.2f}%")
        try:
            torch.save(teacher.state_dict(), teacher_path)
        except Exception:
            pass

        # Train Fix 2
        print("\nTraining Fix 2 (normalized PolyAttn + BatchNorm + KD)...")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model = DeiTTiny()
        acc = train_model(model, train_loader, test_loader, device,
                          args.epochs, teacher=teacher, label="Fix2")
        print(f"  Fix 2: {acc:.2f}%")
        try:
            torch.save(model.state_dict(), model_path)
        except Exception:
            pass

    # ── Run diagnostic ───────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("RUNNING ACTIVATION DIAGNOSTIC")
    print(f"{'='*80}")

    layer_inputs = run_diagnostic(model, train_loader, test_loader, device,
                                  args.max_batches)

    # Analyze each layer
    intervals = [(-2, 2), (-3, 3), (-4, 4), (-5, 5)]
    results = []
    for name, values in layer_inputs.items():
        r = analyze_layer(name, values, intervals)
        results.append(r)

    # Print report
    print_report(results)
    print_coefficients(model)

    # Also show initial vs trained coefficients
    print(f"\n  Initial LSQ fit: a=0.1740, b=0.5000, c=0.1450")
    print(f"  (Compare above to see how much each layer drifted)")

    # Save results
    try:
        out_dir = Path("experiment_results")
        out_dir.mkdir(exist_ok=True)
        # Convert results to JSON-safe format
        for r in results:
            r['n_values'] = int(r['n_values'])
        with open(out_dir / "polygelu_diagnostic.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved: experiment_results/polygelu_diagnostic.json")
    except Exception as e:
        print(f"\n  (Could not save results: {e})")


if __name__ == "__main__":
    main()