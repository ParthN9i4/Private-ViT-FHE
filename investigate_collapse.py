"""
Investigation: Why does PolyGELU + PolyAttn + BatchNorm collapse?
==================================================================

Known results from substitution_ablation.py:
  Teacher (standard):                     69.73%
  C: BatchNorm only        (with KD)      82.28%   ← BatchNorm alone is GOOD
  D: PolyGELU + PolyAttn   (with KD)      71.84%   ← LayerNorm keeps it stable
  E: All three             (with KD)      11.06%   ← Adding BatchNorm kills it

The question: Why does replacing LayerNorm with BatchNorm break the model
ONLY when both PolyGELU and PolyAttn are also present?

Hypothesis: LayerNorm normalizes each token independently (per-token).
When PolyAttn produces unbounded attention outputs, some tokens get
very large activations. LayerNorm catches and fixes this per-token.
BatchNorm normalizes per-feature across the batch — it does NOT catch
individual tokens with exploded magnitudes.

EXPERIMENTS:
  Part 1: Confirm D vs E with 3 seeds (reproducibility check)
  Part 2: Track activation magnitudes through layers for D and E
  Part 3: Test fixes — can we make E work?
    Fix 1: Add LayerNorm after PolyAttn only (BatchNorm everywhere else)
    Fix 2: Clamp PolyAttn output to [-1, 1]
    Fix 3: Normalize PolyAttn output (divide by row sum)
    Fix 4: Use RMSNorm instead of LayerNorm (simpler, still per-token)

Usage: python investigate_collapse.py
Time:  ~3 hours on A6000
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import timm
import json
from pathlib import Path


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


# ── Polynomial modules ───────────────────────────────────────────────

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


class PolyAttn(nn.Module):
    """Standard polynomial attention — no normalization."""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.05))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.25))

    def forward(self, x):
        return self.a * x * x + self.b * x + self.c


class PolyAttnClamped(nn.Module):
    """Fix 2: Polynomial attention with output clamped to [0, 1]."""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.05))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.25))

    def forward(self, x):
        out = self.a * x * x + self.b * x + self.c
        return out.clamp(min=0.0, max=1.0)


class PolyAttnNormed(nn.Module):
    """Fix 3: Polynomial attention with row-wise normalization.
    After applying polynomial, divide each row by its sum (like softmax).
    Note: this division is expensive under CKKS (~20 levels), but we're
    testing whether normalization is the missing ingredient."""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.05))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.25))

    def forward(self, x):
        out = self.a * x * x + self.b * x + self.c
        out = out.clamp(min=1e-6)  # ensure positive before normalizing
        return out / (out.sum(dim=-1, keepdim=True) + 1e-8)


class RMSNorm(nn.Module):
    """Fix 4: RMSNorm — per-token normalization without mean subtraction.
    Simpler than LayerNorm: just divides by root-mean-square.
    Under CKKS: needs one polynomial division (~cheaper than full LayerNorm)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# ── Model ────────────────────────────────────────────────────────────

class InvestigationBlock(nn.Module):
    """Transformer block with configurable everything."""

    def __init__(self, dim, num_heads=3, mlp_ratio=4.0,
                 norm_type='layernorm', attn_type='poly', gelu_type='poly'):
        super().__init__()
        self.norm_type = norm_type

        # Norms
        self.norm1 = self._make_norm(norm_type, dim)
        self.norm2 = self._make_norm(norm_type, dim)

        # Attention
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_type = attn_type

        if attn_type == 'poly':
            self.attn_act = PolyAttn()
        elif attn_type == 'poly_clamped':
            self.attn_act = PolyAttnClamped()
        elif attn_type == 'poly_normed':
            self.attn_act = PolyAttnNormed()
        # 'standard' uses regular softmax — no module needed

        # FFN
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = PolyGELU() if gelu_type == 'poly' else nn.GELU()

        # For activation tracking
        self.last_attn_out_mag = 0.0
        self.last_ffn_out_mag = 0.0

    def _make_norm(self, norm_type, dim):
        if norm_type == 'batchnorm':
            return nn.BatchNorm1d(dim)
        elif norm_type == 'rmsnorm':
            return RMSNorm(dim)
        else:
            return nn.LayerNorm(dim)

    def _apply_norm(self, norm, x):
        if self.norm_type == 'batchnorm':
            return norm(x.transpose(1, 2)).transpose(1, 2)
        else:
            return norm(x)

    def forward(self, x):
        # Attention
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
        attn_out = self.proj(out)
        self.last_attn_out_mag = attn_out.detach().abs().mean().item()
        x = x + attn_out

        # FFN
        h = self._apply_norm(self.norm2, x)
        ffn_out = self.fc2(self.act(self.fc1(h)))
        self.last_ffn_out_mag = ffn_out.detach().abs().mean().item()
        x = x + ffn_out

        return x


class InvestigationModel(nn.Module):
    def __init__(self, num_classes=10, img_size=32, patch_size=4,
                 embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0,
                 norm_type='layernorm', attn_type='poly', gelu_type='poly'):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.norm_type = norm_type

        self.blocks = nn.ModuleList([
            InvestigationBlock(embed_dim, num_heads, mlp_ratio,
                               norm_type=norm_type, attn_type=attn_type,
                               gelu_type=gelu_type)
            for _ in range(depth)
        ])

        if norm_type == 'batchnorm':
            self.norm = nn.BatchNorm1d(embed_dim)
        elif norm_type == 'rmsnorm':
            self.norm = RMSNorm(embed_dim)
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

    def get_activation_magnitudes(self):
        """Return per-layer activation magnitudes from the last forward pass."""
        attn_mags = [blk.last_attn_out_mag for blk in self.blocks]
        ffn_mags = [blk.last_ffn_out_mag for blk in self.blocks]
        return attn_mags, ffn_mags


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


def train_and_track(model, train_loader, test_loader, device, epochs=100,
                    teacher=None, kd_alpha=0.1, kd_temp=4.0, label="",
                    track_activations=False):
    """Train model, optionally tracking activation magnitudes per epoch."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if teacher is not None:
        teacher = teacher.to(device)
        teacher.eval()

    best_acc = 0.0
    activation_log = []  # per-epoch activation magnitudes

    print(f"\n  Training: {label}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        epoch_attn_mags = []
        epoch_ffn_mags = []

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

            if track_activations:
                attn_m, ffn_m = model.get_activation_magnitudes()
                epoch_attn_mags.append(attn_m)
                epoch_ffn_mags.append(ffn_m)

        scheduler.step()
        acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, acc)

        if track_activations and epoch_attn_mags:
            # Average across batches
            n_layers = len(epoch_attn_mags[0])
            avg_attn = [sum(b[i] for b in epoch_attn_mags) / len(epoch_attn_mags)
                        for i in range(n_layers)]
            avg_ffn = [sum(b[i] for b in epoch_ffn_mags) / len(epoch_ffn_mags)
                       for i in range(n_layers)]
            activation_log.append({
                'epoch': epoch + 1,
                'attn_magnitudes': avg_attn,
                'ffn_magnitudes': avg_ffn,
                'acc': acc,
            })

        if (epoch + 1) % 20 == 0:
            mag_str = ""
            if track_activations and activation_log:
                last = activation_log[-1]
                mag_str = (f"  attn_mag=[{last['attn_magnitudes'][0]:.2f}→"
                           f"{last['attn_magnitudes'][-1]:.2f}]"
                           f"  ffn_mag=[{last['ffn_magnitudes'][0]:.2f}→"
                           f"{last['ffn_magnitudes'][-1]:.2f}]")
            print(f"    Ep {epoch+1:3d}  loss={total_loss/len(train_loader):.4f}"
                  f"  acc={acc:.2f}%  best={best_acc:.2f}%{mag_str}")

    print(f"  ► Best: {best_acc:.2f}%")
    return best_acc, activation_log


# ── Main ─────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    train_loader, test_loader = get_cifar10()
    epochs = 100

    # ── Train teacher once ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training teacher")
    print(f"{'='*60}")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    teacher = InvestigationModel(
        norm_type='layernorm', attn_type='standard', gelu_type='standard')
    teacher_acc, _ = train_and_track(
        teacher, train_loader, test_loader, device, epochs,
        label="Teacher (standard everything)")

    # ══════════════════════════════════════════════════════════════════
    # PART 1: Confirm D vs E with 3 seeds
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PART 1: Confirm D vs E across 3 seeds")
    print(f"{'='*60}")

    seeds = [42, 123, 456]
    d_accs = []
    e_accs = []

    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model_d = InvestigationModel(
            norm_type='layernorm', attn_type='poly', gelu_type='poly')
        acc_d, _ = train_and_track(
            model_d, train_loader, test_loader, device, epochs,
            teacher=teacher, label=f"D: PolyGELU+PolyAttn+LayerNorm (seed={seed})")
        d_accs.append(acc_d)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model_e = InvestigationModel(
            norm_type='batchnorm', attn_type='poly', gelu_type='poly')
        acc_e, _ = train_and_track(
            model_e, train_loader, test_loader, device, epochs,
            teacher=teacher, label=f"E: PolyGELU+PolyAttn+BatchNorm (seed={seed})")
        e_accs.append(acc_e)

    print(f"\n  D (LayerNorm): {[f'{a:.2f}' for a in d_accs]}  mean={sum(d_accs)/3:.2f}%")
    print(f"  E (BatchNorm): {[f'{a:.2f}' for a in e_accs]}  mean={sum(e_accs)/3:.2f}%")
    print(f"  Confirmed: {'YES — E consistently fails' if max(e_accs) < min(d_accs) else 'Mixed results'}")

    # ══════════════════════════════════════════════════════════════════
    # PART 2: Track activation magnitudes for D vs E
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PART 2: Activation magnitudes through layers (D vs E)")
    print(f"{'='*60}")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model_d = InvestigationModel(
        norm_type='layernorm', attn_type='poly', gelu_type='poly')
    _, d_log = train_and_track(
        model_d, train_loader, test_loader, device, epochs,
        teacher=teacher, track_activations=True,
        label="D with activation tracking")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model_e = InvestigationModel(
        norm_type='batchnorm', attn_type='poly', gelu_type='poly')
    _, e_log = train_and_track(
        model_e, train_loader, test_loader, device, epochs,
        teacher=teacher, track_activations=True,
        label="E with activation tracking")

    # Print comparison at key epochs
    print(f"\n  Activation magnitudes (mean absolute value of layer outputs):")
    print(f"  {'Epoch':>5} │ {'D attn L0→L5':<20} │ {'E attn L0→L5':<20} │ {'D ffn L0→L5':<16} │ E ffn L0→L5")
    print(f"  {'─'*5} │ {'─'*20} │ {'─'*20} │ {'─'*16} │ {'─'*15}")

    for d_entry, e_entry in zip(d_log, e_log):
        ep = d_entry['epoch']
        if ep in [1, 5, 10, 20, 50, 100]:
            da = d_entry['attn_magnitudes']
            ea = e_entry['attn_magnitudes']
            df = d_entry['ffn_magnitudes']
            ef = e_entry['ffn_magnitudes']
            d_attn = f"{da[0]:.1f}→{da[-1]:.1f}"
            e_attn = f"{ea[0]:.1f}→{ea[-1]:.1f}"
            d_ffn = f"{df[0]:.2f}→{df[-1]:.2f}"
            e_ffn = f"{ef[0]:.2f}→{ef[-1]:.2f}"
            print(f"  {ep:>5} │ {d_attn:<20} │ {e_attn:<20} │ {d_ffn:<16} │ {e_ffn}")

    # ══════════════════════════════════════════════════════════════════
    # PART 3: Test fixes
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PART 3: Can we fix config E?")
    print(f"{'='*60}")

    fixes = [
        # (label, norm_type, attn_type, gelu_type)
        ("Fix 1: PolyAttn clamped to [0,1] + BatchNorm",
         'batchnorm', 'poly_clamped', 'poly'),
        ("Fix 2: PolyAttn row-normalized + BatchNorm",
         'batchnorm', 'poly_normed', 'poly'),
        ("Fix 3: RMSNorm (per-token, simpler than LayerNorm)",
         'rmsnorm', 'poly', 'poly'),
    ]

    fix_results = {}
    for label, nt, at, gt in fixes:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model = InvestigationModel(
            norm_type=nt, attn_type=at, gelu_type=gt)
        acc, log = train_and_track(
            model, train_loader, test_loader, device, epochs,
            teacher=teacher, track_activations=True, label=label)
        fix_results[label] = acc

    # ══════════════════════════════════════════════════════════════════
    # FINAL RESULTS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Teacher (standard):                      {teacher_acc:.2f}%")
    print(f"")
    print(f"  Config D: PolyGELU+PolyAttn+LayerNorm    {sum(d_accs)/3:.2f}% (±{max(d_accs)-min(d_accs):.2f})")
    print(f"  Config E: PolyGELU+PolyAttn+BatchNorm    {sum(e_accs)/3:.2f}% (±{max(e_accs)-min(e_accs):.2f})")
    print(f"")
    print(f"  Fixes for Config E:")
    for label, acc in fix_results.items():
        status = "WORKS" if acc > 50 else "FAILS"
        print(f"    {label:<50} {acc:.2f}%  [{status}]")

    print(f"\n  CONCLUSION:")
    best_fix = max(fix_results, key=fix_results.get)
    best_fix_acc = fix_results[best_fix]

    if best_fix_acc > 50:
        print(f"  The all-polynomial model CAN work with: {best_fix}")
        print(f"  Best fix accuracy: {best_fix_acc:.2f}%")
        print(f"  The root cause is unbounded PolyAttn outputs compounding")
        print(f"  through layers without per-token normalization.")
    else:
        print(f"  None of the fixes recovered the all-polynomial model.")
        print(f"  The interaction between polynomial activations without")
        print(f"  any per-token normalization is fundamentally broken.")

    # ── Implications for CKKS ────────────────────────────────────────
    print(f"\n  IMPLICATIONS FOR FHE:")
    if fix_results.get("Fix 3: RMSNorm (per-token, simpler than LayerNorm)", 0) > 50:
        print(f"  RMSNorm works → use it. Cheaper than LayerNorm under CKKS")
        print(f"  (no mean subtraction needed, just division by RMS).")
    if fix_results.get("Fix 1: PolyAttn clamped to [0,1] + BatchNorm", 0) > 50:
        print(f"  Clamped PolyAttn works → the key is bounding attention outputs.")
        print(f"  Clamping under CKKS needs a polynomial min/max approximation.")
    if fix_results.get("Fix 2: PolyAttn row-normalized + BatchNorm", 0) > 50:
        print(f"  Normalized PolyAttn works → but adds division (Goldschmidt, ~20 levels).")
        print(f"  Still cheaper than full softmax (no max subtraction needed).")

    # Save results as lightweight JSON (not .pth — avoids disk space issues)
    summary = {
        'teacher_acc': teacher_acc,
        'd_accs': d_accs,
        'e_accs': e_accs,
        'fix_results': fix_results,
    }

    out_dir = Path('experiment_results')
    out_dir.mkdir(exist_ok=True)

    try:
        with open(out_dir / 'investigation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to experiment_results/investigation_summary.json")
    except Exception as e:
        print(f"\nWarning: could not save summary ({e})")
        print(f"Results are printed above — copy them manually.")

    try:
        with open(out_dir / 'activation_magnitudes.json', 'w') as f:
            json.dump({'d_log': d_log, 'e_log': e_log}, f, indent=2)
        print(f"Activation logs: experiment_results/activation_magnitudes.json")
    except Exception as e:
        print(f"Warning: could not save activation logs ({e})")


if __name__ == '__main__':
    main()