"""
fix2_bloodmnist.py
==================
Medical-dataset validation of the polynomial-ViT interaction-collapse finding.
Replaces the RetinaMNIST attempt, which was underpowered (1,080 training
samples, teacher stuck at 34.7% balanced accuracy, no usable KD signal).

WHY BLOODMNIST
--------------
  Samples    : 11,959 train / 1,712 val / 3,421 test   (~11x more than RetinaMNIST)
  Classes    : 8       (blood cell types: neutrophil, eosinophil, lymphocyte, ...)
  Imbalance  : mild    (vs. RetinaMNIST's severe imbalance)
  Baseline   : MedMNIST paper reports ~95% with ResNet-18; DeiT-Tiny teacher
               at 32x32 should reach 80-90%. This is the level of teacher
               performance that makes KD signal meaningful.

WHAT THIS SCRIPT TESTS
----------------------
The same three configurations as verify_fixes.py (CIFAR-10) and the failed
fix2_retinamnist.py:
  Teacher         : LayerNorm + softmax + GELU    (standard)
  Config_E        : BatchNorm + PolyAttn + PolyGELU + KD   (expected: collapse)
  Fix2_Normalized : BatchNorm + PolyAttnNormed + PolyGELU + KD (expected: recover)

The architecture is bit-identical to verify_fixes.py: 32x32 input, patch 4,
6 layers, 192 dim, 3 heads, T=4, alpha=0.1, no T^2 scaling, grad clip 5.0,
seeds 42-46.

CHANGES FROM fix2_retinamnist.py
--------------------------------
1. BloodMNIST loader replaces RetinaMNIST. Everything else stays.
2. num_classes = 8 (was 5).
3. Augmentations are blood-cell-appropriate: horizontal AND vertical flips
   are safe (cells have no canonical orientation, unlike fundus images where
   macula position fixes orientation). Rotation added for the same reason.
4. Collapse-detection threshold in the interpretation block is slightly
   stricter because BloodMNIST baseline is high; we need to distinguish
   'mild degradation' from 'actual collapse'.

USAGE
-----
  # Smoke test (1 seed, 30 epochs, ~5 min)
  python fix2_bloodmnist.py --seeds 42 --epochs 30

  # Full run (5 seeds x 3 configs = 15 runs, ~2-3 hours on A6000)
  python fix2_bloodmnist.py

  # Reuse checkpoints if present
  python fix2_bloodmnist.py --skip-train

EXPECTED OUTCOMES
-----------------
If the CIFAR-10 finding reproduces on medical data:
  Teacher         : 80-90% test_acc, 75-85% bal_acc
  Config_E        : significantly below teacher (collapse)
  Fix2_Normalized : within 2-3pp of teacher (recovery)

If it does NOT reproduce:
  Config_E and Fix2 perform similarly and both near teacher
  -> The collapse may be sensitive to dataset structure; run
     activation tracking on BloodMNIST to compare per-layer magnitudes.

Panchan Nagar, SSSIHL / IIT-H / April 2026
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    import medmnist
    from medmnist import INFO
except ImportError as e:
    raise ImportError("pip install medmnist  # see https://medmnist.com") from e

try:
    from scipy import stats as scipy_stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("[WARN] scipy not available; Welch's t-tests will be skipped.")


# =====================================================================
# 0. UTILITIES
# =====================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# 1. POLYNOMIAL ACTIVATIONS
# =====================================================================

class PolyGELU(nn.Module):
    """Degree-2 polynomial replacement for GELU: f(x) = a*x^2 + b*x + c.

    LSQ fit to GELU on [-3, 3] with 10,000 samples -> (a, b, c) ~= (0.174, 0.500, 0.145).
    Observed after training (polygelu_diagnostic.py on CIFAR-10): b collapses
    from 0.500 to ~0.01 across all layers. Trained polynomial is near-pure
    quadratic, NOT a GELU approximation.
    """

    def __init__(self, fit_range=(-3.0, 3.0), num_samples: int = 10000):
        super().__init__()
        xs = torch.linspace(fit_range[0], fit_range[1], num_samples)
        ys = F.gelu(xs)
        X = torch.stack([xs ** 2, xs, torch.ones_like(xs)], dim=1)
        coeffs = torch.linalg.lstsq(X, ys.unsqueeze(1)).solution.squeeze()
        self.a = nn.Parameter(coeffs[0].clone())
        self.b = nn.Parameter(coeffs[1].clone())
        self.c = nn.Parameter(coeffs[2].clone())

    def forward(self, x):
        return self.a * x * x + self.b * x + self.c


class PolyAttn(nn.Module):
    """Config E's broken attention: element-wise polynomial, no normalization."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.05))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.25))

    def forward(self, scores):
        return self.a * scores * scores + self.b * scores + self.c


class PolyAttnNormed(nn.Module):
    """Fix 2: polynomial attention with ReLU + row normalization.

    poly(s) = a*s^2 + b*s + c;  clipped = ReLU(poly(s));  out = clipped / (rowsum + eps)
    CKKS cost: 1 level (polynomial) + ~20 levels (row division).
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.05))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.25))
        self.eps = eps

    def forward(self, scores):
        poly = self.a * scores * scores + self.b * scores + self.c
        poly = F.relu(poly)
        row_sum = poly.sum(dim=-1, keepdim=True) + self.eps
        return poly / row_sum


# =====================================================================
# 2. CKKS-FRIENDLY NORMALIZATION
# =====================================================================

class TokenBatchNorm(nn.Module):
    """BatchNorm over (batch * tokens) for each feature.

    At CKKS inference: y = scale * x + shift (precomputed), 0 levels.
    Tradeoff: normalizes per-feature across (batch, token), missing
    per-token outliers -- root cause of the all-polynomial collapse.
    """

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim, eps=eps, momentum=momentum)

    def forward(self, x):
        B, N, D = x.shape
        x = x.reshape(B * N, D)
        x = self.bn(x)
        return x.reshape(B, N, D)


# =====================================================================
# 3. TRANSFORMER BLOCK
# =====================================================================

class Block(nn.Module):
    """ViT encoder block with plug-in norm / attn / gelu."""

    def __init__(self, dim: int = 192, num_heads: int = 3, mlp_ratio: int = 4,
                 norm_type: str = "layernorm", attn_type: str = "standard",
                 gelu_type: str = "standard"):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_type = attn_type

        if norm_type == "layernorm":
            self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        elif norm_type == "batchnorm":
            self.norm1, self.norm2 = TokenBatchNorm(dim), TokenBatchNorm(dim)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        if attn_type == "standard":
            self.attn_act = None
        elif attn_type == "poly":
            self.attn_act = PolyAttn()
        elif attn_type == "poly_normed":
            self.attn_act = PolyAttnNormed()
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

        hidden = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        if gelu_type == "standard":
            self.act = nn.GELU()
        elif gelu_type == "poly":
            self.act = PolyGELU()
        else:
            raise ValueError(f"Unknown gelu_type: {gelu_type}")

    def forward(self, x):
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) * self.scale
        if self.attn_type == "standard":
            attn = F.softmax(scores, dim=-1)
        else:
            attn = self.attn_act(scores)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        x = x + out

        h = self.norm2(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        x = x + h
        return x


# =====================================================================
# 4. MODEL
# =====================================================================

class DeiTTiny(nn.Module):
    """6-layer DeiT-Tiny. Architecture identical to verify_fixes.py
    except num_classes=8 for BloodMNIST."""

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_channels: int = 3, num_classes: int = 8,
                 embed_dim: int = 192, depth: int = 6, num_heads: int = 3,
                 norm_type: str = "layernorm", attn_type: str = "standard",
                 gelu_type: str = "standard"):
        super().__init__()
        self.num_classes = num_classes
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads,
                  norm_type=norm_type, attn_type=attn_type, gelu_type=gelu_type)
            for _ in range(depth)
        ])

        if norm_type == "layernorm":
            self.norm = nn.LayerNorm(embed_dim)
        elif norm_type == "batchnorm":
            self.norm = TokenBatchNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])


# =====================================================================
# 5. DATA: BloodMNIST
# =====================================================================

def get_bloodmnist_loaders(batch_size: int = 128, num_workers: int = 2,
                           data_root: str = "./data"):
    """BloodMNIST resized 28 -> 32 for architecture match with CIFAR-10.

    Augmentation notes (different from RetinaMNIST):
      - Horizontal AND vertical flips are both safe: blood cells in
        microscopy have no canonical orientation. (In fundus images,
        vertical flip is NOT safe because macula position anchors 'up'.)
      - Random rotation is safe for the same reason.
      - Color jitter kept mild: H&E stain variation matters but cell
        morphology should dominate.
    Batch size raised to 128 (from 64) because the train set is larger;
    this also means fewer optimizer steps per epoch -> more epochs matter
    less. 150 epochs remains appropriate.
    """
    info = INFO["bloodmnist"]
    DataClass = getattr(medmnist, info["python_class"])

    train_tf = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

    train_ds = DataClass(split="train", transform=train_tf, download=True, root=data_root)
    val_ds = DataClass(split="val", transform=eval_tf, download=True, root=data_root)
    test_ds = DataClass(split="test", transform=eval_tf, download=True, root=data_root)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, drop_last=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


# =====================================================================
# 6. KD LOSS
# =====================================================================

def kd_loss_fn(s_logits, t_logits, T: float = 4.0):
    """KL divergence (no T^2 scaling -- T^2=16 drowned out CE on CIFAR-100)."""
    s_log = F.log_softmax(s_logits / T, dim=-1)
    t_prob = F.softmax(t_logits / T, dim=-1)
    return F.kl_div(s_log, t_prob, reduction="batchmean")


def compute_loss(s_logits, t_logits, labels, T: float = 4.0, alpha: float = 0.1):
    ce = F.cross_entropy(s_logits, labels)
    if t_logits is None:
        return ce, ce.item(), 0.0
    kd = kd_loss_fn(s_logits, t_logits, T)
    total = alpha * kd + (1 - alpha) * ce
    return total, ce.item(), kd.item()


# =====================================================================
# 7. EVAL / TRAIN LOOPS
# =====================================================================

@torch.no_grad()
def evaluate(model, loader, device):
    """Return (accuracy, balanced_accuracy) on loader.
    Balanced accuracy is the honest metric for (mildly) imbalanced classes."""
    model.eval()
    all_p, all_y = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.squeeze(-1).long().to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=-1)
        all_p.append(preds.cpu())
        all_y.append(labels.cpu())
    preds = torch.cat(all_p).numpy()
    labels = torch.cat(all_y).numpy()
    acc = float((preds == labels).mean())
    classes = np.unique(labels)
    recalls = [(preds[labels == c] == c).mean() for c in classes if (labels == c).sum() > 0]
    bal_acc = float(np.mean(recalls)) if recalls else 0.0
    return acc, bal_acc


def train_one(config_name, config, seed, train_loader, val_loader, test_loader,
              teacher_model=None, epochs: int = 150, lr: float = 1e-3,
              wd: float = 0.05, device: str = "cuda",
              ckpt_dir: str = "./checkpoints_bloodmnist", print_every: int = 25):
    """Train one (config, seed) and evaluate on test set."""
    set_seed(seed)

    model = DeiTTiny(
        img_size=32, patch_size=4, in_channels=3, num_classes=8,
        embed_dim=192, depth=6, num_heads=3,
        norm_type=config["norm"], attn_type=config["attn"], gelu_type=config["gelu"],
    ).to(device)

    if teacher_model is not None and config["kd"]:
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val, best_bal, best_epoch, best_state = 0.0, 0.0, -1, None
    print(f"\n=== Training {config_name} seed={seed} ===")
    print(f"    norm={config['norm']} attn={config['attn']} "
          f"gelu={config['gelu']} kd={config['kd']}")
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.squeeze(-1).long().to(device)

            optimizer.zero_grad()
            s_logits = model(imgs)

            if teacher_model is not None and config["kd"]:
                with torch.no_grad():
                    t_logits = teacher_model(imgs)
                loss, _, _ = compute_loss(s_logits, t_logits, labels, T=4.0, alpha=0.1)
            else:
                loss, _, _ = compute_loss(s_logits, None, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Use balanced accuracy for model selection (imbalance-aware)
        val_acc, val_bal = evaluate(model, val_loader, device)
        if val_bal > best_bal:
            best_bal = val_bal
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % print_every == 0 or epoch == 0:
            elapsed = time.time() - t_start
            avg_loss = total_loss / max(n_batches, 1)
            print(f"  epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f} | "
                  f"val_acc={val_acc*100:.2f}% | val_bal={val_bal*100:.2f}% | "
                  f"best_bal={best_bal*100:.2f}%@{best_epoch+1} | {elapsed:.0f}s")

    model.load_state_dict(best_state)
    test_acc, test_bal_acc = evaluate(model, test_loader, device)
    elapsed = time.time() - t_start
    print(f"  FINAL: test_acc={test_acc*100:.2f}% | "
          f"test_bal={test_bal_acc*100:.2f}% | {elapsed:.0f}s")

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{config_name}_seed{seed}.pt")
    torch.save({
        "state_dict": best_state,
        "config": config,
        "seed": seed,
        "test_acc": test_acc,
        "test_bal_acc": test_bal_acc,
        "best_val": best_val,
        "best_val_bal": best_bal,
        "best_epoch": best_epoch,
    }, ckpt_path)

    return {
        "config_name": config_name, "seed": seed,
        "test_acc": test_acc, "test_bal_acc": test_bal_acc,
        "best_val_acc": best_val, "best_val_bal": best_bal,
        "best_epoch": best_epoch,
        "total_epochs": epochs, "time_seconds": elapsed,
        "ckpt_path": ckpt_path,
    }


def load_checkpoint(ckpt_path: str, device: str):
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = obj["config"]
    model = DeiTTiny(
        img_size=32, patch_size=4, in_channels=3, num_classes=8,
        embed_dim=192, depth=6, num_heads=3,
        norm_type=cfg["norm"], attn_type=cfg["attn"], gelu_type=cfg["gelu"],
    ).to(device)
    model.load_state_dict(obj["state_dict"])
    return model, obj


# =====================================================================
# 8. MAIN ORCHESTRATION
# =====================================================================

CONFIGS = {
    "Teacher":         {"norm": "layernorm", "attn": "standard",    "gelu": "standard", "kd": False},
    "Config_E":        {"norm": "batchnorm", "attn": "poly",        "gelu": "poly",     "kd": True},
    "Fix2_Normalized": {"norm": "batchnorm", "attn": "poly_normed", "gelu": "poly",     "kd": True},
}


def summarize_results(all_results: dict) -> dict:
    summary = {"per_config": {}, "tests": {}}
    for name, runs in all_results.items():
        accs = np.array([r["test_acc"] for r in runs])
        bal = np.array([r["test_bal_acc"] for r in runs])
        summary["per_config"][name] = {
            "test_acc_mean": float(accs.mean()),
            "test_acc_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "test_acc_min": float(accs.min()),
            "test_acc_max": float(accs.max()),
            "test_acc_seeds": accs.tolist(),
            "test_bal_acc_mean": float(bal.mean()),
            "test_bal_acc_std": float(bal.std(ddof=1)) if len(bal) > 1 else 0.0,
            "test_bal_acc_seeds": bal.tolist(),
            "n_seeds": int(len(accs)),
        }
    if HAVE_SCIPY:
        pairs = [
            ("Fix2_Normalized", "Config_E"),
            ("Fix2_Normalized", "Teacher"),
            ("Config_E",        "Teacher"),
        ]
        for a, b in pairs:
            if a in all_results and b in all_results:
                acc_a = np.array([r["test_acc"] for r in all_results[a]])
                acc_b = np.array([r["test_acc"] for r in all_results[b]])
                bal_a = np.array([r["test_bal_acc"] for r in all_results[a]])
                bal_b = np.array([r["test_bal_acc"] for r in all_results[b]])
                if len(acc_a) > 1 and len(acc_b) > 1:
                    t_acc, p_acc = scipy_stats.ttest_ind(acc_a, acc_b, equal_var=False)
                    t_bal, p_bal = scipy_stats.ttest_ind(bal_a, bal_b, equal_var=False)
                    summary["tests"][f"{a}_vs_{b}"] = {
                        "acc_t": float(t_acc), "acc_p": float(p_acc),
                        "acc_diff_pp": float((acc_a.mean() - acc_b.mean()) * 100),
                        "bal_t": float(t_bal), "bal_p": float(p_bal),
                        "bal_diff_pp": float((bal_a.mean() - bal_b.mean()) * 100),
                    }
    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 76)
    print("FINAL RESULTS SUMMARY: BloodMNIST (8-class blood cell morphology)")
    print("=" * 76)
    for name, s in summary["per_config"].items():
        print(f"\n{name}  (n={s['n_seeds']} seeds):")
        print(f"  test_acc : {s['test_acc_mean']*100:6.2f}% +/- {s['test_acc_std']*100:.2f}%  "
              f"[{s['test_acc_min']*100:.2f}%, {s['test_acc_max']*100:.2f}%]")
        print(f"  bal_acc  : {s['test_bal_acc_mean']*100:6.2f}% +/- {s['test_bal_acc_std']*100:.2f}%")
        print(f"  seeds(acc): {[f'{a*100:.2f}' for a in s['test_acc_seeds']]}")

    if summary["tests"]:
        print("\nWelch's t-tests (two-sided):")
        for k, t in summary["tests"].items():
            sig_acc = "  *" if t["acc_p"] < 0.05 else ""
            sig_bal = "  *" if t["bal_p"] < 0.05 else ""
            print(f"  {k}")
            print(f"    acc : dAcc={t['acc_diff_pp']:+6.2f}pp  t={t['acc_t']:+.3f}  "
                  f"p={t['acc_p']:.4f}{sig_acc}")
            print(f"    bal : dBal={t['bal_diff_pp']:+6.2f}pp  t={t['bal_t']:+.3f}  "
                  f"p={t['bal_p']:.4f}{sig_bal}")

    # Interpretation panel -- uses bal_acc (imbalance-aware)
    # BloodMNIST teacher is expected to reach 80-90% bal_acc; threshold
    # for 'learning happened' is 60% bal_acc (well above random 12.5%).
    t_bal = summary["per_config"]["Teacher"]["test_bal_acc_mean"]
    e_bal = summary["per_config"]["Config_E"]["test_bal_acc_mean"]
    f_bal = summary["per_config"]["Fix2_Normalized"]["test_bal_acc_mean"]

    print("\nInterpretation:")
    if t_bal < 0.6:
        # Guardrail: if teacher itself didn't learn, all comparisons are meaningless
        print(f"  -> WARNING: teacher bal_acc={t_bal*100:.1f}% is too low to anchor KD.")
        print("     Re-check training recipe before interpreting student configs.")
        print("     (BloodMNIST DeiT-Tiny should reach 75-85% bal_acc.)")
    else:
        collapsed = e_bal < 0.75 * t_bal  # Config E >=25% below teacher
        recovered = f_bal >= 0.93 * t_bal  # Fix 2 within 7% of teacher
        if collapsed and recovered:
            print("  -> Config E collapsed AND Fix 2 recovered.")
            print("     The CIFAR-10 finding REPRODUCES on medical data.")
            print(f"     Teacher {t_bal*100:.1f}% | Config E {e_bal*100:.1f}% | "
                  f"Fix 2 {f_bal*100:.1f}% (balanced).")
        elif collapsed and not recovered:
            print("  -> Config E collapsed; Fix 2 partial recovery.")
            print("     Row normalization helps but something else is also at play.")
            print("     Diagnostic to run: investigate_collapse.py on BloodMNIST.")
        elif not collapsed and recovered:
            print("  -> Config E did NOT collapse; Fix 2 matches teacher.")
            print("     Hypothesis: the collapse is sensitive to data statistics")
            print("     (natural images w/ low per-token variance vs. medical).")
            print("     Run activation tracking on BloodMNIST to verify this directly.")
        else:
            print("  -> No collapse, no clear differentiation between configs.")
            print("     All three configs train to similar accuracy -- different")
            print("     behavior from CIFAR-10. Worth investigating per-layer")
            print("     activation norms to understand the difference.")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--teacher-seed", type=int, default=42)
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints_bloodmnist")
    parser.add_argument("--results-dir", type=str, default="./experiment_results")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device        : {device}")
    print(f"Seeds         : {args.seeds}")
    print(f"Epochs        : {args.epochs}   LR: {args.lr}   WD: {args.wd}   BS: {args.batch_size}")
    print(f"Teacher-for-KD: seed={args.teacher_seed} (shared across all students)")
    print(f"Ckpt dir      : {args.ckpt_dir}")

    train_loader, val_loader, test_loader = get_bloodmnist_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        data_root=args.data_root,
    )
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  "
          f"Test: {len(test_loader.dataset)}")

    all_results = {name: [] for name in CONFIGS}

    # PHASE 1: Teacher at all seeds
    for seed in args.seeds:
        ckpt = os.path.join(args.ckpt_dir, f"Teacher_seed{seed}.pt")
        if args.skip_train and os.path.exists(ckpt):
            obj = torch.load(ckpt, map_location="cpu", weights_only=False)
            print(f"\n[SKIP] Teacher seed={seed}  "
                  f"(cached test_acc={obj['test_acc']*100:.2f}%, "
                  f"bal={obj['test_bal_acc']*100:.2f}%)")
            all_results["Teacher"].append({
                "config_name": "Teacher", "seed": seed,
                "test_acc": obj["test_acc"], "test_bal_acc": obj["test_bal_acc"],
                "best_val_acc": obj["best_val"], "best_val_bal": obj.get("best_val_bal", 0.0),
                "best_epoch": obj["best_epoch"],
                "total_epochs": args.epochs, "time_seconds": 0.0,
                "ckpt_path": ckpt,
            })
        else:
            res = train_one(
                "Teacher", CONFIGS["Teacher"], seed,
                train_loader, val_loader, test_loader,
                teacher_model=None, epochs=args.epochs, lr=args.lr, wd=args.wd,
                device=device, ckpt_dir=args.ckpt_dir,
            )
            all_results["Teacher"].append(res)

    # GUARDRAIL: abort if teacher isn't learning
    teacher_bals = [r["test_bal_acc"] for r in all_results["Teacher"]]
    if np.mean(teacher_bals) < 0.55:
        print(f"\n[ABORT] Teacher mean bal_acc={np.mean(teacher_bals)*100:.1f}% is too low.")
        print("        BloodMNIST DeiT-Tiny should reach 75-85% bal_acc.")
        print("        Likely causes: training recipe regression, data pipeline bug,")
        print("                       GPU precision issue. Investigate before proceeding.")
        return

    # Load shared teacher
    teacher_ckpt = os.path.join(args.ckpt_dir, f"Teacher_seed{args.teacher_seed}.pt")
    teacher_model, teacher_obj = load_checkpoint(teacher_ckpt, device)
    print(f"\nLoaded shared teacher (seed={args.teacher_seed}): "
          f"test_acc={teacher_obj['test_acc']*100:.2f}% | "
          f"bal_acc={teacher_obj['test_bal_acc']*100:.2f}%")

    # PHASE 2: Students
    for cfg_name in ["Config_E", "Fix2_Normalized"]:
        for seed in args.seeds:
            ckpt = os.path.join(args.ckpt_dir, f"{cfg_name}_seed{seed}.pt")
            if args.skip_train and os.path.exists(ckpt):
                obj = torch.load(ckpt, map_location="cpu", weights_only=False)
                print(f"\n[SKIP] {cfg_name} seed={seed}  "
                      f"(cached test_acc={obj['test_acc']*100:.2f}%, "
                      f"bal={obj['test_bal_acc']*100:.2f}%)")
                all_results[cfg_name].append({
                    "config_name": cfg_name, "seed": seed,
                    "test_acc": obj["test_acc"], "test_bal_acc": obj["test_bal_acc"],
                    "best_val_acc": obj["best_val"], "best_val_bal": obj.get("best_val_bal", 0.0),
                    "best_epoch": obj["best_epoch"],
                    "total_epochs": args.epochs, "time_seconds": 0.0,
                    "ckpt_path": ckpt,
                })
            else:
                res = train_one(
                    cfg_name, CONFIGS[cfg_name], seed,
                    train_loader, val_loader, test_loader,
                    teacher_model=teacher_model, epochs=args.epochs,
                    lr=args.lr, wd=args.wd,
                    device=device, ckpt_dir=args.ckpt_dir,
                )
                all_results[cfg_name].append(res)

    # PHASE 3: Aggregate, print, save
    summary = summarize_results(all_results)
    print_summary(summary)

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(
        args.results_dir,
        f"fix2_bloodmnist_seeds{'_'.join(str(s) for s in args.seeds)}.json",
    )
    with open(out_path, "w") as f:
        json.dump({
            "args": vars(args),
            "results": all_results,
            "summary": summary,
            "dataset": "BloodMNIST",
            "num_classes": 8,
            "img_size": 32,
            "patch_size": 4,
            "depth": 6,
            "embed_dim": 192,
            "num_heads": 3,
            "kd_T": 4.0,
            "kd_alpha": 0.1,
            "grad_clip": 5.0,
        }, f, indent=2, default=float)
    print(f"\nResults JSON: {out_path}")


if __name__ == "__main__":
    main()