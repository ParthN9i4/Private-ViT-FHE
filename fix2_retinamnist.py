"""
fix2_retinamnist.py
===================
Medical-dataset validation of the polynomial-ViT interaction-collapse finding.

SCIENTIFIC QUESTION
-------------------
Does the interaction-collapse failure mode (Config E) and the Fix 2 recovery
that we established on CIFAR-10 (verify_fixes.py) reproduce on a medical
dataset (RetinaMNIST 5-class DR grading)?

If YES  -> the paper's core scientific claim extends to the motivating domain.
If NO   -> the CIFAR-10 collapse may be natural-image-specific; investigate.

CONFIGURATIONS
--------------
  Teacher          : LayerNorm + softmax + GELU    (standard ViT, no KD)
  Config_E         : BatchNorm + PolyAttn + PolyGELU + KD   (expected: collapse)
  Fix2_Normalized  : BatchNorm + PolyAttnNormed + PolyGELU + KD (expected: recover)

ARCHITECTURE
------------
6-layer DeiT-Tiny, embed_dim=192, 3 heads, patch_size=4. Identical to
verify_fixes.py. RetinaMNIST images (28x28 native) are resized to 32x32
so the architecture is bit-for-bit the same as the CIFAR-10 run; only
the dataset changes. This is the cleanest experimental design for
isolating data-dependence.

SEEDS: 42, 43, 44, 45, 46.
TEACHER SHARING: Seed 42 teacher is reused as the KD target for all student
seeds. This isolates student-side variance from teacher-side variance
(same convention as verify_fixes.py).

USAGE
-----
  # Full run (default: 150 epochs, 5 seeds)
  python fix2_retinamnist.py

  # Reuse checkpoints if present
  python fix2_retinamnist.py --skip-train

  # Quick smoke test (1 seed, fewer epochs)
  python fix2_retinamnist.py --seeds 42 --epochs 30

Expected runtime on RTX A6000: ~8-15 minutes per run * 15 runs ~= 2-4 hours.
RetinaMNIST train set is tiny (1,080 samples), so each epoch is fast.

DEPENDENCIES
------------
  pip install medmnist scipy

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
    """Seed every RNG we touch. cudnn.deterministic trades speed for reproducibility."""
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
    """Degree-2 polynomial replacement for GELU.

        f(x) = a*x^2 + b*x + c        (a, b, c trainable, per layer)

    Initialization: least-squares fit to GELU over [-3, 3] with 10,000
    samples, giving approximately (a, b, c) = (0.174, 0.500, 0.145).

    Observed after training (polygelu_diagnostic.py): b collapses from
    0.500 to ~0.01 across all layers. The trained polynomial is a near-
    pure quadratic and is NOT a GELU approximation anymore.

    CKKS cost: 1 multiplicative level.
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
    """Degree-2 polynomial applied element-wise to attention scores,
    WITHOUT normalization. This is Config E's attention — the broken one.

        f(s) = a*s^2 + b*s + c

    Initialization: a=0.05, b=0.5, c=0.25 (hand-chosen; softmax is not
    element-wise, so LSQ fitting doesn't apply).

    Without a normalization step, attention-row outputs compound across
    layers. Observed: ~90 at layer 5 under Config D (LayerNorm) versus
    ~29,311 at layer 5 under Config E (BatchNorm). The 326x gap is the
    root cause of the all-polynomial collapse.
    """

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.05))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.25))

    def forward(self, scores):
        # scores: [B, H, N, N]
        return self.a * scores * scores + self.b * scores + self.c


class PolyAttnNormed(nn.Module):
    """Fix 2: degree-2 polynomial attention WITH row normalization.

        poly(s) = a*s^2 + b*s + c         (same init as PolyAttn)
        clipped = ReLU(poly(s))            (ensures row-sum > 0)
        out     = clipped / (rowsum + eps) (produces a probability distribution)

    This is NOT an approximation of softmax. The polynomial is an
    arbitrary learnable element-wise nonlinearity; row normalization is
    what makes it behave like softmax (outputs form a distribution). The
    model learns via KD despite the internal attention pattern not
    matching the teacher's.

    CKKS cost: 1 level (polynomial) + ~20 levels (row division).
    The division is the main depth-budget cost of Fix 2 and is the
    target for future optimization (polynomial approximation of 1/x,
    or switch to Power-Softmax x^p / sum x^p).
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

    ViT tokens have shape [B, N, D]. nn.BatchNorm1d expects [B, C] or
    [B, C, L], so we reshape to [B*N, D], apply BN, reshape back.

    At CKKS inference: running_mean / running_var are precomputed, so
    this becomes y = scale * x + shift  (one plaintext-ciphertext
    multiply + one add), consuming 0 multiplicative levels.

    Tradeoff: BatchNorm normalizes per-feature across (batch, token)
    rather than per-token across features (LayerNorm). This is cheap
    under CKKS but misses per-token outliers — the root cause of the
    all-polynomial collapse in Config E.
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

        # --- Normalization ---
        if norm_type == "layernorm":
            self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        elif norm_type == "batchnorm":
            self.norm1, self.norm2 = TokenBatchNorm(dim), TokenBatchNorm(dim)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # --- Attention linears ---
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # --- Attention activation (softmax replacement) ---
        if attn_type == "standard":
            self.attn_act = None  # F.softmax inline
        elif attn_type == "poly":
            self.attn_act = PolyAttn()
        elif attn_type == "poly_normed":
            self.attn_act = PolyAttnNormed()
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

        # --- MLP (4x expansion, standard ViT) ---
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

        # ---- Attention sub-block ----
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, d]
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        if self.attn_type == "standard":
            attn = F.softmax(scores, dim=-1)
        else:
            attn = self.attn_act(scores)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        x = x + out

        # ---- MLP sub-block ----
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
    """6-layer DeiT-Tiny for 32x32 images. Architecture is identical to
    verify_fixes.py so that only the dataset differs between runs."""

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_channels: int = 3, num_classes: int = 5,
                 embed_dim: int = 192, depth: int = 6, num_heads: int = 3,
                 norm_type: str = "layernorm", attn_type: str = "standard",
                 gelu_type: str = "standard"):
        super().__init__()
        self.num_classes = num_classes
        num_patches = (img_size // patch_size) ** 2  # 32/4 -> 8*8 = 64

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
        x = self.patch_embed(x)             # [B, D, h, w]
        x = x.flatten(2).transpose(1, 2)    # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)      # [B, N+1, D]
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])           # classify the CLS token


# =====================================================================
# 5. DATA: RetinaMNIST
# =====================================================================

def get_retinamnist_loaders(batch_size: int = 64, num_workers: int = 2,
                            data_root: str = "./data"):
    """RetinaMNIST resized 28 -> 32 so architecture matches CIFAR-10 setup.

    Splits (standard MedMNIST):
        train: 1,080    val: 120    test: 400
    Task: 5-class ordinal DR grading.
    Class distribution is imbalanced (class 0 "no DR" dominates), so
    we report BOTH accuracy and balanced accuracy.
    """
    info = INFO["retinamnist"]
    DataClass = getattr(medmnist, info["python_class"])

    # Horizontal flip is safe for fundus; vertical is NOT (macula position).
    # Color jitter kept mild to preserve disease-relevant features.
    train_tf = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
    """KL divergence between student and teacher soft distributions.

    NOTE: We deliberately do NOT multiply by T^2 (Hinton et al. 2015).
    With T=4, T^2=16 amplifies the KD gradient 16x and drowns out the
    CE signal, which caused collapse on CIFAR-100 for polynomial ViTs.
    See PROJECT_CONTEXT.md (KD setup).
    """
    s_log = F.log_softmax(s_logits / T, dim=-1)
    t_prob = F.softmax(t_logits / T, dim=-1)
    return F.kl_div(s_log, t_prob, reduction="batchmean")


def compute_loss(s_logits, t_logits, labels, T: float = 4.0, alpha: float = 0.1):
    """alpha * KD + (1 - alpha) * CE.  alpha=0.1 -> CE dominates (90%)."""
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
    """Return (accuracy, balanced_accuracy) on loader."""
    model.eval()
    all_p, all_y = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.squeeze(-1).long().to(device)  # MedMNIST labels are [B, 1]
        logits = model(imgs)
        preds = logits.argmax(dim=-1)
        all_p.append(preds.cpu())
        all_y.append(labels.cpu())
    preds = torch.cat(all_p).numpy()
    labels = torch.cat(all_y).numpy()
    acc = float((preds == labels).mean())
    # Balanced accuracy = mean per-class recall (honest for imbalanced data)
    classes = np.unique(labels)
    recalls = [(preds[labels == c] == c).mean() for c in classes if (labels == c).sum() > 0]
    bal_acc = float(np.mean(recalls)) if recalls else 0.0
    return acc, bal_acc


def train_one(config_name, config, seed, train_loader, val_loader, test_loader,
              teacher_model=None, epochs: int = 150, lr: float = 1e-3,
              wd: float = 0.05, device: str = "cuda",
              ckpt_dir: str = "./checkpoints_retinamnist", print_every: int = 25):
    """Train one (config, seed) to convergence and evaluate on test."""
    set_seed(seed)

    model = DeiTTiny(
        img_size=32, patch_size=4, in_channels=3, num_classes=5,
        embed_dim=192, depth=6, num_heads=3,
        norm_type=config["norm"], attn_type=config["attn"], gelu_type=config["gelu"],
    ).to(device)

    if teacher_model is not None and config["kd"]:
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val, best_epoch, best_state = 0.0, -1, None
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
        val_acc, _ = evaluate(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % print_every == 0 or epoch == 0:
            elapsed = time.time() - t_start
            avg_loss = total_loss / max(n_batches, 1)
            print(f"  epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f} | "
                  f"val_acc={val_acc*100:.2f}% | "
                  f"best_val={best_val*100:.2f}%@{best_epoch+1} | {elapsed:.0f}s")

    # Restore best weights and evaluate on test
    model.load_state_dict(best_state)
    test_acc, test_bal_acc = evaluate(model, test_loader, device)
    elapsed = time.time() - t_start
    print(f"  FINAL: test_acc={test_acc*100:.2f}% | "
          f"bal_acc={test_bal_acc*100:.2f}% | {elapsed:.0f}s")

    # Save checkpoint
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{config_name}_seed{seed}.pt")
    torch.save({
        "state_dict": best_state,
        "config": config,
        "seed": seed,
        "test_acc": test_acc,
        "test_bal_acc": test_bal_acc,
        "best_val": best_val,
        "best_epoch": best_epoch,
    }, ckpt_path)

    return {
        "config_name": config_name, "seed": seed,
        "test_acc": test_acc, "test_bal_acc": test_bal_acc,
        "best_val_acc": best_val, "best_epoch": best_epoch,
        "total_epochs": epochs, "time_seconds": elapsed,
        "ckpt_path": ckpt_path,
    }


def load_checkpoint(ckpt_path: str, device: str):
    """Re-instantiate a model from a checkpoint (used to load the shared teacher)."""
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = obj["config"]
    model = DeiTTiny(
        img_size=32, patch_size=4, in_channels=3, num_classes=5,
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
    """Per-config statistics + Welch's t-tests for the three key comparisons."""
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
            "n_seeds": int(len(accs)),
        }
    if HAVE_SCIPY:
        pairs = [
            ("Fix2_Normalized", "Config_E"),  # primary: does the fix recover?
            ("Fix2_Normalized", "Teacher"),   # does Fix 2 match / beat the teacher?
            ("Config_E",        "Teacher"),   # does Config E collapse vs. teacher?
        ]
        for a, b in pairs:
            if a in all_results and b in all_results:
                acc_a = np.array([r["test_acc"] for r in all_results[a]])
                acc_b = np.array([r["test_acc"] for r in all_results[b]])
                if len(acc_a) > 1 and len(acc_b) > 1:
                    t, p = scipy_stats.ttest_ind(acc_a, acc_b, equal_var=False)
                    summary["tests"][f"{a}_vs_{b}"] = {
                        "t_statistic": float(t),
                        "p_value": float(p),
                        "mean_diff_pp": float((acc_a.mean() - acc_b.mean()) * 100),
                    }
    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 72)
    print("FINAL RESULTS SUMMARY: RetinaMNIST (5-class DR grading)")
    print("=" * 72)
    for name, s in summary["per_config"].items():
        print(f"\n{name}  (n={s['n_seeds']} seeds):")
        print(f"  test_acc : {s['test_acc_mean']*100:6.2f}% +/- {s['test_acc_std']*100:.2f}%  "
              f"[{s['test_acc_min']*100:.2f}%, {s['test_acc_max']*100:.2f}%]")
        print(f"  bal_acc  : {s['test_bal_acc_mean']*100:6.2f}% +/- {s['test_bal_acc_std']*100:.2f}%")
        print(f"  seeds    : {[f'{a*100:.2f}' for a in s['test_acc_seeds']]}")

    if summary["tests"]:
        print("\nWelch's t-tests (two-sided):")
        for k, t in summary["tests"].items():
            sig = "  [p<.05]" if t["p_value"] < 0.05 else ""
            print(f"  {k:<36s}  t={t['t_statistic']:+.3f}  "
                  f"p={t['p_value']:.4f}  dAcc={t['mean_diff_pp']:+.2f}pp{sig}")

    # Lightweight interpretation (the "what it means" panel)
    e_acc = summary["per_config"]["Config_E"]["test_acc_mean"]
    t_acc = summary["per_config"]["Teacher"]["test_acc_mean"]
    f_acc = summary["per_config"]["Fix2_Normalized"]["test_acc_mean"]
    collapsed = e_acc < 0.7 * t_acc           # Config E markedly below teacher
    recovered = f_acc >= 0.95 * t_acc         # Fix 2 within 5% of teacher
    print("\nInterpretation:")
    if collapsed and recovered:
        print("  -> Config E collapsed AND Fix 2 recovered.")
        print("     The CIFAR-10 finding REPRODUCES on medical data.")
        print("     Paper's core claim extends to the motivating domain.")
    elif collapsed and not recovered:
        print("  -> Config E collapsed; Fix 2 did NOT fully recover.")
        print("     Partial reproduction. Check: class-imbalance effect,")
        print("     training-set size (1,080 samples may be too small),")
        print("     teacher quality on imbalanced data.")
    elif not collapsed and recovered:
        print("  -> Config E did NOT collapse on RetinaMNIST.")
        print("     Hypothesis: CIFAR-10 collapse is sensitive to natural-image")
        print("     statistics (smooth backgrounds, low per-token variance).")
        print("     Next: run investigate_collapse.py activation tracking on")
        print("     RetinaMNIST to compare per-layer magnitudes directly.")
    else:
        print("  -> Ambiguous outcome. Re-check training hyperparameters and")
        print("     inspect per-seed curves before drawing conclusions.")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-train", action="store_true",
                        help="Reuse cached checkpoints if present.")
    parser.add_argument("--teacher-seed", type=int, default=42,
                        help="Seed of the teacher used for student KD.")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints_retinamnist")
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

    train_loader, val_loader, test_loader = get_retinamnist_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        data_root=args.data_root,
    )
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  "
          f"Test: {len(test_loader.dataset)}")

    all_results = {name: [] for name in CONFIGS}

    # -------- PHASE 1: Teacher at all seeds --------
    for seed in args.seeds:
        ckpt = os.path.join(args.ckpt_dir, f"Teacher_seed{seed}.pt")
        if args.skip_train and os.path.exists(ckpt):
            obj = torch.load(ckpt, map_location="cpu", weights_only=False)
            print(f"\n[SKIP] Teacher seed={seed}  (cached test_acc={obj['test_acc']*100:.2f}%)")
            all_results["Teacher"].append({
                "config_name": "Teacher", "seed": seed,
                "test_acc": obj["test_acc"], "test_bal_acc": obj["test_bal_acc"],
                "best_val_acc": obj["best_val"], "best_epoch": obj["best_epoch"],
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

    # -------- Load shared teacher (seed=42 by default) --------
    teacher_ckpt = os.path.join(args.ckpt_dir, f"Teacher_seed{args.teacher_seed}.pt")
    teacher_model, teacher_obj = load_checkpoint(teacher_ckpt, device)
    print(f"\nLoaded shared teacher (seed={args.teacher_seed}): "
          f"test_acc={teacher_obj['test_acc']*100:.2f}%")

    # -------- PHASE 2: Students (Config_E, Fix2_Normalized) at all seeds --------
    for cfg_name in ["Config_E", "Fix2_Normalized"]:
        for seed in args.seeds:
            ckpt = os.path.join(args.ckpt_dir, f"{cfg_name}_seed{seed}.pt")
            if args.skip_train and os.path.exists(ckpt):
                obj = torch.load(ckpt, map_location="cpu", weights_only=False)
                print(f"\n[SKIP] {cfg_name} seed={seed}  "
                      f"(cached test_acc={obj['test_acc']*100:.2f}%)")
                all_results[cfg_name].append({
                    "config_name": cfg_name, "seed": seed,
                    "test_acc": obj["test_acc"], "test_bal_acc": obj["test_bal_acc"],
                    "best_val_acc": obj["best_val"], "best_epoch": obj["best_epoch"],
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

    # -------- PHASE 3: Aggregate, print, save --------
    summary = summarize_results(all_results)
    print_summary(summary)

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(
        args.results_dir,
        f"fix2_retinamnist_seeds{'_'.join(str(s) for s in args.seeds)}.json",
    )
    with open(out_path, "w") as f:
        json.dump({
            "args": vars(args),
            "results": all_results,
            "summary": summary,
            "dataset": "RetinaMNIST",
            "num_classes": 5,
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