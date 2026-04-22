"""
investigate_collapse_bloodmnist.py
===================================
Mechanistic validation on medical data: does the all-polynomial collapse
on BloodMNIST have the same root cause as on CIFAR-10?

WHAT THIS REPLICATES FROM CIFAR-10 (investigate_collapse.py)
------------------------------------------------------------
  Config D (LayerNorm + all-poly, works)   -> control, shrinking norms
  Config E (BatchNorm + all-poly, collapses) -> exploding norms

WHAT'S NEW ON TOP
-----------------
  1. Fix 2 (row-normalized PolyAttn) is tracked as a third config.
     The CIFAR-10 script never directly showed Fix 2's norms stay
     bounded -- it was inferred from accuracy. Direct measurement
     closes the mechanistic argument.
  2. MLP (PolyGELU) output norms are tracked alongside attention norms.
     PolyGELU is also unbounded; if explosion has two sources we want
     to see both.
  3. Gradient norms are tracked per layer at the end of each epoch.
     Closes the 'activations explode -> gradients collapse' claim that
     was asserted but never measured on CIFAR-10.
  4. Fewer epochs (20 instead of 150). On CIFAR-10 the ratio was
     already 99x at epoch 1; we expect the same here.

OUTPUT
------
  experiment_results/investigate_collapse_bloodmnist.json
    Per-epoch, per-config, per-layer:
      - attention_l2_mean         (mean attention output L2 norm)
      - attention_l2_max          (max over batch)
      - mlp_l2_mean               (mean MLP/PolyGELU output L2 norm)
      - grad_norm                 (gradient L2 norm of block)
      - train_loss, val_acc, val_bal

  figures/collapse_mechanism_bloodmnist.png
    Three-panel figure:
      (a) attention norms vs layer, at a fixed epoch
      (b) attention norm vs epoch, at a fixed layer (L5)
      (c) gradient norm vs epoch, at a fixed layer (L5)

USAGE
-----
  # Default: 20 epochs, 1 seed, all three configs
  python investigate_collapse_bloodmnist.py

  # Quick smoke test
  python investigate_collapse_bloodmnist.py --epochs 5

Expected runtime on RTX A6000: ~10-15 minutes total (3 configs x ~4 min).

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
    raise ImportError("pip install medmnist") from e


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
# 1. ACTIVATIONS (same as fix2_bloodmnist.py)
# =====================================================================

class PolyGELU(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.05))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.25))

    def forward(self, scores):
        return self.a * scores * scores + self.b * scores + self.c


class PolyAttnNormed(nn.Module):
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


class TokenBatchNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim, eps=eps, momentum=momentum)

    def forward(self, x):
        B, N, D = x.shape
        x = x.reshape(B * N, D)
        x = self.bn(x)
        return x.reshape(B, N, D)


# =====================================================================
# 2. INSTRUMENTED BLOCK -- stores attn & MLP outputs for inspection
# =====================================================================

class InstrumentedBlock(nn.Module):
    """Same as Block in fix2_bloodmnist.py, but caches intermediate
    tensors so we can probe their norms after a forward pass.

    The caching is done via .detach() so it costs no gradient memory.
    """

    def __init__(self, dim=192, num_heads=3, mlp_ratio=4,
                 norm_type="layernorm", attn_type="standard", gelu_type="standard"):
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
            raise ValueError

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        if attn_type == "standard":
            self.attn_act = None
        elif attn_type == "poly":
            self.attn_act = PolyAttn()
        elif attn_type == "poly_normed":
            self.attn_act = PolyAttnNormed()

        hidden = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU() if gelu_type == "standard" else PolyGELU()

        # Probes -- filled during forward pass
        self._attn_output = None   # post-attention, pre-residual
        self._mlp_output = None    # post-MLP, pre-residual

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
        # Probe: attention branch output (before residual)
        self._attn_output = out.detach()
        x = x + out

        h = self.norm2(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        # Probe: MLP branch output (before residual)
        self._mlp_output = h.detach()
        x = x + h
        return x


# =====================================================================
# 3. MODEL (uses InstrumentedBlock)
# =====================================================================

class DeiTTiny(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=8,
                 embed_dim=192, depth=6, num_heads=3,
                 norm_type="layernorm", attn_type="standard", gelu_type="standard"):
        super().__init__()
        self.depth = depth
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            InstrumentedBlock(dim=embed_dim, num_heads=num_heads,
                              norm_type=norm_type, attn_type=attn_type,
                              gelu_type=gelu_type)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim) if norm_type == "layernorm" \
                    else TokenBatchNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])

    # ------------- Diagnostic helpers -------------
    def probe_activations(self, x):
        """Forward pass; return per-layer L2 norms of attention and MLP outputs.

        For each of the 6 blocks, compute the L2 norm of its attention-branch
        output and MLP-branch output, averaged over (batch, tokens), and
        also the max over (batch, tokens).

        Returns: dict with keys
          attn_mean[L], attn_max[L], mlp_mean[L], mlp_max[L]   for L in 0..depth-1
        """
        self.eval()
        with torch.no_grad():
            _ = self.forward(x)
        probe = {"attn_mean": [], "attn_max": [], "mlp_mean": [], "mlp_max": []}
        for blk in self.blocks:
            # _attn_output: [B, N, D]. L2 norm per (b, n) position.
            a = blk._attn_output.norm(dim=-1)   # [B, N]
            m = blk._mlp_output.norm(dim=-1)    # [B, N]
            probe["attn_mean"].append(float(a.mean().item()))
            probe["attn_max"].append(float(a.max().item()))
            probe["mlp_mean"].append(float(m.mean().item()))
            probe["mlp_max"].append(float(m.max().item()))
        return probe

    def probe_grad_norms(self):
        """Per-block gradient L2 norm -- run AFTER loss.backward()."""
        norms = []
        for blk in self.blocks:
            total = 0.0
            for p in blk.parameters():
                if p.grad is not None:
                    total += float(p.grad.detach().norm().item()) ** 2
            norms.append(total ** 0.5)
        return norms


# =====================================================================
# 4. DATA (same as fix2_bloodmnist.py)
# =====================================================================

def get_bloodmnist_loaders(batch_size=128, num_workers=2, data_root="./data"):
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

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, drop_last=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
    )


# =====================================================================
# 5. TEACHER LOADING (for KD)
# =====================================================================

def load_or_train_teacher(train_loader, val_loader, device,
                         teacher_ckpt="./checkpoints_bloodmnist/Teacher_seed42.pt",
                         epochs=30):
    """Use the teacher trained by fix2_bloodmnist.py if available.
    Otherwise train a quick one (20 epochs is enough for diagnostic KD)."""
    teacher = DeiTTiny(num_classes=8).to(device)

    if os.path.exists(teacher_ckpt):
        print(f"Loading cached teacher: {teacher_ckpt}")
        obj = torch.load(teacher_ckpt, map_location=device, weights_only=False)
        teacher.load_state_dict(obj["state_dict"])
        return teacher

    print("No cached teacher found; training a diagnostic teacher (30 epochs)...")
    set_seed(42)
    teacher = DeiTTiny(num_classes=8).to(device)
    opt = torch.optim.AdamW(teacher.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        teacher.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.squeeze(-1).long().to(device)
            opt.zero_grad()
            loss = F.cross_entropy(teacher(imgs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=5.0)
            opt.step()
        sched.step()
    return teacher


# =====================================================================
# 6. TRAINING WITH INSTRUMENTATION
# =====================================================================

CONFIGS = {
    # D: LayerNorm + all-poly -- works. Control for mechanism isolation.
    "Config_D_LN":       {"norm": "layernorm", "attn": "poly",        "gelu": "poly",     "kd": True},
    # E: BatchNorm + all-poly -- collapses. Primary subject.
    "Config_E_BN":       {"norm": "batchnorm", "attn": "poly",        "gelu": "poly",     "kd": True},
    # Fix 2: BatchNorm + row-normalized PolyAttn -- recovers. New in this diagnostic.
    "Fix2_Normalized":   {"norm": "batchnorm", "attn": "poly_normed", "gelu": "poly",     "kd": True},
}


def train_and_probe(config_name, config, train_loader, val_loader,
                    teacher, device, epochs=20, seed=42, probe_batch=None):
    """Train and log per-epoch activation + gradient + accuracy diagnostics."""
    set_seed(seed)
    model = DeiTTiny(
        num_classes=8,
        norm_type=config["norm"], attn_type=config["attn"], gelu_type=config["gelu"],
    ).to(device)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    log = {
        "config_name": config_name,
        "epoch": [], "train_loss": [], "val_acc": [], "val_bal": [],
        "attn_mean": [], "attn_max": [], "mlp_mean": [], "mlp_max": [],
        "grad_norm": [],
    }

    print(f"\n=== {config_name} (norm={config['norm']} "
          f"attn={config['attn']} gelu={config['gelu']}) ===")

    # Epoch-0 probe: activations at initialization, before any training
    probe = model.probe_activations(probe_batch)
    grad_norms = [0.0] * model.depth   # no gradients at init
    _append_log(log, 0, float("nan"), float("nan"), float("nan"),
                probe, grad_norms)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        last_grad_norms = None
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.squeeze(-1).long().to(device)
            opt.zero_grad()
            s_logits = model(imgs)
            with torch.no_grad():
                t_logits = teacher(imgs)
            ce = F.cross_entropy(s_logits, labels)
            kd = F.kl_div(F.log_softmax(s_logits / 4.0, dim=-1),
                          F.softmax(t_logits / 4.0, dim=-1), reduction="batchmean")
            loss = 0.9 * ce + 0.1 * kd
            loss.backward()
            # Capture grad norms BEFORE optimizer step (otherwise grads get zeroed)
            last_grad_norms = model.probe_grad_norms()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        sched.step()

        # Activations: probed on eval() with fixed held-out batch
        probe = model.probe_activations(probe_batch)
        val_acc, val_bal = _evaluate(model, val_loader, device)
        avg_loss = total_loss / max(n_batches, 1)
        _append_log(log, ep, avg_loss, val_acc, val_bal, probe,
                    last_grad_norms or [0.0] * model.depth)

        # Print compact per-epoch summary (first 10 epochs + every 5 after)
        if ep <= 10 or ep % 5 == 0:
            print(f"  ep {ep:2d} | loss={avg_loss:.3f} | "
                  f"val_bal={val_bal*100:5.2f}% | "
                  f"L0 attn={probe['attn_mean'][0]:8.1f}  "
                  f"L5 attn={probe['attn_mean'][-1]:8.1f}  "
                  f"L5 grad={last_grad_norms[-1]:.3f}")

    return log


def _append_log(log, ep, loss, acc, bal, probe, grad_norms):
    log["epoch"].append(ep)
    log["train_loss"].append(float(loss))
    log["val_acc"].append(float(acc))
    log["val_bal"].append(float(bal))
    log["attn_mean"].append(probe["attn_mean"])
    log["attn_max"].append(probe["attn_max"])
    log["mlp_mean"].append(probe["mlp_mean"])
    log["mlp_max"].append(probe["mlp_max"])
    log["grad_norm"].append([float(g) for g in grad_norms])


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    all_p, all_y = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.squeeze(-1).long().to(device)
        preds = model(imgs).argmax(dim=-1)
        all_p.append(preds.cpu())
        all_y.append(labels.cpu())
    p = torch.cat(all_p).numpy()
    y = torch.cat(all_y).numpy()
    acc = float((p == y).mean())
    classes = np.unique(y)
    recalls = [(p[y == c] == c).mean() for c in classes if (y == c).sum() > 0]
    bal = float(np.mean(recalls)) if recalls else 0.0
    return acc, bal


# =====================================================================
# 7. FIGURE
# =====================================================================

def plot_mechanism(logs, out_path):
    """Three-panel figure: (a) norms vs layer @ fixed epoch,
    (b) norms vs epoch @ fixed layer (L5), (c) grad vs epoch @ L5."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipping figure.")
        return

    colors = {"Config_D_LN": "tab:green",
              "Config_E_BN": "tab:red",
              "Fix2_Normalized": "tab:blue"}
    labels = {"Config_D_LN": "Config D (LayerNorm, works)",
              "Config_E_BN": "Config E (BatchNorm, collapses)",
              "Fix2_Normalized": "Fix 2 (row-normalized)"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Pick a probe epoch: epoch 5 if available, else halfway
    max_epoch = max(max(log["epoch"]) for log in logs.values())
    probe_ep = min(5, max_epoch)
    probe_idx = probe_ep   # epochs are 0-indexed in log (we start at 0)

    # Panel (a): norms vs layer, at probe_ep
    ax = axes[0]
    for name, log in logs.items():
        if probe_idx < len(log["attn_mean"]):
            y = log["attn_mean"][probe_idx]
            ax.plot(range(len(y)), y, "o-", color=colors[name], label=labels[name])
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Mean attn output L2 norm (epoch {probe_ep})")
    ax.set_yscale("log")
    ax.set_title("(a) Norms grow/shrink through depth")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel (b): L5 attn norm vs epoch
    ax = axes[1]
    for name, log in logs.items():
        y = [layer_norms[-1] for layer_norms in log["attn_mean"]]   # L5 == last block
        ax.plot(log["epoch"], y, "o-", color=colors[name], label=labels[name])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean attn output L2 norm at layer 5")
    ax.set_yscale("log")
    ax.set_title("(b) Layer 5 over training")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel (c): L5 grad norm vs epoch
    ax = axes[2]
    for name, log in logs.items():
        y = [gn[-1] for gn in log["grad_norm"]]
        ax.plot(log["epoch"], y, "o-", color=colors[name], label=labels[name])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient L2 norm at layer 5")
    ax.set_yscale("log")
    ax.set_title("(c) Gradients follow activations")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Collapse mechanism: BloodMNIST", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out_path}")


# =====================================================================
# 8. MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--probe-batch-size", type=int, default=256,
                        help="Batch size for the fixed diagnostic probe batch.")
    parser.add_argument("--teacher-ckpt", type=str,
                        default="./checkpoints_bloodmnist/Teacher_seed42.pt")
    parser.add_argument("--results-dir", type=str, default="./experiment_results")
    parser.add_argument("--figures-dir", type=str, default="./figures")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}   Epochs: {args.epochs}   Seed: {args.seed}")

    train_loader, val_loader = get_bloodmnist_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        data_root=args.data_root,
    )
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}")

    teacher = load_or_train_teacher(train_loader, val_loader, device,
                                    teacher_ckpt=args.teacher_ckpt)
    t_acc, t_bal = _evaluate(teacher, val_loader, device)
    print(f"Teacher: val_acc={t_acc*100:.2f}%  val_bal={t_bal*100:.2f}%")

    # Pick a FIXED probe batch (same images every epoch, deterministic)
    # so probe_activations measures the same distribution across configs & epochs.
    set_seed(args.seed)
    probe_imgs, _ = next(iter(
        DataLoader(val_loader.dataset, batch_size=args.probe_batch_size,
                   shuffle=False, num_workers=0)
    ))
    probe_imgs = probe_imgs.to(device)

    logs = {}
    t0 = time.time()
    for cfg_name, cfg in CONFIGS.items():
        logs[cfg_name] = train_and_probe(
            cfg_name, cfg, train_loader, val_loader, teacher, device,
            epochs=args.epochs, seed=args.seed, probe_batch=probe_imgs,
        )
    print(f"\nTotal wall time: {time.time() - t0:.0f}s")

    # Save JSON
    os.makedirs(args.results_dir, exist_ok=True)
    out_json = os.path.join(args.results_dir, "investigate_collapse_bloodmnist.json")
    with open(out_json, "w") as f:
        json.dump({
            "args": vars(args),
            "teacher_val_acc": t_acc,
            "teacher_val_bal": t_bal,
            "logs": logs,
        }, f, indent=2, default=float)
    print(f"JSON saved: {out_json}")

    # Save figure
    os.makedirs(args.figures_dir, exist_ok=True)
    plot_mechanism(logs, os.path.join(args.figures_dir,
                                      "collapse_mechanism_bloodmnist.png"))

    # Print the headline comparison (what CIFAR-10 reported)
    print("\n" + "=" * 72)
    print("HEADLINE NUMBERS (match CIFAR-10 investigate_collapse.py output)")
    print("=" * 72)
    for ep in [1, 5, 10, min(20, args.epochs)]:
        if ep > args.epochs:
            continue
        print(f"\nEpoch {ep}, mean attention L2 norm by layer:")
        print("  Layer  |  Config D (LN)  |  Config E (BN)  |  Fix 2  |  E/D ratio")
        print("  -------+-----------------+-----------------+---------+-----------")
        for L in range(6):
            d = logs["Config_D_LN"]["attn_mean"][ep][L]
            e = logs["Config_E_BN"]["attn_mean"][ep][L]
            f = logs["Fix2_Normalized"]["attn_mean"][ep][L]
            ratio = e / d if d > 0 else float("inf")
            print(f"  {L}      |  {d:13.2f}  |  {e:13.2f}  |  {f:6.2f} |  {ratio:8.2f}x")


if __name__ == "__main__":
    main()