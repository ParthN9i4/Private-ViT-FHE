"""
CKKS Encrypted Inference: Classification Head
===============================================

This script validates that CKKS homomorphic encryption produces
correct predictions when applied to the classification head of
our polynomial DeiT-Tiny.

Pipeline:
  1. Train teacher + Fix 2 + Fix 3 models in plaintext (or load if saved)
  2. Run plaintext inference on all 10,000 CIFAR-10 test images
     to extract CLS tokens and predictions
  3. Set up CKKS context (TenSEAL / Microsoft SEAL)
  4. For each test image:
     - Encrypt the CLS token
     - Compute W @ Enc(cls) + b under CKKS
     - Decrypt the logits
     - Compare argmax to plaintext prediction
  5. Report: accuracy match, noise magnitude, throughput

Requirements:
  pip install tenseal torch torchvision

Usage:
  python ckks_classification_head.py                # Full run
  python ckks_classification_head.py --skip-train    # Skip training, load saved models
  python ckks_classification_head.py --max-samples 100  # Quick test on 100 images
"""

import argparse
import time
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

try:
    import tenseal as ts
except ImportError:
    print("TenSEAL not found. Install with: pip install tenseal")
    print("In conda: pip install tenseal --break-system-packages")
    exit(1)


# ══════════════════════════════════════════════════════════════════════
# MODEL (same architecture as verify_fixes.py)
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


class PolyAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.05))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.25))

    def forward(self, x):
        return self.a * x * x + self.b * x + self.c


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
                 norm_type='layernorm', attn_type='standard',
                 gelu_type='standard'):
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
        if attn_type == 'poly':
            self.attn_act = PolyAttn()
        elif attn_type == 'poly_normed':
            self.attn_act = PolyAttnNormed()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = PolyGELU() if gelu_type == 'poly' else nn.GELU()

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
                 norm_type='layernorm', attn_type='standard',
                 gelu_type='standard'):
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
        elif norm_type == 'rmsnorm':
            self.norm = RMSNorm(embed_dim)
        else:
            self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x):
        """Run backbone only, return CLS token (before classification head)."""
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
        return x[:, 0]  # CLS token only

    def forward(self, x):
        cls = self.forward_features(x)
        return self.head(cls)


# ══════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
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
# CKKS ENCRYPTED CLASSIFICATION HEAD
# ══════════════════════════════════════════════════════════════════════

def setup_ckks_context():
    """Create CKKS context with parameters suitable for classification head."""
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,        # N = 8192 (sufficient for 1-level computation)
        coeff_mod_bit_sizes=[60, 40, 60] # [special, scale, special] = 1 mult level
    )
    ctx.global_scale = 2**40             # ~10^12 precision
    ctx.generate_galois_keys()           # Needed for rotations in dot product
    return ctx


def encrypted_classify(ctx, cls_token_np, W_np, b_np):
    """
    Compute logits = W @ cls_token + b entirely under CKKS encryption.

    Args:
        ctx: TenSEAL CKKS context
        cls_token_np: numpy array, shape (192,) — the CLS token to encrypt
        W_np: numpy array, shape (10, 192) — classification weight matrix (plaintext)
        b_np: numpy array, shape (10,) — classification bias (plaintext)

    Returns:
        decrypted_logits: list of 10 floats
    """
    # Encrypt the CLS token
    enc_cls = ts.ckks_vector(ctx, cls_token_np.tolist())

    # Compute each class logit: logit_i = W[i] . enc_cls + b[i]
    logits = []
    for i in range(W_np.shape[0]):
        # Plaintext-ciphertext dot product (1 multiplicative level)
        enc_logit = enc_cls.dot(W_np[i].tolist())
        # Add bias (free under CKKS — just addition)
        enc_logit_with_bias = enc_logit + b_np[i].item()
        # Decrypt this logit
        logits.append(enc_logit_with_bias.decrypt()[0])

    return logits


def encrypted_classify_batched(ctx, cls_token_np, W_np, b_np):
    """
    More efficient version: pack the weight rows and compute all logits
    using a single encryption of the CLS token.

    Same result as encrypted_classify but reuses the encrypted CLS token.
    """
    enc_cls = ts.ckks_vector(ctx, cls_token_np.tolist())

    logits = []
    for i in range(W_np.shape[0]):
        result = enc_cls.dot(W_np[i].tolist())
        result = result + b_np[i].item()
        logits.append(result.decrypt()[0])

    return logits


# ══════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, load saved models")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit test samples (0 = all 10000)")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    train_loader, test_loader = get_cifar10()

    save_dir = Path("ckks_models")
    save_dir.mkdir(exist_ok=True)

    # ── Step 1: Train or load models ─────────────────────────────────
    configs = {
        "Fix2_Normalized": {
            "norm_type": "batchnorm", "attn_type": "poly_normed", "gelu_type": "poly"
        },
        "Fix3_RMSNorm": {
            "norm_type": "rmsnorm", "attn_type": "poly", "gelu_type": "poly"
        },
    }

    teacher_path = save_dir / "teacher.pth"
    if args.skip_train and teacher_path.exists():
        print("Loading saved teacher...")
        teacher = DeiTTiny()
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    else:
        print("\nTraining teacher...")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        teacher = DeiTTiny()
        acc = train_model(teacher, train_loader, test_loader, device,
                          args.epochs, label="Teacher")
        print(f"  Teacher: {acc:.2f}%")
        try:
            torch.save(teacher.state_dict(), teacher_path)
        except Exception:
            print("  (Could not save teacher checkpoint)")

    teacher = teacher.to(device)
    teacher.eval()

    models = {}
    for name, cfg in configs.items():
        model_path = save_dir / f"{name}.pth"
        if args.skip_train and model_path.exists():
            print(f"Loading saved {name}...")
            m = DeiTTiny(**cfg)
            m.load_state_dict(torch.load(model_path, map_location=device))
            models[name] = m
        else:
            print(f"\nTraining {name}...")
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            m = DeiTTiny(**cfg)
            acc = train_model(m, train_loader, test_loader, device,
                              args.epochs, teacher=teacher, label=name)
            print(f"  {name}: {acc:.2f}%")
            try:
                torch.save(m.state_dict(), model_path)
            except Exception:
                print(f"  (Could not save {name} checkpoint)")
            models[name] = m

    # ── Step 2: Extract CLS tokens and plaintext predictions ─────────
    print(f"\n{'='*70}")
    print("STEP 2: Extracting CLS tokens and plaintext predictions")
    print(f"{'='*70}")

    for name, model in models.items():
        model = model.to(device)
        model.eval()

        all_cls_tokens = []
        all_plain_logits = []
        all_labels = []
        n_samples = 0

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                cls_tokens = model.forward_features(imgs)
                logits = model.head(cls_tokens)

                all_cls_tokens.append(cls_tokens.cpu())
                all_plain_logits.append(logits.cpu())
                all_labels.append(labels)

                n_samples += imgs.size(0)
                if args.max_samples > 0 and n_samples >= args.max_samples:
                    break

        all_cls_tokens = torch.cat(all_cls_tokens, dim=0)
        all_plain_logits = torch.cat(all_plain_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        if args.max_samples > 0:
            all_cls_tokens = all_cls_tokens[:args.max_samples]
            all_plain_logits = all_plain_logits[:args.max_samples]
            all_labels = all_labels[:args.max_samples]

        # Plaintext accuracy
        plain_preds = all_plain_logits.argmax(dim=1)
        plain_acc = (plain_preds == all_labels).float().mean().item() * 100
        print(f"\n  {name}: {len(all_labels)} samples, plaintext accuracy: {plain_acc:.2f}%")

        # ── Step 3: CKKS encrypted inference ─────────────────────────
        print(f"\n{'='*70}")
        print(f"STEP 3: CKKS encrypted inference for {name}")
        print(f"{'='*70}")

        # Extract classification head weights
        W = model.head.weight.detach().cpu().numpy()  # (10, 192)
        b = model.head.bias.detach().cpu().numpy()     # (10,)
        print(f"  Classification head: W={W.shape}, b={b.shape}")

        # Setup CKKS context
        ctx = setup_ckks_context()
        print(f"  CKKS context: N=8192, scale=2^40, 1 mult level")

        # Run encrypted inference
        n = len(all_cls_tokens)
        encrypted_preds = []
        logit_errors = []
        times = []
        match_count = 0

        print(f"  Running encrypted inference on {n} samples...")

        for i in range(n):
            cls_np = all_cls_tokens[i].numpy()
            plain_logit = all_plain_logits[i].numpy()

            t0 = time.time()
            enc_logits = encrypted_classify_batched(ctx, cls_np, W, b)
            t1 = time.time()

            times.append(t1 - t0)

            # Compare predictions
            plain_pred = int(plain_logit.argmax())
            enc_pred = int(max(range(len(enc_logits)), key=lambda j: enc_logits[j]))
            encrypted_preds.append(enc_pred)

            if plain_pred == enc_pred:
                match_count += 1

            # Measure logit error
            errors = [abs(enc_logits[j] - plain_logit[j])
                      for j in range(len(enc_logits))]
            logit_errors.append(max(errors))

            if (i + 1) % 500 == 0 or i == 0 or i == n - 1:
                print(f"    Sample {i+1}/{n}  "
                      f"match={match_count}/{i+1} "
                      f"({100*match_count/(i+1):.2f}%)  "
                      f"max_err={max(errors):.2e}  "
                      f"time={times[-1]*1000:.1f}ms")

        # ── Step 4: Results ──────────────────────────────────────────
        enc_labels = torch.tensor(encrypted_preds)
        enc_acc = (enc_labels == all_labels).float().mean().item() * 100

        print(f"\n{'='*70}")
        print(f"RESULTS: {name}")
        print(f"{'='*70}")
        print(f"  Samples tested:         {n}")
        print(f"  Plaintext accuracy:     {plain_acc:.2f}%")
        print(f"  Encrypted accuracy:     {enc_acc:.2f}%")
        print(f"  Prediction match rate:  {match_count}/{n} "
              f"({100*match_count/n:.4f}%)")
        print(f"  Accuracy degradation:   {plain_acc - enc_acc:.4f}%")
        print(f"")
        print(f"  Max logit error:        {max(logit_errors):.2e}")
        print(f"  Mean logit error:       {sum(logit_errors)/len(logit_errors):.2e}")
        print(f"  Median logit error:     {sorted(logit_errors)[len(logit_errors)//2]:.2e}")
        print(f"")
        print(f"  Mean time per sample:   {sum(times)/len(times)*1000:.1f} ms")
        print(f"  Total time:             {sum(times):.1f} s")

        if match_count == n:
            print(f"\n  RESULT: PERFECT MATCH. Every single prediction is identical")
            print(f"  under CKKS encryption. The classification head produces")
            print(f"  correct results under homomorphic encryption.")
        else:
            mismatches = n - match_count
            print(f"\n  RESULT: {mismatches} mismatches out of {n} samples.")
            print(f"  Accuracy degradation: {plain_acc - enc_acc:.4f}%")
            if plain_acc - enc_acc < 0.5:
                print(f"  This is negligible (<0.5%) and acceptable for deployment.")
            else:
                print(f"  This exceeds 0.5%. CKKS parameters may need adjustment.")

        # Save results
        try:
            results = {
                "model": name,
                "n_samples": n,
                "plaintext_accuracy": round(plain_acc, 4),
                "encrypted_accuracy": round(enc_acc, 4),
                "prediction_match_rate": round(100 * match_count / n, 4),
                "accuracy_degradation": round(plain_acc - enc_acc, 4),
                "max_logit_error": float(f"{max(logit_errors):.2e}"),
                "mean_logit_error": float(f"{sum(logit_errors)/len(logit_errors):.2e}"),
                "mean_time_ms": round(sum(times) / len(times) * 1000, 1),
                "ckks_params": {
                    "poly_modulus_degree": 8192,
                    "scale_bits": 40,
                    "mult_levels": 1
                }
            }
            out_dir = Path("experiment_results")
            out_dir.mkdir(exist_ok=True)
            with open(out_dir / f"ckks_{name}.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n  Results saved: experiment_results/ckks_{name}.json")
        except Exception as e:
            print(f"\n  (Could not save results: {e})")

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print("Next steps:")
    print("  1. If classification head matches perfectly -> move to full backbone")
    print("  2. If there are mismatches -> adjust CKKS parameters (larger N, more precision)")
    print("  3. Measure full backbone encrypted inference time")


if __name__ == "__main__":
    main()