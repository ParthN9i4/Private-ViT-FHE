"""
Step 3: Polynomial GELU + Polynomial Softmax (LayerNorm kept)
==============================================================
Replaces 24 of 49 non-polynomial operations in DeiT-Tiny:
  - 12 GELU activations → degree-2 polynomial (proven in Step 5B)
  - 12 softmax in attention → poly_softmax (repeated squaring)
  - 25 LayerNorm → KEPT AS-IS (handled at FHE inference time)

Why keep LayerNorm:
  Removing LayerNorm causes NaN explosion. Without normalization,
  residual connections cause activations to grow unboundedly across
  12 blocks. The polynomial activation (x²) amplifies large values,
  and poly_exp overflows.

  PolyTransformer (Zimerman et al., ICML 2024) keeps LayerNorm too.
  At FHE inference time, LayerNorm can be handled by:
  (a) Client-aided: client decrypts after each block, normalizes,
      re-encrypts (adds communication rounds but zero FHE depth)
  (b) Polynomial approximation of reciprocal sqrt with careful
      range management (adds ~4-6 multiplicative levels per norm)
  (c) Absorb into adjacent linear layers for frozen blocks

  This is a known open problem in the FHE-ML literature and is
  orthogonal to the GELU+softmax replacement contribution.

Training: Cold-start KD from Step 1 teacher (proven in Step 5B).

Usage:
    python step3_poly_gelu_softmax_kd.py --dataset all
    python step3_poly_gelu_softmax_kd.py --dataset blood --softmax_depth 4
"""

import os
import argparse
import json
import time
import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import timm
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize

import medmnist
from medmnist import (
    RetinaMNIST, PneumoniaMNIST, BloodMNIST,
    DermaMNIST, BreastMNIST, PathMNIST
)


# ═══════════════════════════════════════════════════════════════
# 1. GELU Replacement: Trainable Polynomial
# ═══════════════════════════════════════════════════════════════

class PolyActivation(nn.Module):
    """f(x) = a·x² + b·x + c via Horner's method. 1 CKKS mult level."""
    def __init__(self, degree=2, init_method="gelu_fit"):
        super().__init__()
        self.degree = degree
        if init_method == "gelu_fit" and degree == 2:
            init_coeffs = [0.0711, 0.5, 0.2576]
        elif init_method == "gelu_fit" and degree == 4:
            init_coeffs = [0.0711, 0.5, 0.2576, 0.0, -0.0128]
        else:
            init_coeffs = [0.0] * (degree + 1)
            init_coeffs[1] = 1.0
        self.coeffs = nn.Parameter(torch.tensor(init_coeffs, dtype=torch.float32))

    def forward(self, x):
        result = self.coeffs[-1]
        for i in range(len(self.coeffs) - 2, -1, -1):
            result = result * x + self.coeffs[i]
        return result


# ═══════════════════════════════════════════════════════════════
# 2. Softmax Replacement: Polynomial Exp + Normalization
# ═══════════════════════════════════════════════════════════════

def poly_exp(x, depth=3):
    """
    exp(x) ≈ (1 + x/2^d)^{2^d} via d repeated squarings.
    CKKS cost: d multiplicative levels.
    """
    scale = 2 ** depth
    result = 1.0 + x / scale
    result = result.clamp(min=0.0)
    for _ in range(depth):
        result = result * result
    return result


def poly_softmax(x, depth=3, dim=-1):
    """Polynomial softmax: poly_exp numerator, exact division."""
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    e = poly_exp(x_shifted, depth=depth)
    e_sum = e.sum(dim=dim, keepdim=True).clamp(min=1e-10)
    return e / e_sum


# ═══════════════════════════════════════════════════════════════
# KD Loss
# ═══════════════════════════════════════════════════════════════

def kd_loss(student_logits, teacher_logits, labels, temperature, alpha,
            label_smoothing=0.1):
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    log_soft_student = F.log_softmax(student_logits / temperature, dim=1)
    distill_loss = F.kl_div(
        log_soft_student, soft_teacher, reduction='batchmean'
    ) * (temperature ** 2)
    task_loss = F.cross_entropy(
        student_logits, labels, label_smoothing=label_smoothing
    )
    total = alpha * distill_loss + (1.0 - alpha) * task_loss
    return total, distill_loss.item(), task_loss.item()


# ═══════════════════════════════════════════════════════════════
# Model Modification
# ═══════════════════════════════════════════════════════════════

def replace_gelu(model, degree=2, init_method="gelu_fit"):
    """Replace nn.GELU in each MLP block with PolyActivation."""
    count = 0
    for block in model.blocks:
        if hasattr(block, 'mlp') and hasattr(block.mlp, 'act'):
            if isinstance(block.mlp.act, nn.GELU):
                block.mlp.act = PolyActivation(degree=degree, init_method=init_method)
                count += 1
    return count


def replace_attention_softmax(model, depth=3):
    """Replace F.softmax in attention with poly_softmax."""
    count = 0
    for block in model.blocks:
        attn_module = block.attn

        if hasattr(attn_module, 'fused_attn'):
            attn_module.fused_attn = False

        if hasattr(attn_module, 'head_dim'):
            head_dim = attn_module.head_dim
        else:
            head_dim = attn_module.qkv.in_features // attn_module.num_heads

        attn_module._poly_softmax_depth = depth
        attn_module._head_dim = head_dim

        def make_poly_forward(d):
            def poly_attn_forward(self, x, attn_mask=None):
                B, N, C = x.shape
                qkv = self.qkv(x).reshape(
                    B, N, 3, self.num_heads, self._head_dim
                ).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)

                q = q * self.scale
                attn = q @ k.transpose(-2, -1)

                if attn_mask is not None:
                    attn = attn + attn_mask

                attn = poly_softmax(attn, depth=self._poly_softmax_depth, dim=-1)
                attn = self.attn_drop(attn)

                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = self.proj(x)
                x = self.proj_drop(x)
                return x
            return poly_attn_forward

        attn_module.forward = types.MethodType(make_poly_forward(depth), attn_module)
        count += 1

    return count


def get_poly_coefficients(model):
    coeffs = {}
    for name, module in model.named_modules():
        if isinstance(module, PolyActivation):
            coeffs[name] = module.coeffs.data.cpu().tolist()
    return coeffs


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

DATASET_CONFIG = {
    "retina":      {"class": RetinaMNIST,      "n_classes": 5, "task": "multi-class"},
    "pneumonia":   {"class": PneumoniaMNIST,   "n_classes": 2, "task": "binary"},
    "blood":       {"class": BloodMNIST,       "n_classes": 8, "task": "multi-class"},
    "derma":       {"class": DermaMNIST,       "n_classes": 7, "task": "multi-class"},
    "breast":      {"class": BreastMNIST,      "n_classes": 2, "task": "binary"},
    "path":        {"class": PathMNIST,        "n_classes": 9, "task": "multi-class"},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 3: Poly GELU + Poly Softmax (LayerNorm kept) with cold-start KD"
    )
    parser.add_argument("--dataset", type=str, default="all",
                        choices=list(DATASET_CONFIG.keys()) + ["all"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--freeze_blocks", type=int, default=8)
    parser.add_argument("--poly_degree", type=int, default=2, choices=[2, 4])
    parser.add_argument("--softmax_depth", type=int, default=3)
    parser.add_argument("--init_method", type=str, default="gelu_fit",
                        choices=["gelu_fit", "identity"])
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="results_step3_poly_softmax")
    parser.add_argument("--baseline_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_dataloaders(dataset_name, batch_size):
    config = DATASET_CONFIG[dataset_name]
    DatasetClass = config["class"]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_dataset = DatasetClass(split="train", transform=train_transform,
                                 download=True, as_rgb=True)
    val_dataset   = DatasetClass(split="val",   transform=eval_transform,
                                 download=True, as_rgb=True)
    test_dataset  = DatasetClass(split="test",  transform=eval_transform,
                                 download=True, as_rgb=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, config


# ═══════════════════════════════════════════════════════════════
# Teacher and Student
# ═══════════════════════════════════════════════════════════════

def load_teacher(dataset_name, num_classes, baseline_dir, device):
    teacher = timm.create_model("deit_tiny_patch16_224", pretrained=False,
                                num_classes=num_classes)
    ckpt_path = os.path.join(baseline_dir, dataset_name, "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Teacher not found: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    teacher.load_state_dict(state_dict, strict=True)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    teacher = teacher.to(device)
    print(f"  Teacher: loaded from {ckpt_path}")
    return teacher


def create_student(num_classes, freeze_blocks, poly_degree, softmax_depth,
                   init_method, device):
    """
    Polynomial GELU + polynomial softmax student. LayerNorm KEPT.

    Replaced: 12 GELU + 12 softmax = 24 non-polynomial ops
    Kept: 25 LayerNorm (handled at FHE inference time)
    """
    student = timm.create_model("deit_tiny_patch16_224", pretrained=True,
                                num_classes=num_classes)
    print(f"  Student: ImageNet pretrained (cold-start)")

    n_gelu = replace_gelu(student, degree=poly_degree, init_method=init_method)
    print(f"  Replaced {n_gelu} GELU → degree-{poly_degree} polynomial")

    n_softmax = replace_attention_softmax(student, depth=softmax_depth)
    print(f"  Replaced {n_softmax} softmax → poly_softmax (depth {softmax_depth})")

    # Count remaining LayerNorm (kept intentionally)
    n_ln = sum(1 for _, m in student.named_modules() if isinstance(m, nn.LayerNorm))
    print(f"  Kept {n_ln} LayerNorm (handled at FHE inference time)")
    print(f"  Non-poly ops replaced: {n_gelu + n_softmax}/49 | Remaining: {n_ln} LayerNorm")

    # Verify GELU replaced
    n_gelu_remain = sum(1 for _, m in student.named_modules() if isinstance(m, nn.GELU))
    assert n_gelu_remain == 0, f"GELU still present: {n_gelu_remain}"

    # Quick forward test to catch NaN early
    student_test = student.to(device)
    with torch.no_grad():
        test_in = torch.randn(1, 3, 224, 224).to(device)
        test_out = student_test(test_in)
        if torch.isnan(test_out).any():
            print(f"  ⚠ WARNING: NaN detected in forward pass!")
            print(f"  This should not happen with LayerNorm kept.")
        else:
            print(f"  ✓ Forward pass OK (no NaN)")

    # Freeze blocks
    for param in student.patch_embed.parameters():
        param.requires_grad = False
    if hasattr(student, "cls_token") and student.cls_token is not None:
        student.cls_token.requires_grad = False
    if hasattr(student, "pos_embed") and student.pos_embed is not None:
        student.pos_embed.requires_grad = False

    total_blocks = len(student.blocks)
    actual_freeze = min(freeze_blocks, total_blocks)
    for i in range(actual_freeze):
        for param in student.blocks[i].parameters():
            param.requires_grad = False
        # Keep poly GELU coefficients trainable
        act = student.blocks[i].mlp.act
        if isinstance(act, PolyActivation):
            act.coeffs.requires_grad = True

    student = student.to(device)

    total_params = sum(p.numel() for p in student.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad) / 1e6
    print(f"  Student: {total_params:.2f}M total | {trainable_params:.2f}M trainable")
    print(f"  Blocks frozen: {actual_freeze}/{total_blocks}")

    return student


# ═══════════════════════════════════════════════════════════════
# Training and Evaluation
# ═══════════════════════════════════════════════════════════════

def train_one_epoch_kd(student, teacher, loader, optimizer, device,
                       temperature, alpha):
    student.train()
    running_total, running_distill, running_task = 0.0, 0.0, 0.0
    correct, total = 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        student_logits = student(images)
        with torch.no_grad():
            teacher_logits = teacher(images)

        optimizer.zero_grad()
        total_loss, distill_val, task_val = kd_loss(
            student_logits, teacher_logits, labels, temperature, alpha
        )

        # NaN safety: skip batch if loss is NaN
        if torch.isnan(total_loss):
            continue

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        running_total += total_loss.item()
        running_distill += distill_val
        running_task += task_val
        _, predicted = student_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    n = len(loader)
    acc = 100.0 * correct / total if total > 0 else 0.0
    return (running_total / n, running_distill / n, running_task / n, acc)


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.squeeze().long()
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_probs.append(probs)
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    accuracy = 100.0 * (all_preds == all_labels).mean()

    try:
        if n_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            labels_oh = label_binarize(all_labels, classes=list(range(n_classes)))
            auc = roc_auc_score(labels_oh, all_probs, multi_class="ovr", average="macro")
    except Exception as e:
        print(f"  Warning: AUC failed ({e})")
        auc = 0.0

    return accuracy, auc, all_preds, all_labels


def load_json_results(directory, dataset_name):
    path = os.path.join(directory, dataset_name, "results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════

def run_experiment(dataset_name, args, device):
    print(f"\n{'='*70}")
    print(f"  Step 3: Poly GELU + Poly Softmax — {dataset_name.upper()}")
    print(f"  GELU deg: {args.poly_degree} | Softmax depth: {args.softmax_depth} | "
          f"τ={args.temperature} α={args.alpha}")
    print(f"{'='*70}")

    config = DATASET_CONFIG[dataset_name]
    n_classes = config["n_classes"]

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        dataset_name, args.batch_size
    )

    teacher = load_teacher(dataset_name, n_classes, args.baseline_dir, device)
    teacher_acc, teacher_auc, _, _ = evaluate(teacher, test_loader, device, n_classes)
    print(f"  Teacher test acc: {teacher_acc:.2f}% | AUC: {teacher_auc:.4f}")

    student = create_student(
        n_classes, args.freeze_blocks, args.poly_degree,
        args.softmax_depth, args.init_method, device
    )

    # Load Step 5B results for comparison
    step5b_result = load_json_results("results_step5b_coldstart_kd", dataset_name)
    step5b_acc = step5b_result["test_accuracy"] if step5b_result else None

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999),
    )
    warmup_epochs = 5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6
    )

    best_val_auc, best_val_acc, best_epoch = 0.0, 0.0, 0
    history = []

    save_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    for old in ["best_model.pth", "final_model.pth"]:
        p = os.path.join(save_dir, old)
        if os.path.exists(p):
            os.remove(p)

    print(f"\n  Training for {args.epochs} epochs (poly GELU + poly softmax, cold-start KD)...")

    start_time = time.time()

    for epoch in range(args.epochs):
        ep = epoch + 1

        if epoch < warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        total_loss, distill_loss, task_loss, train_acc = train_one_epoch_kd(
            student, teacher, train_loader, optimizer, device,
            args.temperature, args.alpha
        )

        val_acc, val_auc, _, _ = evaluate(student, val_loader, device, n_classes)

        if epoch >= warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_epoch = ep
            torch.save(student.state_dict(), os.path.join(save_dir, "best_model.pth"))

        history.append({
            "epoch": ep, "total_loss": round(total_loss, 4),
            "distill_loss": round(distill_loss, 4), "task_loss": round(task_loss, 4),
            "train_acc": round(train_acc, 2), "val_acc": round(val_acc, 2),
            "val_auc": round(val_auc, 4), "lr": round(current_lr, 7),
        })

        print(f"  Epoch {ep:2d}/{args.epochs} | "
              f"Total: {total_loss:.4f} (D:{distill_loss:.3f} T:{task_loss:.3f}) | "
              f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | "
              f"AUC: {val_auc:.4f} | LR: {current_lr:.2e}")

    train_time = time.time() - start_time
    torch.save(student.state_dict(), os.path.join(save_dir, "final_model.pth"))

    best_ckpt = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(best_ckpt) and best_epoch > 0:
        student.load_state_dict(torch.load(best_ckpt, weights_only=True))
        print(f"\n  Loaded best checkpoint (epoch {best_epoch})")
    else:
        best_epoch = args.epochs

    test_acc, test_auc, test_preds, test_labels = evaluate(
        student, test_loader, device, n_classes
    )

    acc_delta = test_acc - teacher_acc
    final_coeffs = get_poly_coefficients(student)

    print(f"\n  ┌───────────────────────────────────────────────────────────────┐")
    print(f"  │  RESULTS: Poly GELU + Poly Softmax + KD (LayerNorm kept)     │")
    print(f"  │  Dataset: {dataset_name.upper():<15s}                                  │")
    print(f"  ├───────────────────────────────────────────────────────────────┤")
    print(f"  │  Step 1 Baseline (teacher):   {teacher_acc:6.2f}%                    │")
    if step5b_acc is not None:
        print(f"  │  Step 5B (GELU-only + KD):    {step5b_acc:6.2f}%                    │")
    print(f"  │  Step 3 (GELU+softmax + KD):  {test_acc:6.2f}%  (AUC: {test_auc:.4f}) │")
    print(f"  │                                                               │")
    print(f"  │  Δ vs Baseline:               {acc_delta:+6.2f}%                      │")
    if step5b_acc is not None:
        print(f"  │  Δ vs GELU-only (Step 5B):    {test_acc - step5b_acc:+6.2f}%                      │")
    print(f"  │  Ops replaced: 24/49 (12 GELU + 12 softmax)                  │")
    print(f"  │  Ops remaining: 25 LayerNorm (FHE inference handled)          │")
    print(f"  │  Best Epoch: {best_epoch:3d} | Train Time: {train_time:.1f}s                │")
    print(f"  └───────────────────────────────────────────────────────────────┘")

    print(f"\n  Learned GELU polynomial coefficients:")
    for name, c in final_coeffs.items():
        block_idx = name.split('.')[1] if 'blocks' in name else '?'
        if len(c) >= 3:
            print(f"    Block {block_idx}: f(x) = {c[2]:.4f}·x² + {c[1]:.4f}·x + {c[0]:.4f}")

    target_names = [f"Class {i}" for i in range(n_classes)]
    print(f"\n  Per-class classification report:")
    print(classification_report(
        test_labels, test_preds, target_names=target_names,
        digits=3, zero_division=0
    ))

    results = {
        "step": 3,
        "description": (f"Poly GELU (deg-{args.poly_degree}) + poly_softmax (depth-{args.softmax_depth}) "
                        f"+ cold-start KD (τ={args.temperature}, α={args.alpha}). LayerNorm kept."),
        "dataset": dataset_name,
        "test_accuracy": round(test_acc, 2),
        "test_auc": round(test_auc, 4),
        "baseline_accuracy": round(teacher_acc, 2),
        "baseline_auc": round(teacher_auc, 4),
        "step5b_accuracy": step5b_acc,
        "accuracy_delta_vs_baseline": round(acc_delta, 2),
        "ops_replaced": {"gelu": 12, "softmax": 12, "total_replaced": 24, "layernorm_kept": 25},
        "poly_degree": args.poly_degree,
        "softmax_depth": args.softmax_depth,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "train_time_seconds": round(train_time, 1),
        "learned_gelu_coefficients": final_coeffs,
        "history": history,
    }

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {save_dir}/results.json")

    return results


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  Step 3: Poly GELU + Poly Softmax + Cold-Start KD           ║")
    print(f"  ║  Replacing: 12 GELU + 12 softmax = 24 ops                   ║")
    print(f"  ║  Keeping:   25 LayerNorm (handled at FHE inference)          ║")
    print(f"  ║  GELU deg: {args.poly_degree}  |  Softmax depth: {args.softmax_depth}  |  τ={args.temperature}  α={args.alpha}    ║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")

    if not os.path.exists(args.baseline_dir):
        print(f"\n  ERROR: Baseline directory '{args.baseline_dir}' not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    datasets = list(DATASET_CONFIG.keys()) if args.dataset == "all" else [args.dataset]

    all_results = {}
    for ds in datasets:
        result = run_experiment(ds, args, device)
        all_results[ds] = result

    # Summary
    print(f"\n{'='*85}")
    print(f"  SUMMARY: Step 3 — Poly GELU + Poly Softmax + KD (LayerNorm kept)")
    print(f"{'='*85}")
    print(f"  {'Dataset':<14} {'Baseline':>9} {'GELU-KD':>9} {'GELU+SM':>9} {'Δ Base':>8} {'Δ GELU-KD':>10}")
    print(f"  {'─'*64}")

    for ds, res in all_results.items():
        ba = res["baseline_accuracy"]
        s5b = res.get("step5b_accuracy")
        s3 = res["test_accuracy"]
        db = res["accuracy_delta_vs_baseline"]
        d5b = round(s3 - s5b, 2) if s5b is not None else None

        s5b_str = f"{s5b:.2f}%" if s5b is not None else "N/A"
        d5b_str = f"{d5b:+.2f}%" if d5b is not None else "N/A"

        print(f"  {ds:<14} {ba:>8.2f}% {s5b_str:>9} {s3:>8.2f}% {db:>+7.2f}% {d5b_str:>10}")

    print(f"  {'─'*64}")
    deltas = [r["accuracy_delta_vs_baseline"] for r in all_results.values()]
    print(f"\n  Average Δ vs baseline: {np.mean(deltas):+.2f}%")
    print(f"  Ops replaced: 24/49 (GELU + softmax)")
    print(f"  Ops remaining: 25 LayerNorm")
    print(f"\n  LayerNorm handling at FHE inference:")
    print(f"  → Client-aided normalization (decrypt-normalize-re-encrypt)")
    print(f"  → Or polynomial reciprocal sqrt approximation (+4-6 levels)")
    print(f"  → Detailed in thesis as future work / system design")

    summary = {ds: {
        "baseline": r["baseline_accuracy"],
        "step5b_gelu_only": r.get("step5b_accuracy"),
        "step3_gelu_softmax": r["test_accuracy"],
        "delta_baseline": r["accuracy_delta_vs_baseline"],
    } for ds, r in all_results.items()}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()