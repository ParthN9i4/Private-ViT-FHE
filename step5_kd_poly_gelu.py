"""
Step 5: Knowledge Distillation for Polynomial-GELU DeiT-Tiny
=============================================================
Core contribution: Demonstrating that KD recovers accuracy lost from
polynomial activation replacement in Vision Transformers for medical imaging.

Architecture:
  Teacher: DeiT-Tiny with standard GELU (Step 1 checkpoint, frozen)
  Student: DeiT-Tiny with degree-2 polynomial activations (trainable)

KD Loss (Hinton et al., 2015):
  L = α · T² · KL(softmax(z_t/T) || softmax(z_s/T)) + (1-α) · CE(y, z_s)

  Where:
  - z_t = teacher logits, z_s = student logits
  - T = temperature (softens distributions to reveal "dark knowledge")
  - α = distillation weight (balances task loss vs teacher guidance)
  - T² factor compensates for 1/T² gradient scaling in KL divergence

  The "dark knowledge" insight (Hinton, 2015): A teacher's soft predictions
  like [0.7, 0.2, 0.05, 0.05] contain more information than hard labels
  [1, 0, 0, 0]. The 0.2 for class 1 tells the student "this image has some
  similarity to class 1" — information that one-hot labels cannot convey.
  This is especially valuable for small datasets where each sample's
  information content must be maximized.

Hyperparameters from literature:
  Baruch et al. (2022): τ=10, α=0.1 (for HE-friendly CNNs)
  MGF-softmax (2026):   τ=2.0, α=0.5 (for polynomial transformers)
  DeiT (2021):          τ=3.0, α=0.5 (for ViT distillation)
  Default here:         τ=4.0, α=0.5 (balanced, between Baruch and DeiT)

Training strategy:
  Same warm-start + smooth transition as Step 2v2, but with KD active
  throughout. The teacher provides soft labels during ALL phases:
  - GELU phase: KD helps maintain accuracy during warmup
  - Transition phase: KD guides the polynomial to mimic GELU behavior
  - POLY phase: KD recovers accuracy lost from polynomial approximation

References:
  - Hinton et al. (2015): "Distilling the Knowledge in a Neural Network"
  - Baruch et al. (ACNS 2022): KD for HE-friendly NNs (1.1-3.2% recovery)
  - Touvron et al. (ICML 2021): DeiT distillation token approach
  - MGF-softmax (arXiv 2602.01621): KD for polynomial ViT softmax

Usage:
    python step5_kd_poly_gelu.py --dataset all
    python step5_kd_poly_gelu.py --dataset blood --temperature 10 --alpha 0.1
    python step5_kd_poly_gelu.py --dataset retina --temperature 2.0 --alpha 0.5
"""

import os
import argparse
import json
import time
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
# Polynomial Activation (same as Step 2v2)
# ═══════════════════════════════════════════════════════════════

class PolyActivation(nn.Module):
    """Trainable degree-n polynomial via Horner's method."""
    def __init__(self, degree=2, init_method="gelu_fit"):
        super().__init__()
        self.degree = degree
        if init_method == "gelu_fit":
            if degree == 2:
                init_coeffs = [0.0711, 0.5, 0.2576]
            elif degree == 4:
                init_coeffs = [0.0711, 0.5, 0.2576, 0.0, -0.0128]
            else:
                init_coeffs = [0.0] * (degree + 1)
                init_coeffs[1] = 0.5
        elif init_method == "identity":
            init_coeffs = [0.0] * (degree + 1)
            init_coeffs[1] = 1.0
        else:
            init_coeffs = [0.0] * (degree + 1)
            init_coeffs[1] = 0.5
        self.coeffs = nn.Parameter(torch.tensor(init_coeffs, dtype=torch.float32))

    def forward(self, x):
        result = self.coeffs[-1]
        for i in range(len(self.coeffs) - 2, -1, -1):
            result = result * x + self.coeffs[i]
        return result


class SmoothTransitionActivation(nn.Module):
    """Blend: output = (1-α)·GELU(x) + α·poly(x)"""
    def __init__(self, degree=2, init_method="gelu_fit"):
        super().__init__()
        self.gelu = nn.GELU()
        self.poly = PolyActivation(degree=degree, init_method=init_method)
        self.alpha = 0.0

    def set_alpha(self, alpha):
        self.alpha = max(0.0, min(1.0, alpha))

    def forward(self, x):
        if self.alpha <= 0.0:
            return self.gelu(x)
        elif self.alpha >= 1.0:
            return self.poly(x)
        else:
            return (1.0 - self.alpha) * self.gelu(x) + self.alpha * self.poly(x)


# ═══════════════════════════════════════════════════════════════
# Knowledge Distillation Loss
# ═══════════════════════════════════════════════════════════════

def kd_loss(student_logits, teacher_logits, labels, temperature, alpha,
            label_smoothing=0.1):
    """
    Combined KD + task loss (Hinton et al., 2015).

    L = α · T² · KL(p_teacher^T || p_student^T) + (1-α) · CE(y, z_student)

    Args:
        student_logits: Raw logits from student model [B, C]
        teacher_logits: Raw logits from teacher model [B, C] (detached)
        labels: Ground truth class indices [B]
        temperature: Softmax temperature for soft targets
        alpha: Weight for distillation loss (0=task only, 1=distill only)
        label_smoothing: Label smoothing for task loss

    Returns:
        total_loss, distill_loss_value, task_loss_value
    """
    # Soft targets: teacher's predictions softened by temperature
    # Higher T → softer distribution → more "dark knowledge" transferred
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    log_soft_student = F.log_softmax(student_logits / temperature, dim=1)

    # KL divergence: measures how student's soft predictions differ from teacher's
    # F.kl_div expects log-probabilities as input and probabilities as target
    # reduction='batchmean' averages over batch (standard for KD)
    # T² compensates for the 1/T² factor in gradient magnitudes
    distill_loss = F.kl_div(
        log_soft_student, soft_teacher, reduction='batchmean'
    ) * (temperature ** 2)

    # Hard target: standard cross-entropy with true labels
    task_loss = F.cross_entropy(
        student_logits, labels, label_smoothing=label_smoothing
    )

    # Combined loss
    total = alpha * distill_loss + (1.0 - alpha) * task_loss

    return total, distill_loss.item(), task_loss.item()


# ═══════════════════════════════════════════════════════════════
# Model Utilities
# ═══════════════════════════════════════════════════════════════

def replace_gelu_with_poly(model, degree=2, init_method="gelu_fit"):
    """Replace all nn.GELU in DeiT MLP blocks with SmoothTransitionActivation."""
    replaced = 0
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'mlp') and hasattr(block.mlp, 'act'):
            if isinstance(block.mlp.act, nn.GELU):
                block.mlp.act = SmoothTransitionActivation(
                    degree=degree, init_method=init_method
                )
                replaced += 1
    return replaced


def set_transition_alpha(model, alpha):
    for module in model.modules():
        if isinstance(module, SmoothTransitionActivation):
            module.set_alpha(alpha)


def get_poly_coefficients(model):
    coeffs = {}
    for name, module in model.named_modules():
        if isinstance(module, SmoothTransitionActivation):
            coeffs[name] = module.poly.coeffs.data.cpu().tolist()
        elif isinstance(module, PolyActivation) and '.poly' not in name:
            coeffs[name] = module.coeffs.data.cpu().tolist()
    return coeffs


def compute_transition_alpha(epoch, transition_start, transition_end):
    if epoch < transition_start:
        return 0.0
    elif epoch >= transition_end:
        return 1.0
    else:
        return (epoch - transition_start) / (transition_end - transition_start)


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
        description="Step 5: Knowledge Distillation for polynomial-GELU DeiT-Tiny"
    )
    parser.add_argument("--dataset", type=str, default="all",
                        choices=list(DATASET_CONFIG.keys()) + ["all"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--freeze_blocks", type=int, default=8)
    parser.add_argument("--poly_degree", type=int, default=2, choices=[2, 4])
    parser.add_argument("--init_method", type=str, default="gelu_fit",
                        choices=["gelu_fit", "identity"])

    # KD hyperparameters
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="KD temperature (default: 4.0)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="KD weight: 0=task only, 1=distill only (default: 0.5)")

    # Transition schedule
    parser.add_argument("--transition_start", type=int, default=3)
    parser.add_argument("--transition_end", type=int, default=15)

    # Paths
    parser.add_argument("--output_dir", type=str, default="results_step5_kd")
    parser.add_argument("--baseline_dir", type=str, default="results",
                        help="Directory with Step 1 checkpoints (teacher source)")
    parser.add_argument("--step2_dir", type=str, default=None,
                        help="Directory with Step 2 results (for comparison)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# Data Loading (identical to Step 1/2 for fair comparison)
# ═══════════════════════════════════════════════════════════════

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

    train_dataset = DatasetClass(split="train", transform=train_transform, download=True, as_rgb=True)
    val_dataset   = DatasetClass(split="val",   transform=eval_transform,  download=True, as_rgb=True)
    test_dataset  = DatasetClass(split="test",  transform=eval_transform,  download=True, as_rgb=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, config


# ═══════════════════════════════════════════════════════════════
# Teacher Model (frozen, standard GELU)
# ═══════════════════════════════════════════════════════════════

def load_teacher(dataset_name, num_classes, baseline_dir, device):
    """
    Load Step 1 checkpoint as teacher model.
    Teacher has standard GELU activations and is completely frozen.
    """
    teacher = timm.create_model("deit_tiny_patch16_224", pretrained=False,
                                num_classes=num_classes)

    ckpt_path = os.path.join(baseline_dir, dataset_name, "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {ckpt_path}\n"
            f"Run Step 1 first: python baseline_deit_improved.py --dataset {dataset_name}"
        )

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    teacher.load_state_dict(state_dict, strict=True)
    print(f"  Teacher: loaded from {ckpt_path}")

    # Freeze ALL teacher parameters — teacher never trains
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    teacher = teacher.to(device)
    return teacher


# ═══════════════════════════════════════════════════════════════
# Student Model (polynomial GELU, warm-started from Step 1)
# ═══════════════════════════════════════════════════════════════

def create_student(dataset_name, num_classes, freeze_blocks, poly_degree,
                   init_method, baseline_dir, device):
    """
    Create polynomial-GELU student, warm-started from Step 1 checkpoint.
    Same approach as Step 2v2: load checkpoint → replace GELU → freeze blocks.
    """
    student = timm.create_model("deit_tiny_patch16_224", pretrained=False,
                                num_classes=num_classes)

    # Load Step 1 checkpoint (same starting point as teacher)
    ckpt_path = os.path.join(baseline_dir, dataset_name, "best_model.pth")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    student.load_state_dict(state_dict, strict=True)
    print(f"  Student: warm-started from {ckpt_path}")

    # Replace GELU → polynomial (starts at α=0 = pure GELU)
    n_replaced = replace_gelu_with_poly(student, degree=poly_degree,
                                        init_method=init_method)
    print(f"  Student: replaced {n_replaced} GELU → degree-{poly_degree} polynomial")

    # Freeze blocks (same as Steps 1 and 2)
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
        # Keep polynomial coefficients trainable in frozen blocks
        act = student.blocks[i].mlp.act
        if isinstance(act, SmoothTransitionActivation):
            act.poly.coeffs.requires_grad = True

    student = student.to(device)

    total_params = sum(p.numel() for p in student.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad) / 1e6
    print(f"  Student: {total_params:.2f}M total | {trainable_params:.2f}M trainable")

    return student


# ═══════════════════════════════════════════════════════════════
# Training with KD
# ═══════════════════════════════════════════════════════════════

def train_one_epoch_kd(student, teacher, loader, optimizer, device,
                       temperature, alpha):
    """
    Train student for one epoch with KD loss from frozen teacher.

    Both student and teacher see the same input batch.
    Teacher is always in eval mode; student is in train mode.
    """
    student.train()
    # Teacher stays in eval mode (frozen, no dropout, fixed batchnorm)

    running_total_loss = 0.0
    running_distill_loss = 0.0
    running_task_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        # Forward pass through both models
        student_logits = student(images)

        with torch.no_grad():
            teacher_logits = teacher(images)

        # Compute KD loss
        optimizer.zero_grad()
        total_loss, distill_val, task_val = kd_loss(
            student_logits, teacher_logits, labels, temperature, alpha
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        running_total_loss += total_loss.item()
        running_distill_loss += distill_val
        running_task_loss += task_val
        _, predicted = student_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    n = len(loader)
    return (running_total_loss / n, running_distill_loss / n,
            running_task_loss / n, 100.0 * correct / total)


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


# ═══════════════════════════════════════════════════════════════
# Load Previous Results for Comparison
# ═══════════════════════════════════════════════════════════════

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
    print(f"  Step 5: KD for Polynomial GELU — {dataset_name.upper()}")
    print(f"  Temperature: {args.temperature} | Alpha: {args.alpha} | Degree: {args.poly_degree}")
    print(f"{'='*70}")

    config = DATASET_CONFIG[dataset_name]
    n_classes = config["n_classes"]

    # Data
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        dataset_name, args.batch_size
    )

    # ── Load Teacher (frozen Step 1 model) ──
    teacher = load_teacher(dataset_name, n_classes, args.baseline_dir, device)

    # Verify teacher accuracy
    teacher_acc, teacher_auc, _, _ = evaluate(teacher, test_loader, device, n_classes)
    print(f"  Teacher test acc: {teacher_acc:.2f}% | AUC: {teacher_auc:.4f}")

    # ── Create Student (polynomial GELU, warm-started) ──
    student = create_student(
        dataset_name, n_classes, args.freeze_blocks,
        args.poly_degree, args.init_method, args.baseline_dir, device
    )

    # Verify student starts identical to teacher (at α=0)
    set_transition_alpha(student, 0.0)
    student_init_acc, student_init_auc, _, _ = evaluate(
        student, test_loader, device, n_classes
    )
    acc_match = abs(student_init_acc - teacher_acc) < 0.1
    print(f"  Student init acc: {student_init_acc:.2f}% (match teacher: "
          f"{'✓' if acc_match else '✗'})")

    # Load Step 2 results for comparison
    step2_result = None
    if args.step2_dir:
        step2_result = load_json_results(args.step2_dir, dataset_name)
    # Try default locations
    for s2_dir in ["results_step2v2", "results_step2"]:
        if step2_result is None:
            step2_result = load_json_results(s2_dir, dataset_name)

    step2_acc = step2_result["test_accuracy"] if step2_result else None

    # ── Optimizer and Scheduler ──
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999),
    )
    warmup_epochs = 2
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6
    )

    # ── Training ──
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    history = []

    save_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # Clear stale checkpoints
    for old_ckpt in ["best_model.pth", "final_model.pth"]:
        old_path = os.path.join(save_dir, old_ckpt)
        if os.path.exists(old_path):
            os.remove(old_path)

    print(f"\n  Training for {args.epochs} epochs with KD (τ={args.temperature}, α={args.alpha})...")
    print(f"  Schedule: warmup (ep 1-{warmup_epochs}) → "
          f"transition (ep {args.transition_start}-{args.transition_end}) → "
          f"pure poly (ep {args.transition_end+1}-{args.epochs})")

    start_time = time.time()

    for epoch in range(args.epochs):
        ep = epoch + 1

        # Set transition alpha
        alpha_transition = compute_transition_alpha(
            ep, args.transition_start, args.transition_end
        )
        set_transition_alpha(student, alpha_transition)

        # LR warmup
        if epoch < warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # Train with KD
        total_loss, distill_loss, task_loss, train_acc = train_one_epoch_kd(
            student, teacher, train_loader, optimizer, device,
            args.temperature, args.alpha
        )

        # Evaluate student
        val_acc, val_auc, _, _ = evaluate(student, val_loader, device, n_classes)

        if epoch >= warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # Save best model from POLY phase only (lesson from Step 2v2 bug)
        if alpha_transition >= 1.0 and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_epoch = ep
            torch.save(student.state_dict(), os.path.join(save_dir, "best_model.pth"))

        history.append({
            "epoch": ep,
            "alpha_transition": round(alpha_transition, 3),
            "total_loss": round(total_loss, 4),
            "distill_loss": round(distill_loss, 4),
            "task_loss": round(task_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_acc": round(val_acc, 2),
            "val_auc": round(val_auc, 4),
            "lr": round(current_lr, 7),
        })

        # Phase indicator
        if alpha_transition <= 0:
            phase = "GELU"
        elif alpha_transition >= 1:
            phase = "POLY"
        else:
            phase = f"α={alpha_transition:.2f}"

        print(f"  Epoch {ep:2d}/{args.epochs} [{phase:>7s}] | "
              f"Total: {total_loss:.4f} (D:{distill_loss:.3f} T:{task_loss:.3f}) | "
              f"Train: {train_acc:.1f}% | "
              f"Val: {val_acc:.1f}% | "
              f"AUC: {val_auc:.4f} | "
              f"LR: {current_lr:.2e}")

    train_time = time.time() - start_time

    # Save final model
    torch.save(student.state_dict(), os.path.join(save_dir, "final_model.pth"))

    # ── Final Test ──
    best_ckpt = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(best_ckpt) and best_epoch > 0:
        student.load_state_dict(torch.load(best_ckpt, weights_only=True))
        print(f"\n  Loaded best POLY-phase checkpoint (epoch {best_epoch})")
    else:
        print(f"\n  Using final epoch model (epoch {args.epochs})")
        best_epoch = args.epochs

    set_transition_alpha(student, 1.0)  # Ensure pure polynomial

    test_acc, test_auc, test_preds, test_labels = evaluate(
        student, test_loader, device, n_classes
    )

    # ── Results Comparison ──
    baseline_acc = teacher_acc
    baseline_auc = teacher_auc
    acc_delta_baseline = test_acc - baseline_acc
    auc_delta_baseline = test_auc - baseline_auc

    # KD recovery = how much of Step 2's loss was recovered
    if step2_acc is not None:
        step2_drop = step2_acc - baseline_acc        # negative
        kd_drop = test_acc - baseline_acc             # negative but smaller
        recovery = (step2_drop - kd_drop)             # positive = KD helped
        recovery_pct = (recovery / abs(step2_drop) * 100) if step2_drop != 0 else 0
    else:
        recovery = None
        recovery_pct = None

    final_coeffs = get_poly_coefficients(student)

    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  RESULTS: KD + Poly-GELU DeiT-Tiny                         │")
    print(f"  │  Dataset: {dataset_name.upper():<15s}  τ={args.temperature}  α={args.alpha}           │")
    print(f"  ├─────────────────────────────────────────────────────────────┤")
    print(f"  │  Step 1 Baseline (teacher): {baseline_acc:6.2f}%  (AUC: {baseline_auc:.4f})│")
    if step2_acc is not None:
        print(f"  │  Step 2 Poly-GELU (no KD):  {step2_acc:6.2f}%                      │")
    print(f"  │  Step 5 Poly-GELU + KD:     {test_acc:6.2f}%  (AUC: {test_auc:.4f})│")
    print(f"  │                                                             │")
    print(f"  │  Δ vs Baseline:             {acc_delta_baseline:+6.2f}%                     │")
    if step2_acc is not None and recovery is not None:
        print(f"  │  KD Recovery:               {recovery:+6.2f}%  ({recovery_pct:.1f}% recovered)│")
    print(f"  │  Best Epoch:                {best_epoch:3d}                            │")
    print(f"  │  Train Time:                {train_time:.1f}s                        │")
    print(f"  └─────────────────────────────────────────────────────────────┘")

    # Learned coefficients
    print(f"\n  Learned polynomial coefficients per block:")
    for name, c in final_coeffs.items():
        block_idx = name.split('.')[1] if 'blocks' in name else '?'
        if len(c) >= 3:
            print(f"    Block {block_idx}: f(x) = {c[2]:.4f}·x² + {c[1]:.4f}·x + {c[0]:.4f}")

    # Per-class report
    target_names = [f"Class {i}" for i in range(n_classes)]
    print(f"\n  Per-class classification report:")
    print(classification_report(
        test_labels, test_preds, target_names=target_names,
        digits=3, zero_division=0
    ))

    # Save results
    results = {
        "step": 5,
        "description": f"KD (τ={args.temperature}, α={args.alpha}) + degree-{args.poly_degree} polynomial GELU",
        "dataset": dataset_name,
        "test_accuracy": round(test_acc, 2),
        "test_auc": round(test_auc, 4),
        "baseline_accuracy": round(baseline_acc, 2),
        "baseline_auc": round(baseline_auc, 4),
        "step2_accuracy": step2_acc,
        "accuracy_delta_vs_baseline": round(acc_delta_baseline, 2),
        "kd_recovery": round(recovery, 2) if recovery is not None else None,
        "kd_recovery_pct": round(recovery_pct, 1) if recovery_pct is not None else None,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "poly_degree": args.poly_degree,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "lr": args.lr,
        "freeze_blocks": args.freeze_blocks,
        "train_time_seconds": round(train_time, 1),
        "learned_coefficients": final_coeffs,
        "history": history,
    }

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {save_dir}/results.json")

    return results


# ═══════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════

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
    print(f"  ║  Step 5: Knowledge Distillation + Polynomial GELU           ║")
    print(f"  ║  Teacher: Step 1 DeiT-Tiny (GELU, frozen)                   ║")
    print(f"  ║  Student: DeiT-Tiny (poly GELU, warm-start)                 ║")
    print(f"  ║  KD: τ={args.temperature:<4}  α={args.alpha:<4}  Degree: {args.poly_degree}                       ║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")

    if not os.path.exists(args.baseline_dir):
        print(f"\n  ERROR: Baseline directory '{args.baseline_dir}' not found.")
        print(f"  Run Step 1 first: python baseline_deit_improved.py --dataset all")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    datasets = list(DATASET_CONFIG.keys()) if args.dataset == "all" else [args.dataset]

    all_results = {}
    for ds in datasets:
        result = run_experiment(ds, args, device)
        all_results[ds] = result

    # ── Summary Table ──
    print(f"\n{'='*85}")
    print(f"  SUMMARY: Step 5 — KD + Polynomial GELU (τ={args.temperature}, α={args.alpha})")
    print(f"{'='*85}")
    print(f"  {'Dataset':<14} {'Baseline':>9} {'Step2':>9} {'Step5 KD':>9} {'Δ Base':>8} {'KD Recov':>10}")
    print(f"  {'─'*62}")

    for ds, res in all_results.items():
        ba = res["baseline_accuracy"]
        s2 = res.get("step2_accuracy")
        s5 = res["test_accuracy"]
        db = res["accuracy_delta_vs_baseline"]
        kr = res.get("kd_recovery")
        kp = res.get("kd_recovery_pct")

        s2_str = f"{s2:.2f}%" if s2 is not None else "N/A"
        kr_str = f"+{kr:.2f}% ({kp:.0f}%)" if kr is not None else "N/A"

        print(f"  {ds:<14} {ba:>8.2f}% {s2_str:>9} {s5:>8.2f}% {db:>+7.2f}% {kr_str:>10}")

    print(f"  {'─'*62}")

    # Average stats
    deltas = [r["accuracy_delta_vs_baseline"] for r in all_results.values()]
    recoveries = [r["kd_recovery"] for r in all_results.values()
                  if r.get("kd_recovery") is not None]

    print(f"\n  Average Δ vs baseline: {np.mean(deltas):+.2f}%")
    if recoveries:
        print(f"  Average KD recovery:  {np.mean(recoveries):+.2f}%")

    print(f"\n  Interpretation:")
    print(f"  - Positive KD recovery = KD helped vs Step 2 (polynomial without KD)")
    print(f"  - This is the core finding: KD recovers accuracy lost from poly replacement")
    print(f"\n  Next: Step 3 (replace softmax) or Step 6 (encrypted inference with CKKS)")

    # Save summary
    summary = {ds: {
        "baseline": r["baseline_accuracy"],
        "step2": r.get("step2_accuracy"),
        "step5_kd": r["test_accuracy"],
        "delta_baseline": r["accuracy_delta_vs_baseline"],
        "kd_recovery": r.get("kd_recovery"),
        "kd_recovery_pct": r.get("kd_recovery_pct"),
    } for ds, r in all_results.items()}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()