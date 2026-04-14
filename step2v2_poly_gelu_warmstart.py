"""
Step 2 v2: GELU → Polynomial with Warm-Start from Step 1 Checkpoint
====================================================================
CRITICAL FIX: The original Step 2 trained from ImageNet weights, which
caused collapse on small datasets (RetinaMNIST, BreastMNIST, DermaMNIST).

Root cause: Small datasets lack sufficient gradient signal to simultaneously
learn (a) task-specific features, (b) classification head, AND (c) polynomial
coefficients during the activation transition.

Fix: Follow Baruch et al.'s (2022) actual two-phase methodology:
  Phase 1 (Step 1): Train with standard GELU to convergence → checkpoint
  Phase 2 (this script): Load checkpoint, replace GELU→poly, fine-tune

This way the model already knows the task; the ONLY thing changing is the
activation function. Even with 546 samples (BreastMNIST), this is tractable.

Verification: Before training begins, we evaluate the loaded checkpoint to
confirm it reproduces Step 1 baseline numbers exactly.

Based on:
- Baruch et al. (ACNS/SiMLA 2022): Two-phase training methodology
- CaPriDe (CVPR 2023): Polynomial GELU approximation coefficients
- AESPA (2022): Degree-2 polynomial sufficiency
- PolyTransformer (ICML 2024): Range-aware training

Usage:
    python step2v2_poly_gelu_warmstart.py --dataset all
    python step2v2_poly_gelu_warmstart.py --dataset retina
    python step2v2_poly_gelu_warmstart.py --dataset breast --poly_degree 4
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
# Polynomial Activation Module (unchanged from Step 2 v1)
# ═══════════════════════════════════════════════════════════════

class PolyActivation(nn.Module):
    """
    Trainable polynomial activation: f(x) = a_n·x^n + ... + a_1·x + a_0
    Evaluated via Horner's method for numerical stability and minimal
    multiplicative depth in CKKS (degree-2 = 1 mult level).
    """

    def __init__(self, degree=2, init_method="gelu_fit"):
        super().__init__()
        self.degree = degree

        if init_method == "gelu_fit":
            if degree == 2:
                # CaPriDe (CVPR 2023) degree-2 GELU fit: c, b, a
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
        # Horner's method: f(x) = c0 + x·(c1 + x·(c2 + ...))
        result = self.coeffs[-1]
        for i in range(len(self.coeffs) - 2, -1, -1):
            result = result * x + self.coeffs[i]
        return result

    def extra_repr(self):
        terms = []
        for i, c in enumerate(self.coeffs.data):
            if i == 0:
                terms.append(f"{c.item():.4f}")
            elif i == 1:
                terms.append(f"{c.item():.4f}·x")
            else:
                terms.append(f"{c.item():.4f}·x^{i}")
        return f"f(x) = {' + '.join(terms)}"


class SmoothTransitionActivation(nn.Module):
    """
    Smooth blend: output = (1-α)·GELU(x) + α·poly(x)
    α increases from 0→1 over the transition phase.
    At α=0, behavior is identical to the original GELU model.
    At α=1, behavior is pure polynomial (FHE-compatible).
    """

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
# Model Modification Functions
# ═══════════════════════════════════════════════════════════════

def replace_gelu_with_poly(model, degree=2, init_method="gelu_fit"):
    """
    Replace all nn.GELU in DeiT's MLP blocks with SmoothTransitionActivation.

    In timm's DeiT, GELU is at: model.blocks[i].mlp.act
    nn.GELU has NO parameters, so this doesn't affect loaded weights.
    The new SmoothTransitionActivation starts at α=0 (pure GELU),
    so the model's behavior is IDENTICAL to before replacement.
    """
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
    """Set blending factor for all SmoothTransitionActivation modules."""
    for module in model.modules():
        if isinstance(module, SmoothTransitionActivation):
            module.set_alpha(alpha)


def get_poly_coefficients(model):
    """Extract learned polynomial coefficients, one per block."""
    coeffs = {}
    for name, module in model.named_modules():
        if isinstance(module, SmoothTransitionActivation):
            # Use this entry (don't also capture the child PolyActivation)
            coeffs[name] = module.poly.coeffs.data.cpu().tolist()
        elif isinstance(module, PolyActivation) and '.poly' not in name:
            # Standalone PolyActivation (not inside SmoothTransition)
            coeffs[name] = module.coeffs.data.cpu().tolist()
    return coeffs


def compute_transition_alpha(epoch, transition_start, transition_end):
    """Linear alpha ramp: 0 before start, 0→1 during transition, 1 after end."""
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
        description="Step 2 v2: Warm-start polynomial GELU replacement"
    )
    parser.add_argument("--dataset", type=str, default="all",
                        choices=list(DATASET_CONFIG.keys()) + ["all"])
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs (default: 30, gives 15 POLY epochs)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5, lower since model already trained)")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--freeze_blocks", type=int, default=8)
    parser.add_argument("--poly_degree", type=int, default=2,
                        choices=[2, 4], help="Polynomial degree (default: 2)")
    parser.add_argument("--init_method", type=str, default="gelu_fit",
                        choices=["gelu_fit", "identity"])
    parser.add_argument("--transition_start", type=int, default=3,
                        help="Epoch to start transition (default: 3, earlier than v1)")
    parser.add_argument("--transition_end", type=int, default=15,
                        help="Epoch to complete transition (default: 15)")
    parser.add_argument("--output_dir", type=str, default="results_step2v2")
    parser.add_argument("--baseline_dir", type=str, default="results",
                        help="Directory containing Step 1 checkpoints and results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# Data Loading (identical to Step 1 for fair comparison)
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
# Model Creation with Warm-Start
# ═══════════════════════════════════════════════════════════════

def create_warmstart_poly_model(dataset_name, num_classes, freeze_blocks,
                                poly_degree, init_method, baseline_dir, device):
    """
    Create DeiT-Tiny with warm-start from Step 1 checkpoint.

    Process:
    1. Create DeiT-Tiny with standard GELU (matching Step 1 architecture)
    2. Load Step 1 checkpoint (task-adapted weights)
    3. Replace GELU → SmoothTransitionActivation (starts at α=0 = pure GELU)
    4. Apply freezing (same as Step 1 for fair comparison)

    After step 3, the model behaves IDENTICALLY to the Step 1 checkpoint
    because α=0 means only GELU is active. The polynomial coefficients
    are initialized but unused until α > 0.

    Key detail: nn.GELU has NO learnable parameters, so it contributes
    zero keys to the state_dict. The Step 1 checkpoint loads perfectly
    into the GELU-based model. Then replacing GELU with
    SmoothTransitionActivation only ADDS new parameters (poly.coeffs),
    it doesn't change any existing weights.
    """

    # Step 1: Create model matching Step 1 architecture
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False,
                              num_classes=num_classes)

    # Step 2: Load Step 1 checkpoint
    ckpt_path = os.path.join(baseline_dir, dataset_name, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"  ⚠ No Step 1 checkpoint found at {ckpt_path}")
        print(f"  Falling back to ImageNet pretrained weights")
        model = timm.create_model("deit_tiny_patch16_224", pretrained=True,
                                  num_classes=num_classes)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Load with strict=True — should match exactly since architectures are identical
        model.load_state_dict(state_dict, strict=True)
        print(f"  ✓ Loaded Step 1 checkpoint: {ckpt_path}")

    # Step 3: Replace GELU → SmoothTransitionActivation
    # At α=0, behavior is identical to the loaded checkpoint
    n_replaced = replace_gelu_with_poly(model, degree=poly_degree, init_method=init_method)
    print(f"  Replaced {n_replaced} GELU → degree-{poly_degree} polynomial (α=0, pure GELU mode)")

    # Step 4: Apply freezing (same blocks as Step 1)
    for param in model.patch_embed.parameters():
        param.requires_grad = False
    if hasattr(model, "cls_token") and model.cls_token is not None:
        model.cls_token.requires_grad = False
    if hasattr(model, "pos_embed") and model.pos_embed is not None:
        model.pos_embed.requires_grad = False

    total_blocks = len(model.blocks)
    actual_freeze = min(freeze_blocks, total_blocks)
    for i in range(actual_freeze):
        for param in model.blocks[i].parameters():
            param.requires_grad = False
        # Keep polynomial coefficients trainable even in frozen blocks
        act = model.blocks[i].mlp.act
        if isinstance(act, SmoothTransitionActivation):
            act.poly.coeffs.requires_grad = True

    model = model.to(device)

    # Report parameter counts
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    poly_param_count = sum(
        p.numel() for n, p in model.named_parameters() if 'coeffs' in n
    )

    print(f"  Total: {total_params:.2f}M | Trainable: {trainable_params:.2f}M")
    print(f"  Polynomial params: {poly_param_count} ({n_replaced} layers × {poly_degree+1} coeffs)")
    print(f"  Blocks frozen: {actual_freeze}/{total_blocks} (poly coeffs remain trainable)")

    return model


# ═══════════════════════════════════════════════════════════════
# Training and Evaluation
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total


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
# Main Experiment
# ═══════════════════════════════════════════════════════════════

def load_baseline_results(baseline_dir, dataset_name):
    path = os.path.join(baseline_dir, dataset_name, "results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def run_experiment(dataset_name, args, device):
    print(f"\n{'='*65}")
    print(f"  Step 2v2: Warm-Start Polynomial GELU — {dataset_name.upper()}")
    print(f"  Degree: {args.poly_degree} | Transition: epochs {args.transition_start}→{args.transition_end}")
    print(f"{'='*65}")

    config = DATASET_CONFIG[dataset_name]
    n_classes = config["n_classes"]

    # Data
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        dataset_name, args.batch_size
    )

    # Model: warm-start from Step 1 checkpoint
    model = create_warmstart_poly_model(
        dataset_name, n_classes, args.freeze_blocks,
        args.poly_degree, args.init_method, args.baseline_dir, device
    )

    # ── Verification: confirm checkpoint loaded correctly ──
    # At α=0, the model should reproduce Step 1 baseline numbers
    set_transition_alpha(model, 0.0)  # Ensure pure GELU mode
    verify_acc, verify_auc, _, _ = evaluate(model, test_loader, device, n_classes)

    baseline = load_baseline_results(args.baseline_dir, dataset_name)
    if baseline:
        baseline_acc = baseline["test_accuracy"]
        baseline_auc = baseline["test_auc"]
        acc_match = abs(verify_acc - baseline_acc) < 0.1  # Allow tiny floating point diff
        print(f"\n  ── Checkpoint Verification ──")
        print(f"  Loaded model test acc:  {verify_acc:.2f}%")
        print(f"  Step 1 baseline acc:    {baseline_acc:.2f}%")
        print(f"  Match: {'✓ PASS' if acc_match else '✗ MISMATCH — check checkpoint path'}")
        if not acc_match:
            print(f"  WARNING: Difference = {abs(verify_acc - baseline_acc):.2f}%")
            print(f"  This may indicate the wrong checkpoint was loaded.")
    else:
        baseline_acc = verify_acc
        baseline_auc = verify_auc
        print(f"\n  No Step 1 results found — using loaded model as baseline")
        print(f"  Baseline acc: {verify_acc:.2f}% | AUC: {verify_auc:.4f}")

    # Print initial polynomial coefficients
    init_coeffs = get_poly_coefficients(model)
    print(f"\n  Initial polynomial coefficients (before training):")
    for name, c in list(init_coeffs.items())[:2]:  # Show first 2 blocks
        block_idx = name.split('.')[1] if 'blocks' in name else '?'
        if len(c) >= 3:
            print(f"    Block {block_idx}: f(x) = {c[2]:.4f}·x² + {c[1]:.4f}·x + {c[0]:.4f}")

    # ── Training Setup ──
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Lower LR than Step 1 since model is already well-trained
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999),
    )

    warmup_epochs = 2  # Shorter warmup — model already initialized well
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6
    )

    # ── Training Loop ──
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    history = []

    save_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # Clear stale checkpoints from previous (buggy) runs
    for old_ckpt in ["best_model.pth", "final_model.pth"]:
        old_path = os.path.join(save_dir, old_ckpt)
        if os.path.exists(old_path):
            os.remove(old_path)

    print(f"\n  Training for {args.epochs} epochs (warm-start from Step 1)...")
    print(f"  Schedule: warmup (ep 1-{warmup_epochs}) → "
          f"transition (ep {args.transition_start}-{args.transition_end}) → "
          f"pure poly (ep {args.transition_end+1}-{args.epochs})")

    start_time = time.time()

    for epoch in range(args.epochs):
        ep = epoch + 1  # 1-indexed for display

        # Set transition alpha
        alpha = compute_transition_alpha(ep, args.transition_start, args.transition_end)
        set_transition_alpha(model, alpha)

        # LR warmup
        if epoch < warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate
        val_acc, val_auc, _, _ = evaluate(model, val_loader, device, n_classes)

        if epoch >= warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # BUG FIX: Only save best model from POLY phase (α=1.0)
        # Reason: We will evaluate at α=1.0 (pure polynomial), so the saved
        # model's weights must have been trained under polynomial activations.
        # Saving from GELU phase and then evaluating at α=1.0 produces garbage
        # because the linear layer weights expect GELU outputs.
        if alpha >= 1.0 and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_epoch = ep
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

        history.append({
            "epoch": ep, "alpha": round(alpha, 3),
            "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 2),
            "val_acc": round(val_acc, 2), "val_auc": round(val_auc, 4),
            "lr": round(current_lr, 7),
        })

        # Phase indicator
        if alpha <= 0:
            phase = "GELU"
        elif alpha >= 1:
            phase = "POLY"
        else:
            phase = f"α={alpha:.2f}"

        print(f"  Epoch {ep:2d}/{args.epochs} [{phase:>7s}] | "
              f"Loss: {train_loss:.4f} | "
              f"Train: {train_acc:.1f}% | "
              f"Val: {val_acc:.1f}% | "
              f"AUC: {val_auc:.4f} | "
              f"LR: {current_lr:.2e}")

    train_time = time.time() - start_time

    # Also save the final epoch model (always in POLY phase)
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))

    # ── Final Test ──
    # Use best POLY-phase checkpoint if available, otherwise use final epoch
    best_ckpt = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(best_ckpt) and best_epoch > 0:
        model.load_state_dict(torch.load(best_ckpt, weights_only=True))
        print(f"\n  Loaded best POLY-phase checkpoint (epoch {best_epoch})")
    else:
        # No POLY-phase checkpoint was ever saved (shouldn't happen, but safety)
        print(f"\n  Using final epoch model (epoch {args.epochs})")
        best_epoch = args.epochs

    set_transition_alpha(model, 1.0)  # Ensure pure polynomial for test

    test_acc, test_auc, test_preds, test_labels = evaluate(
        model, test_loader, device, n_classes
    )

    # Compute deltas vs baseline
    acc_delta = test_acc - baseline_acc
    auc_delta = test_auc - baseline_auc

    # Extract final learned coefficients
    final_coeffs = get_poly_coefficients(model)

    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │  RESULTS: Poly-GELU DeiT-Tiny (warm-start)             │")
    print(f"  │  Dataset: {dataset_name.upper():<15s}                          │")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    print(f"  │  Step 1 Baseline:  {baseline_acc:6.2f}%  (AUC: {baseline_auc:.4f})      │")
    print(f"  │  Step 2v2 Result:  {test_acc:6.2f}%  (AUC: {test_auc:.4f})      │")
    print(f"  │  Δ Accuracy:       {acc_delta:+6.2f}%                           │")
    print(f"  │  Δ AUC:            {auc_delta:+6.4f}                           │")
    print(f"  │  Poly Degree:      {args.poly_degree}                                │")
    print(f"  │  Best Epoch:       {best_epoch:3d}                                  │")
    print(f"  │  Train Time:       {train_time:.1f}s                           │")
    print(f"  └─────────────────────────────────────────────────────────┘")

    # Print learned polynomial coefficients per block
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
        "step": "2v2",
        "description": f"GELU→degree-{args.poly_degree} polynomial (warm-start from Step 1)",
        "dataset": dataset_name,
        "test_accuracy": round(test_acc, 2),
        "test_auc": round(test_auc, 4),
        "baseline_accuracy": round(baseline_acc, 2),
        "baseline_auc": round(baseline_auc, 4),
        "accuracy_delta": round(acc_delta, 2),
        "auc_delta": round(auc_delta, 4),
        "checkpoint_verification_acc": round(verify_acc, 2),
        "poly_degree": args.poly_degree,
        "init_method": args.init_method,
        "transition_start": args.transition_start,
        "transition_end": args.transition_end,
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

    print(f"\n  ╔════════════════════════════════════════════════════════════╗")
    print(f"  ║  Step 2v2: GELU → Polynomial (Warm-Start from Step 1)     ║")
    print(f"  ║  Degree: {args.poly_degree}  |  Init: {args.init_method:10s}  |  LR: {args.lr:.0e}     ║")
    print(f"  ║  Transition: epochs {args.transition_start:2d} → {args.transition_end:2d}                          ║")
    print(f"  ╚════════════════════════════════════════════════════════════╝")

    # Verify baseline directory exists
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
    print(f"\n{'='*80}")
    print(f"  SUMMARY: Step 2v2 — Polynomial GELU (degree-{args.poly_degree}, warm-start) vs Baseline")
    print(f"{'='*80}")
    print(f"  {'Dataset':<14} {'Baseline':>10} {'Poly-GELU':>10} {'Δ Acc':>8}"
          f"  {'Base AUC':>10} {'Poly AUC':>10} {'Δ AUC':>8}")
    print(f"  {'─'*74}")

    deltas = []
    for ds, res in all_results.items():
        ba = res["baseline_accuracy"]
        pa = res["test_accuracy"]
        da = res["accuracy_delta"]
        bauc = res["baseline_auc"]
        pauc = res["test_auc"]
        dauc = res["auc_delta"]
        deltas.append(da)
        print(f"  {ds:<14} {ba:>9.2f}% {pa:>9.2f}% {da:>+7.2f}%"
              f"  {bauc:>10.4f} {pauc:>9.4f} {dauc:>+7.4f}")

    print(f"  {'─'*74}")
    avg_delta = np.mean(deltas)
    print(f"\n  Average accuracy change: {avg_delta:+.2f}%")

    if avg_delta > -5:
        print(f"  ✓ Within expected range (Baruch et al.: -0.3% to -5.3%)")
    elif avg_delta > -10:
        print(f"  ~ Moderate degradation — KD in Step 5 should recover most of this")
    else:
        print(f"  ⚠ Large degradation — consider degree-4 polynomial or more epochs")

    # v1 comparison reminder
    print(f"\n  Comparison with Step 2 v1 (cold-start):")
    print(f"  v1 collapsed on small datasets (Retina: -36%, Breast: -14%, Derma: -14%)")
    print(f"  v2 warm-start should show significantly less degradation on these datasets")

    print(f"\n  Next: Step 3 (replace softmax) or Step 5 (apply KD to recover accuracy)")

    # Save summary
    summary = {ds: {
        "accuracy": r["test_accuracy"], "auc": r["test_auc"],
        "accuracy_delta": r["accuracy_delta"], "auc_delta": r["auc_delta"],
        "baseline_accuracy": r["baseline_accuracy"],
    } for ds, r in all_results.items()}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()