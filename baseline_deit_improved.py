"""
Improved DeiT-Tiny Baseline on MedMNIST
========================================
Based on:
- DeiT (Touvron et al., ICML 2021): distillation token, ImageNet pre-training
- MedMNIST v2 (Yang et al., Scientific Data 2023): standardized medical benchmarks
- Standard fine-tuning practices for ViTs on small medical datasets:
  partial freezing, AdamW with weight decay, cosine LR, data augmentation

Runs on: RetinaMNIST, PneumoniaMNIST, BloodMNIST, DermaMNIST, BreastMNIST
Reports: Accuracy, AUC-ROC (standard for MedMNIST), per-class metrics
Saves: best model checkpoint + results summary

Usage:
    python baseline_deit_improved.py                    # all datasets
    python baseline_deit_improved.py --dataset retina   # single dataset
    python baseline_deit_improved.py --freeze_blocks 10 # freeze more layers
"""

import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import timm
from sklearn.metrics import roc_auc_score, classification_report

# ── MedMNIST imports ──
import medmnist
from medmnist import (
    RetinaMNIST, PneumoniaMNIST, BloodMNIST,
    DermaMNIST, BreastMNIST, PathMNIST
)

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
    parser = argparse.ArgumentParser(description="DeiT-Tiny baseline on MedMNIST")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=list(DATASET_CONFIG.keys()) + ["all"],
                        help="Which MedMNIST dataset to run (default: all)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="AdamW weight decay (default: 0.05, following DeiT)")
    parser.add_argument("--freeze_blocks", type=int, default=8,
                        help="Number of transformer blocks to freeze (0-11, default: 8)")
    parser.add_argument("--model", type=str, default="deit_tiny_patch16_224",
                        help="timm model name (default: deit_tiny_patch16_224)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# Data Loading with Augmentation
# ═══════════════════════════════════════════════════════════════
# Following standard medical imaging augmentation:
# - RandomHorizontalFlip: retinal/dermoscopy images have no canonical orientation
# - RandomVerticalFlip: same reason, especially for retinal images
# - RandomRotation: fundus images can be rotated
# - ColorJitter: accounts for varying imaging conditions across clinics
# - No aggressive crops: medical features can be anywhere in the image

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

    # MedMNIST uses PIL images, labels are numpy arrays of shape (N, 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"  Classes: {config['n_classes']} | Task: {config['task']}")

    return train_loader, val_loader, test_loader, config


# ═══════════════════════════════════════════════════════════════
# Model Setup with Partial Freezing
# ═══════════════════════════════════════════════════════════════
# Rationale for freezing (from transfer learning literature):
# - Early ViT blocks learn general features (edges, textures, shapes)
# - Later blocks learn task-specific features
# - With only ~1000 training images, fine-tuning all 5.5M params overfits
# - Freezing first 8/12 blocks: ~1.5M trainable params, sufficient for small datasets
# - This is analogous to HETAL's approach of using frozen backbone + trainable head

def create_model(model_name, num_classes, freeze_blocks, device):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    # Freeze patch embedding
    for param in model.patch_embed.parameters():
        param.requires_grad = False

    # Freeze cls_token and pos_embed
    if hasattr(model, "cls_token") and model.cls_token is not None:
        model.cls_token.requires_grad = False
    if hasattr(model, "pos_embed") and model.pos_embed is not None:
        model.pos_embed.requires_grad = False

    # Freeze first N transformer blocks
    total_blocks = len(model.blocks)
    freeze_blocks = min(freeze_blocks, total_blocks)
    for i in range(freeze_blocks):
        for param in model.blocks[i].parameters():
            param.requires_grad = False

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    frozen_params = total_params - trainable_params

    print(f"  Model: {model_name}")
    print(f"  Total: {total_params:.2f}M | Trainable: {trainable_params:.2f}M | Frozen: {frozen_params:.2f}M")
    print(f"  Blocks frozen: {freeze_blocks}/{total_blocks}")

    return model


# ═══════════════════════════════════════════════════════════════
# Architecture Introspection
# ═══════════════════════════════════════════════════════════════
# Count non-polynomial operations that would need replacement for FHE
# This directly connects the baseline experiment to the FHE research goal

def count_nonpoly_ops(model):
    """Count operations in DeiT that are NOT CKKS-compatible."""
    gelu_count = 0
    softmax_count = 0
    layernorm_count = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.GELU):
            gelu_count += 1
        elif isinstance(module, nn.Softmax):
            softmax_count += 1
        elif isinstance(module, nn.LayerNorm):
            layernorm_count += 1

    # Note: timm's Attention doesn't use nn.Softmax explicitly — it's in the
    # forward() method as F.softmax(). Count attention layers instead.
    attn_count = sum(1 for name, _ in model.named_modules() if "attn" in name and "drop" not in name and "proj" not in name and "qkv" not in name and "." not in name.split("attn")[-1])

    # More reliable: count Attention modules
    attn_modules = sum(1 for _, m in model.named_modules() if type(m).__name__ == "Attention")

    print(f"\n  ── FHE Compatibility Analysis ──")
    print(f"  GELU activations (nn.GELU):     {gelu_count}")
    print(f"  Softmax in attention (implicit): {attn_modules} (one per block)")
    print(f"  LayerNorm layers:                {layernorm_count}")
    print(f"  Total non-polynomial ops:        {gelu_count + attn_modules + layernorm_count}")
    print(f"  ─────────────────────────────────")
    print(f"  To make FHE-friendly, replace:")
    print(f"    GELU → degree-2 polynomial (ax² + bx + c)")
    print(f"    Softmax → MGF-softmax (depth 7-10) or Power-Softmax")
    print(f"    LayerNorm → absorbed into linear layers or poly approx")

    return {
        "gelu": gelu_count,
        "softmax_attn": attn_modules,
        "layernorm": layernorm_count,
        "total_nonpoly": gelu_count + attn_modules + layernorm_count,
    }


# ═══════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════
# Uses AdamW (weight-decoupled Adam) following DeiT's training recipe
# Cosine annealing LR schedule: standard for ViT fine-tuning
# Label smoothing: 0.1 (following DeiT, helps with small datasets)

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

        # Gradient clipping (prevents instability with small datasets)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    """Evaluate model and compute accuracy + AUC-ROC (MedMNIST standard)."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

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

    # AUC-ROC: standard MedMNIST metric
    # For binary: use probability of positive class
    # For multi-class: use one-vs-rest macro average
    try:
        if n_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            # One-hot encode labels for multi-class AUC
            from sklearn.preprocessing import label_binarize
            labels_onehot = label_binarize(all_labels, classes=list(range(n_classes)))
            auc = roc_auc_score(labels_onehot, all_probs, multi_class="ovr", average="macro")
    except Exception as e:
        print(f"  Warning: AUC computation failed ({e}), setting to 0")
        auc = 0.0

    return accuracy, auc, all_preds, all_labels


# ═══════════════════════════════════════════════════════════════
# Main Training Pipeline
# ═══════════════════════════════════════════════════════════════

def run_experiment(dataset_name, args, device):
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")

    # Data
    train_loader, val_loader, test_loader, config = get_dataloaders(
        dataset_name, args.batch_size
    )
    n_classes = config["n_classes"]

    # Model
    model = create_model(args.model, n_classes, args.freeze_blocks, device)

    # FHE compatibility analysis (run once)
    nonpoly_ops = count_nonpoly_ops(model)

    # Loss: CrossEntropy with label smoothing (DeiT standard)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer: AdamW (DeiT recipe)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Scheduler: Cosine annealing with warmup
    # 5 epochs warmup (standard for ViT fine-tuning)
    warmup_epochs = 5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6
    )

    # Training
    best_val_acc = 0.0
    best_val_auc = 0.0
    best_epoch = 0
    history = []

    save_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n  Training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(args.epochs):
        # Warmup: linear LR ramp for first 5 epochs
        if epoch < warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_auc, _, _ = evaluate(model, val_loader, device, n_classes)

        # Step scheduler after warmup
        if epoch >= warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # Save best model (by val AUC — more robust than accuracy for imbalanced medical data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_acc": round(val_acc, 2),
            "val_auc": round(val_auc, 4),
            "lr": round(current_lr, 7),
        })

        print(f"  Epoch {epoch+1:2d}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train: {train_acc:.1f}% | "
              f"Val: {val_acc:.1f}% | "
              f"AUC: {val_auc:.4f} | "
              f"LR: {current_lr:.2e}")

    train_time = time.time() - start_time
    print(f"\n  Training completed in {train_time:.1f}s")
    print(f"  Best val AUC: {best_val_auc:.4f} (epoch {best_epoch})")

    # ── Final Test Evaluation ──
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth"), weights_only=True))
    test_acc, test_auc, test_preds, test_labels = evaluate(model, test_loader, device, n_classes)

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  RESULTS: DeiT-Tiny on {dataset_name.upper():15s}  │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Test Accuracy:  {test_acc:6.2f}%               │")
    print(f"  │  Test AUC-ROC:   {test_auc:6.4f}                │")
    print(f"  │  Best Epoch:     {best_epoch:3d}                    │")
    print(f"  │  Frozen Blocks:  {args.freeze_blocks}/12                  │")
    print(f"  └─────────────────────────────────────────┘")

    # Per-class report
    target_names = [f"Class {i}" for i in range(n_classes)]
    print(f"\n  Per-class classification report:")
    print(classification_report(test_labels, test_preds, target_names=target_names, digits=3))

    # Save results
    results = {
        "dataset": dataset_name,
        "model": args.model,
        "test_accuracy": round(test_acc, 2),
        "test_auc": round(test_auc, 4),
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 2),
        "best_val_auc": round(best_val_auc, 4),
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "freeze_blocks": args.freeze_blocks,
        "batch_size": args.batch_size,
        "train_time_seconds": round(train_time, 1),
        "n_classes": n_classes,
        "nonpoly_ops": nonpoly_ops,
        "history": history,
    }

    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_path}")

    return results


# ═══════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which datasets to run
    if args.dataset == "all":
        datasets = list(DATASET_CONFIG.keys())
    else:
        datasets = [args.dataset]

    all_results = {}
    for ds in datasets:
        result = run_experiment(ds, args, device)
        all_results[ds] = result

    # ── Summary Table ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY: DeiT-Tiny Plaintext Baselines (Row 1 of Results Table)")
    print(f"{'='*70}")
    print(f"  {'Dataset':<18} {'Accuracy':>10} {'AUC-ROC':>10} {'Classes':>8} {'Epoch':>6}")
    print(f"  {'─'*54}")
    for ds, res in all_results.items():
        print(f"  {ds:<18} {res['test_accuracy']:>9.2f}% {res['test_auc']:>10.4f} {res['n_classes']:>8} {res['best_epoch']:>6}")
    print(f"  {'─'*54}")
    print(f"\n  Next steps:")
    print(f"  1. Replace GELU with polynomial → measure accuracy drop")
    print(f"  2. Replace softmax with MGF-softmax → measure accuracy drop")
    print(f"  3. Apply KD from this plaintext model (teacher) → recover accuracy")
    print(f"  4. Encrypt polynomial model with CKKS → measure ciphertext accuracy")

    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    summary = {ds: {"accuracy": r["test_accuracy"], "auc": r["test_auc"]}
               for ds, r in all_results.items()}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()