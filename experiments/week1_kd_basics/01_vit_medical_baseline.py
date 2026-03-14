"""
Week 1, Experiment 1: ViT-Base fine-tuned on MedMNIST (RetinaMNIST).

Establishes the plaintext teacher accuracy before knowledge distillation.
Logs per-class accuracy, AUC, confusion matrix, and training curve.

Usage:
    pip install timm medmnist tqdm scikit-learn
    python experiments/week1_kd_basics/01_vit_medical_baseline.py

Outputs:
    results/week1/vit_retina_baseline.json  — accuracy, AUC, per-class stats
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    import timm
    import medmnist
    from medmnist import RetinaMNIST
    from tqdm import tqdm
    from sklearn.metrics import roc_auc_score, confusion_matrix
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install timm medmnist tqdm scikit-learn")
    sys.exit(1)


RESULTS_DIR = "results/week1"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5  # RetinaMNIST: 5-class ordinal regression
EPOCHS = 20
LR = 3e-4
BATCH_SIZE = 64


def get_retina_loaders(batch_size: int = BATCH_SIZE):
    """Load RetinaMNIST with ImageNet normalization for ViT."""
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_ds = RetinaMNIST(split="train", transform=transform, download=True, root="data/medmnist")
    val_ds   = RetinaMNIST(split="val",   transform=transform, download=True, root="data/medmnist")
    test_ds  = RetinaMNIST(split="test",  transform=transform, download=True, root="data/medmnist")
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2),
        DataLoader(val_ds,   batch_size=256,         shuffle=False, num_workers=2),
        DataLoader(test_ds,  batch_size=256,         shuffle=False, num_workers=2),
    )


def evaluate(model, loader, device, num_classes=NUM_CLASSES):
    """Return accuracy, AUC, per-class accuracy, and confusion matrix."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y.squeeze(1).long())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds  = logits.argmax(dim=-1)

    acc = (preds == labels).float().mean().item() * 100

    # AUC (one-vs-rest, macro)
    probs = torch.softmax(logits, dim=-1).numpy()
    try:
        auc = roc_auc_score(labels.numpy(), probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    # Per-class accuracy
    cm = confusion_matrix(labels.numpy(), preds.numpy(), labels=list(range(num_classes)))
    per_class_acc = [
        100 * cm[c, c] / cm[c].sum() if cm[c].sum() > 0 else float("nan")
        for c in range(num_classes)
    ]

    return {
        "accuracy": round(acc, 3),
        "auc": round(float(auc), 4),
        "per_class_accuracy": [round(a, 3) for a in per_class_acc],
        "confusion_matrix": cm.tolist(),
    }


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.squeeze(1).long().to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(-1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, 100 * correct / total


def main():
    print(f"Device: {DEVICE}")

    train_loader, val_loader, test_loader = get_retina_loaders()
    print(f"RetinaMNIST — train: {len(train_loader.dataset)}, "
          f"val: {len(val_loader.dataset)}, test: {len(test_loader.dataset)}")

    # Load ViT-Base pretrained on ImageNet-21k (the teacher)
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = []
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 3),
            "val_acc": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
            "time_s": round(time.time() - t0, 1),
        }
        history.append(row)
        print(f"[{epoch:2d}/{EPOCHS}] loss={train_loss:.4f} train_acc={train_acc:.1f}% "
              f"val_acc={val_metrics['accuracy']:.1f}% val_auc={val_metrics['auc']:.4f} "
              f"({row['time_s']}s)")

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Final test evaluation with best checkpoint
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, DEVICE)

    results = {
        "model": "vit_base_patch16_224",
        "dataset": "RetinaMNIST",
        "num_classes": NUM_CLASSES,
        "epochs": EPOCHS,
        "best_val_acc": best_val_acc,
        "test_metrics": test_metrics,
        "training_history": history,
    }

    out_path = os.path.join(RESULTS_DIR, "vit_retina_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTest accuracy: {test_metrics['accuracy']:.2f}%  AUC: {test_metrics['auc']:.4f}")
    print(f"Per-class accuracy: {test_metrics['per_class_accuracy']}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
