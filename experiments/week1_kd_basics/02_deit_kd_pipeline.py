"""
Week 1, Experiment 2: DeiT-Tiny KD pipeline on RetinaMNIST.

Compares:
  A) DeiT-Tiny trained from scratch (no distillation)
  B) DeiT-Tiny with KD from ViT-Base teacher (soft labels, temperature sweep)
  C) DeiT-Tiny with hard KD (teacher argmax as label)

Uses DistillationLoss from papers/deit/implementation.py.

Usage:
    # Requires 01_vit_medical_baseline.py to have saved a checkpoint, OR
    # uses timm pretrained ViT-Base as teacher directly (--pretrained-teacher)
    python experiments/week1_kd_basics/02_deit_kd_pipeline.py

Outputs:
    results/week1/kd_vs_scratch.json   — accuracy numbers + training curves
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
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install timm medmnist tqdm")
    sys.exit(1)

try:
    from papers.deit.implementation import DistillationLoss, DeiTStudent
except ImportError:
    # Fallback: define minimal versions inline if paper module not available
    class DeiTStudent(nn.Module):
        """DeiT-Tiny with optional distillation token, built from timm."""
        def __init__(self, num_classes: int, pretrained: bool = False):
            super().__init__()
            self.model = timm.create_model(
                "deit_tiny_patch16_224", pretrained=pretrained, num_classes=num_classes
            )

        def forward(self, x):
            return self.model(x)

    class DistillationLoss(nn.Module):
        def __init__(self, alpha: float = 0.5, temperature: float = 4.0, hard: bool = False):
            super().__init__()
            self.alpha = alpha
            self.temperature = temperature
            self.hard = hard
            self.ce = nn.CrossEntropyLoss()
            self.kl = nn.KLDivLoss(reduction="batchmean")

        def forward(self, student_logits, teacher_logits, labels):
            loss_ce = self.ce(student_logits, labels)
            if self.hard:
                teacher_labels = teacher_logits.argmax(dim=-1)
                loss_distill = self.ce(student_logits, teacher_labels)
            else:
                T = self.temperature
                s = torch.log_softmax(student_logits / T, dim=-1)
                t = torch.softmax(teacher_logits / T, dim=-1)
                loss_distill = self.kl(s, t) * (T ** 2)
            return self.alpha * loss_ce + (1 - self.alpha) * loss_distill


RESULTS_DIR = "results/week1"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
EPOCHS = 20
LR = 3e-4
BATCH_SIZE = 64


def get_retina_loaders(batch_size=BATCH_SIZE):
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


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.squeeze(1).long().to(device)
            preds = model(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100 * correct / total


def train_with_kd(student, teacher, train_loader, val_loader, criterion, epochs, device):
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    teacher.eval()
    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.squeeze(1).long().to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(x)
            student_logits = student(x)
            loss = criterion(student_logits, teacher_logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        scheduler.step()
        val_acc = evaluate(student, val_loader, device)
        history.append({"epoch": epoch, "val_acc": round(val_acc, 3),
                         "train_loss": round(total_loss / len(train_loader.dataset), 4)})
        print(f"  epoch {epoch:2d}  val_acc={val_acc:.2f}%")
    return history


def train_from_scratch(student, train_loader, val_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.squeeze(1).long().to(device)
            optimizer.zero_grad()
            logits = student(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        scheduler.step()
        val_acc = evaluate(student, val_loader, device)
        history.append({"epoch": epoch, "val_acc": round(val_acc, 3),
                         "train_loss": round(total_loss / len(train_loader.dataset), 4)})
        print(f"  epoch {epoch:2d}  val_acc={val_acc:.2f}%")
    return history


def main():
    print(f"Device: {DEVICE}")
    train_loader, val_loader, test_loader = get_retina_loaders()

    # Load ViT-Base teacher (pretrained on ImageNet-21k, fine-tune head in eval mode)
    print("\nLoading ViT-Base teacher...")
    teacher = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=NUM_CLASSES)
    teacher = teacher.to(DEVICE)

    results = {}

    # --- Variant A: DeiT-Tiny from scratch ---
    print("\n--- Variant A: DeiT-Tiny from scratch ---")
    student_a = DeiTStudent(num_classes=NUM_CLASSES).to(DEVICE)
    hist_a = train_from_scratch(student_a, train_loader, val_loader, EPOCHS, DEVICE)
    test_acc_a = evaluate(student_a, test_loader, DEVICE)
    results["scratch"] = {"test_acc": round(test_acc_a, 3), "history": hist_a}
    print(f"  [Scratch] Test accuracy: {test_acc_a:.2f}%")

    # --- Variant B: DeiT-Tiny with soft KD, τ sweep ---
    for tau in [2, 4, 8]:
        print(f"\n--- Variant B: DeiT-Tiny + soft KD (τ={tau}) ---")
        student_b = DeiTStudent(num_classes=NUM_CLASSES).to(DEVICE)
        criterion_b = DistillationLoss(alpha=0.5, temperature=tau, hard=False)
        hist_b = train_with_kd(student_b, teacher, train_loader, val_loader,
                                criterion_b, EPOCHS, DEVICE)
        test_acc_b = evaluate(student_b, test_loader, DEVICE)
        results[f"soft_kd_tau{tau}"] = {"test_acc": round(test_acc_b, 3), "history": hist_b}
        print(f"  [Soft KD τ={tau}] Test accuracy: {test_acc_b:.2f}%")

    # --- Variant C: DeiT-Tiny with hard KD ---
    print("\n--- Variant C: DeiT-Tiny + hard KD ---")
    student_c = DeiTStudent(num_classes=NUM_CLASSES).to(DEVICE)
    criterion_c = DistillationLoss(alpha=0.5, temperature=1.0, hard=True)
    hist_c = train_with_kd(student_c, teacher, train_loader, val_loader,
                            criterion_c, EPOCHS, DEVICE)
    test_acc_c = evaluate(student_c, test_loader, DEVICE)
    results["hard_kd"] = {"test_acc": round(test_acc_c, 3), "history": hist_c}
    print(f"  [Hard KD] Test accuracy: {test_acc_c:.2f}%")

    # Summary
    print("\n=== Summary ===")
    for key, val in results.items():
        print(f"  {key:20s}  test_acc={val['test_acc']:.2f}%")

    out_path = os.path.join(RESULTS_DIR, "kd_vs_scratch.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
