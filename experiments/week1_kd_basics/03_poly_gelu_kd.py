"""
Week 1, Experiment 3: Polynomial GELU vs standard GELU, with and without KD.

Replaces GELU in DeiT-Tiny with PolyGELU (degree-4 approximation) and compares
4 variants:
  1. Teacher (ViT-Base, standard GELU)
  2. Student (DeiT-Tiny, standard GELU, no KD)
  3. Poly-Student (DeiT-Tiny, PolyGELU, no KD)
  4. Poly-Student + KD (DeiT-Tiny, PolyGELU, KD from ViT-Base)

Answers: Does KD close the accuracy gap introduced by replacing GELU with PolyGELU?

Usage:
    python experiments/week1_kd_basics/03_poly_gelu_kd.py

Outputs:
    results/week1/poly_gelu_comparison.json   — 4-row accuracy table
"""

import json
import os
import sys
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    import timm
    from medmnist import RetinaMNIST
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install timm medmnist")
    sys.exit(1)

try:
    from papers.bolt.boba_vit import PolyGELU
    from utils.poly_approx import approx_gelu_degree4
except ImportError:
    # Inline fallback
    class PolyGELU(nn.Module):
        """Degree-4 polynomial approximation of GELU: 0.125x⁴ - 0.25x² + 0.5x + 0.25."""
        def forward(self, x):
            return 0.125 * x.pow(4) - 0.25 * x.pow(2) + 0.5 * x + 0.25


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
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=2),
        DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2),
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


def replace_gelu_with_poly(model: nn.Module) -> nn.Module:
    """Replace all nn.GELU instances with PolyGELU in-place."""
    for name, module in model.named_children():
        if isinstance(module, nn.GELU):
            setattr(model, name, PolyGELU())
        else:
            replace_gelu_with_poly(module)
    return model


def train(model, train_loader, val_loader, epochs, device, teacher=None,
          alpha=0.5, temperature=4.0):
    """Train model. If teacher is provided, use soft KD loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")

    if teacher is not None:
        teacher.eval()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.squeeze(1).long().to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss_ce = ce(logits, y)
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(x)
                T = temperature
                s_log = torch.log_softmax(logits / T, dim=-1)
                t_soft = torch.softmax(t_logits / T, dim=-1)
                loss_kd = kl(s_log, t_soft) * (T ** 2)
                loss = alpha * loss_ce + (1 - alpha) * loss_kd
            else:
                loss = loss_ce
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        scheduler.step()
        val_acc = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "val_acc": round(val_acc, 3)})
        print(f"  epoch {epoch:2d}  val_acc={val_acc:.2f}%")
    return history


def main():
    print(f"Device: {DEVICE}")
    train_loader, val_loader, test_loader = get_retina_loaders()

    # Teacher: ViT-Base (reference accuracy)
    print("\nLoading ViT-Base teacher...")
    teacher = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=NUM_CLASSES)
    teacher = teacher.to(DEVICE)
    teacher_acc = evaluate(teacher, test_loader, DEVICE)
    print(f"Teacher test accuracy (after loading, before fine-tune): {teacher_acc:.2f}%")

    results = {
        "teacher_vit_base": {"test_acc": round(teacher_acc, 3), "gelu": "standard", "kd": False},
    }

    # Variant 2: DeiT-Tiny, standard GELU, no KD
    print("\n--- Variant 2: DeiT-Tiny, standard GELU, no KD ---")
    student_std = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=NUM_CLASSES)
    student_std = student_std.to(DEVICE)
    hist2 = train(student_std, train_loader, val_loader, EPOCHS, DEVICE, teacher=None)
    acc2 = evaluate(student_std, test_loader, DEVICE)
    results["deit_tiny_std_gelu_no_kd"] = {
        "test_acc": round(acc2, 3), "gelu": "standard", "kd": False, "history": hist2,
    }
    print(f"  Test acc: {acc2:.2f}%")

    # Variant 3: DeiT-Tiny, PolyGELU, no KD
    print("\n--- Variant 3: DeiT-Tiny, PolyGELU, no KD ---")
    student_poly = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=NUM_CLASSES)
    student_poly = replace_gelu_with_poly(student_poly).to(DEVICE)
    hist3 = train(student_poly, train_loader, val_loader, EPOCHS, DEVICE, teacher=None)
    acc3 = evaluate(student_poly, test_loader, DEVICE)
    results["deit_tiny_poly_gelu_no_kd"] = {
        "test_acc": round(acc3, 3), "gelu": "poly_degree4", "kd": False, "history": hist3,
    }
    print(f"  Test acc: {acc3:.2f}%")

    # Variant 4: DeiT-Tiny, PolyGELU, with KD
    print("\n--- Variant 4: DeiT-Tiny, PolyGELU, with KD (τ=4) ---")
    student_poly_kd = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=NUM_CLASSES)
    student_poly_kd = replace_gelu_with_poly(student_poly_kd).to(DEVICE)
    hist4 = train(student_poly_kd, train_loader, val_loader, EPOCHS, DEVICE,
                  teacher=teacher, alpha=0.5, temperature=4.0)
    acc4 = evaluate(student_poly_kd, test_loader, DEVICE)
    results["deit_tiny_poly_gelu_kd"] = {
        "test_acc": round(acc4, 3), "gelu": "poly_degree4", "kd": True,
        "kd_temperature": 4.0, "history": hist4,
    }
    print(f"  Test acc: {acc4:.2f}%")

    # Summary table
    print("\n=== Summary ===")
    header = f"{'Variant':<35} {'GELU':<15} {'KD':<6} {'Test Acc'}"
    print(header)
    print("-" * len(header))
    for key, val in results.items():
        print(f"  {key:<33} {val['gelu']:<15} {str(val['kd']):<6} {val['test_acc']:.2f}%")

    print(f"\nGELU accuracy gap (std → poly): {acc2 - acc3:.2f}%")
    print(f"KD recovery (poly no-KD → poly+KD): {acc4 - acc3:.2f}%")

    out_path = os.path.join(RESULTS_DIR, "poly_gelu_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
