"""
Plaintext ViT baseline on CIFAR-10.

Establish accuracy targets before moving to FHE.
Compare:
1. Standard ViT-S (with softmax, GELU, LayerNorm)
2. BoBa-ViT (with linear attention, scalar norm, poly-GELU)

This answers: "How much accuracy do we lose from FHE-friendly modifications?"
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def get_cifar10_loaders(batch_size: int = 128):
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train = DataLoader(
        torchvision.datasets.CIFAR10("data/", train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=2,
    )
    test = DataLoader(
        torchvision.datasets.CIFAR10("data/", train=False, transform=transform_test),
        batch_size=256, shuffle=False,
    )
    return train, test


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100 * correct / total


def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    return total_loss / len(loader)


def run_comparison(epochs: int = 100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = get_cifar10_loaders()
    criterion = nn.CrossEntropyLoss()

    # Import both model types
    import timm
    from papers.bolt.boba_vit import BobaViT

    models = {
        "Standard ViT-S (timm)": timm.create_model(
            "vit_small_patch4_32", pretrained=False, num_classes=10
        ),
        "BoBa-ViT (BOLT-style)": BobaViT(
            dim=256, depth=6, n_heads=8, mlp_ratio=2.0
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining: {name}")
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc = 0
        for epoch in range(epochs):
            loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
            if (epoch + 1) % 20 == 0:
                acc = evaluate(model, test_loader, device)
                best_acc = max(best_acc, acc)
                print(f"  Epoch {epoch+1:3d} | Loss: {loss:.3f} | Acc: {acc:.2f}%")

        results[name] = best_acc
        torch.save(model.state_dict(), f"checkpoints/{name.replace(' ', '_').replace('(', '').replace(')', '')}.pt")

    print("\n" + "=" * 50)
    print("Baseline Comparison Results")
    print("=" * 50)
    for name, acc in results.items():
        print(f"  {name:<40} {acc:.2f}%")

    accuracy_gap = results.get("Standard ViT-S (timm)", 0) - results.get("BoBa-ViT (BOLT-style)", 0)
    print(f"\nAccuracy gap (FHE-friendly overhead): {accuracy_gap:.2f}%")


if __name__ == "__main__":
    run_comparison(epochs=100)
