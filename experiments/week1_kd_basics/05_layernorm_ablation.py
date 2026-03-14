"""
Week 1, Experiment 5: LayerNorm variant ablation on DeiT-Tiny.

Trains DeiT-Tiny on RetinaMNIST with 3 LayerNorm variants, all with KD from ViT-Base:
  (a) Standard LayerNorm (baseline)
  (b) LinearNorm — affine-only (scale + shift), no mean/std division
  (c) No normalization

Answers: How much accuracy does standard LayerNorm contribute? Is LinearNorm
a viable FHE-friendly replacement?

Usage:
    python experiments/week1_kd_basics/05_layernorm_ablation.py

Outputs:
    results/week1/layernorm_ablation.json
"""

import json
import os
import sys

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
    from papers.poly_transformer.implementation import LinearNorm
except ImportError:
    class LinearNorm(nn.Module):
        """FHE-friendly LayerNorm replacement: learnable scale + shift only (no statistics)."""
        def __init__(self, normalized_shape: int):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias   = nn.Parameter(torch.zeros(normalized_shape))

        def forward(self, x):
            return x * self.weight + self.bias


class NoNorm(nn.Module):
    """Identity — no normalization at all."""
    def forward(self, x):
        return x


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


def replace_layernorm(model: nn.Module, replacement_cls, embed_dim: int = None) -> nn.Module:
    """Replace all LayerNorm instances with replacement_cls in-place."""
    for name, module in model.named_children():
        if isinstance(module, nn.LayerNorm):
            if replacement_cls is NoNorm:
                setattr(model, name, NoNorm())
            elif replacement_cls is LinearNorm:
                # Use the same normalized_shape as the original
                dim = module.normalized_shape[0] if module.normalized_shape else embed_dim
                setattr(model, name, LinearNorm(dim))
        else:
            replace_layernorm(module, replacement_cls, embed_dim)
    return model


def train_with_kd(model, teacher, train_loader, val_loader, epochs, device,
                  alpha=0.5, temperature=4.0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")
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
            with torch.no_grad():
                t_logits = teacher(x)
            T = temperature
            loss_kd = kl(
                torch.log_softmax(logits / T, dim=-1),
                torch.softmax(t_logits / T, dim=-1)
            ) * (T ** 2)
            loss = alpha * loss_ce + (1 - alpha) * loss_kd
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

    print("\nLoading ViT-Base teacher...")
    teacher = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=NUM_CLASSES)
    teacher = teacher.to(DEVICE)

    norm_variants = {
        "standard_layernorm": None,           # default DeiT-Tiny
        "linear_norm":        LinearNorm,     # FHE-friendly replacement
        "no_norm":            NoNorm,         # ablation: remove entirely
    }

    results = {}
    for norm_name, norm_cls in norm_variants.items():
        print(f"\n--- {norm_name} ---")
        model = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=NUM_CLASSES)
        if norm_cls is not None:
            model = replace_layernorm(model, norm_cls)
        model = model.to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

        try:
            hist = train_with_kd(model, teacher, train_loader, val_loader, EPOCHS, DEVICE)
            test_acc = evaluate(model, test_loader, DEVICE)
            print(f"  Test accuracy: {test_acc:.2f}%")
            results[norm_name] = {
                "test_acc": round(test_acc, 3),
                "fhe_friendly": norm_cls is not None,
                "history": hist,
            }
        except Exception as ex:
            print(f"  FAILED: {ex}")
            results[norm_name] = {"test_acc": None, "error": str(ex)}

    print("\n=== Summary ===")
    print(f"{'Variant':<25} {'FHE-Friendly':<14} {'Test Acc'}")
    print("-" * 50)
    for name, val in results.items():
        fhe = val.get("fhe_friendly", False)
        acc = f"{val['test_acc']:.2f}%" if val["test_acc"] is not None else "FAILED"
        print(f"  {name:<23} {str(fhe):<14} {acc}")

    out_path = os.path.join(RESULTS_DIR, "layernorm_ablation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
