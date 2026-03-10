"""
AutoFHE: Per-layer polynomial activation search + polynomial-aware KD training.

Implements:
  - EvoReLU: learnable polynomial activation per layer
  - MixedDegreeViT: ViT with different polynomial degrees per block
  - polynomial_aware_training(): ReLU teacher → polynomial student via KD
  - greedy_degree_search(): find the minimal degree per layer that keeps accuracy

Reference: AutoFHE — Automated Adaption of CNNs for Efficient Evaluation over FHE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math


# ---------------------------------------------------------------------------
# EvoReLU — Learnable Polynomial Activation
# ---------------------------------------------------------------------------

class EvoReLU(nn.Module):
    """
    Parameterized polynomial activation with learnable coefficients.

    f(x) = c_0 + c_1*x + c_2*x^2 + ... + c_d*x^d

    Initialized to approximate GELU at the given degree.

    FHE depth cost: ceil(log2(degree))
      degree=2 → 1 level
      degree=4 → 2 levels
      degree=8 → 3 levels

    Args:
        degree: Polynomial degree
        init: 'gelu' (approximate GELU) or 'random'
    """

    # Default GELU approximation coefficients by degree
    _GELU_INIT = {
        2: [0.25, 0.5, 0.125],                              # c0 + c1*x + c2*x^2
        4: [0.0, 0.5, 0.0, 0.1972, 0.0012],                # degree-4 BOLT coeffs
        8: [0.0, 0.5, 0.0, 0.1972, 0.0, -0.001, 0.0, 0.0, 0.0],  # extended
    }

    def __init__(self, degree: int = 4, init: str = "gelu"):
        super().__init__()
        assert degree >= 1
        self.degree = degree
        n_coeffs = degree + 1

        if init == "gelu" and degree in self._GELU_INIT:
            init_vals = self._GELU_INIT[degree][:n_coeffs]
            # Pad if needed
            init_vals += [0.0] * (n_coeffs - len(init_vals))
            self.coeffs = nn.Parameter(torch.tensor(init_vals, dtype=torch.float32))
        else:
            self.coeffs = nn.Parameter(torch.randn(n_coeffs) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate polynomial using Horner's method for numerical stability."""
        result = self.coeffs[-1]
        for c in reversed(self.coeffs[:-1]):
            result = result * x + c
        return result

    def fhe_depth(self) -> int:
        """Multiplicative depth cost in FHE."""
        return math.ceil(math.log2(self.degree)) if self.degree > 1 else 0

    def extra_repr(self) -> str:
        return f"degree={self.degree}, fhe_depth={self.fhe_depth()}"


# ---------------------------------------------------------------------------
# Mixed-Degree ViT
# ---------------------------------------------------------------------------

class MixedDegreeBlock(nn.Module):
    """
    Transformer block with a configurable polynomial activation degree.

    Uses ScalarNorm (free in FHE) and LinearAttention (no softmax).
    """

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float, poly_degree: int):
        super().__init__()
        from papers.bolt.boba_vit import ScalarNorm, LinearAttention
        mlp_dim = int(dim * mlp_ratio)

        self.norm1 = ScalarNorm(dim)
        self.attn = LinearAttention(dim, n_heads)
        self.norm2 = ScalarNorm(dim)
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.act = EvoReLU(degree=poly_degree)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.poly_degree = poly_degree

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.fc2(self.act(self.fc1(self.norm2(x))))
        return x

    def fhe_block_depth(self) -> int:
        """Total FHE depth: attention levels + activation levels."""
        attn_depth = 4  # QKV proj + QK.T + output proj
        act_depth = self.act.fhe_depth()
        ffn_depth = 2 + act_depth  # FC1 + act + FC2 (FC counted as 1 each matmul)
        return attn_depth + ffn_depth


class MixedDegreeViT(nn.Module):
    """
    ViT with per-block polynomial degree assignment.

    Args:
        degree_schedule: List of polynomial degrees, one per block.
                         e.g. [4, 4, 4, 2, 2, 2] for 6 blocks
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        n_classes: int = 10,
        dim: int = 256,
        n_heads: int = 8,
        mlp_ratio: float = 2.0,
        degree_schedule: Optional[List[int]] = None,
    ):
        super().__init__()
        if degree_schedule is None:
            degree_schedule = [4, 4, 4, 2, 2, 2]

        n_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        self.patch_size = patch_size

        self.patch_embed = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))

        self.blocks = nn.ModuleList([
            MixedDegreeBlock(dim, n_heads, mlp_ratio, d)
            for d in degree_schedule
        ])

        from papers.bolt.boba_vit import ScalarNorm
        self.norm = ScalarNorm(dim)
        self.head = nn.Linear(dim, n_classes)
        self.degree_schedule = degree_schedule
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        return x.reshape(B, C, H//p, p, W//p, p).permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C*p*p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from einops import repeat
        B = x.shape[0]
        x = self.patch_embed(self.patchify(x))
        cls = repeat(self.cls_token, "1 1 d -> b 1 d", b=B)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 0]))

    def total_fhe_depth(self) -> int:
        """Sum of FHE depths across all blocks plus patch embed and head."""
        block_depth = sum(b.fhe_block_depth() for b in self.blocks)
        return 1 + block_depth + 1  # patch_embed + blocks + head

    def depth_report(self) -> None:
        """Print per-layer depth breakdown."""
        print(f"{'Block':<8} {'Degree':<8} {'Depth':<8}")
        print("-" * 24)
        total = 1
        print(f"{'embed':<8} {'-':<8} {1:<8}")
        for i, block in enumerate(self.blocks):
            d = block.fhe_block_depth()
            total += d
            print(f"{i:<8} {block.poly_degree:<8} {d:<8}")
        print(f"{'head':<8} {'-':<8} {1:<8}")
        total += 1
        print(f"\nTotal FHE depth: {total}")


# ---------------------------------------------------------------------------
# Greedy Degree Search
# ---------------------------------------------------------------------------

def greedy_degree_search(
    model_fn,
    train_loader,
    test_loader,
    n_blocks: int = 6,
    accuracy_threshold: float = 85.0,
    candidate_degrees: List[int] = None,
    epochs_per_trial: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[int]:
    """
    Greedy search for minimum-depth degree schedule.

    Strategy:
      1. Start from all degree-4 (good accuracy baseline)
      2. For each block in order (last → first), try reducing degree
      3. If accuracy stays >= threshold after short fine-tuning, keep reduction

    Args:
        model_fn: Callable taking degree_schedule -> MixedDegreeViT
        accuracy_threshold: Minimum acceptable test accuracy (%)
        candidate_degrees: Degrees to try per block (default [4, 2])

    Returns:
        Optimal degree_schedule list
    """
    if candidate_degrees is None:
        candidate_degrees = [4, 2]

    schedule = [max(candidate_degrees)] * n_blocks
    print(f"Starting greedy search from all-degree-{max(candidate_degrees)}")

    for block_idx in reversed(range(n_blocks)):
        for degree in sorted(candidate_degrees):
            if degree >= schedule[block_idx]:
                continue  # only try reduction

            trial_schedule = schedule.copy()
            trial_schedule[block_idx] = degree
            model = model_fn(trial_schedule).to(device)

            acc = _quick_eval(model, train_loader, test_loader, epochs_per_trial, device)
            depth = MixedDegreeViT(degree_schedule=trial_schedule).total_fhe_depth()

            print(f"  Block {block_idx}: degree {schedule[block_idx]}→{degree} | "
                  f"acc={acc:.1f}% | total_depth={depth}")

            if acc >= accuracy_threshold:
                schedule[block_idx] = degree
                print(f"  ✓ Accepted")
                break
            else:
                print(f"  ✗ Rejected (acc below {accuracy_threshold}%)")

    print(f"\nFinal schedule: {schedule}")
    print(f"Total FHE depth: {MixedDegreeViT(degree_schedule=schedule).total_fhe_depth()}")
    return schedule


def _quick_eval(model, train_loader, test_loader, epochs, device):
    """Short training run for degree search trials."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(-1) == y).sum().item()
            total += y.size(0)
    return 100 * correct / total


# ---------------------------------------------------------------------------
# Polynomial-Aware Training (KD from ReLU teacher)
# ---------------------------------------------------------------------------

def polynomial_aware_training(
    teacher: nn.Module,
    degree_schedule: List[int],
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 3e-4,
    tau: float = 3.0,
    alpha: float = 0.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MixedDegreeViT:
    """
    Train a polynomial student using KD from a ReLU teacher.

    Args:
        teacher: Trained ReLU model (e.g. BoBa-ViT or standard ViT)
        degree_schedule: Per-block polynomial degrees for student
        tau: KD temperature
        alpha: Weight on distillation loss

    Returns:
        Trained MixedDegreeViT
    """
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = T.Compose([
        T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_loader = DataLoader(
        torchvision.datasets.CIFAR10("data/", train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=2,
    )
    test_loader = DataLoader(
        torchvision.datasets.CIFAR10("data/", train=False, download=True, transform=transform_test),
        batch_size=256, shuffle=False,
    )

    student = MixedDegreeViT(degree_schedule=degree_schedule).to(device)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")

    print(f"Polynomial-aware KD training | schedule={degree_schedule}")
    student.depth_report()

    best_acc = 0.0
    for epoch in range(epochs):
        student.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)

            ce_loss = ce(s_logits, y)
            kl_loss = kl(
                F.log_softmax(s_logits / tau, dim=-1),
                F.softmax(t_logits / tau, dim=-1),
            ) * (tau ** 2)
            loss = (1 - alpha) * ce_loss + alpha * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            student.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    correct += (student(x).argmax(-1) == y).sum().item()
                    total += y.size(0)
            acc = 100 * correct / total
            best_acc = max(best_acc, acc)
            print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.3f} | Acc: {acc:.2f}%")

    print(f"\nBest accuracy: {best_acc:.2f}%")
    return student


if __name__ == "__main__":
    # Demo: mixed-degree ViT with [4, 4, 4, 2, 2, 2] schedule
    schedule = [4, 4, 4, 2, 2, 2]
    model = MixedDegreeViT(degree_schedule=schedule)
    model.depth_report()
    print(f"\nParams: {sum(p.numel() for p in model.parameters()):,}")

    # Quick forward pass check
    x = torch.randn(2, 3, 32, 32)
    print(f"Output shape: {model(x).shape}")
