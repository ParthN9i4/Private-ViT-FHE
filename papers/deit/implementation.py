"""
DeiT: Data-efficient Image Transformers with Knowledge Distillation.

Implements:
  - DeiT-Tiny student with [cls] + [distill] tokens
  - Hard and soft distillation losses
  - Training loop on CIFAR-10

Paper: https://arxiv.org/abs/2012.12877
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


# ---------------------------------------------------------------------------
# Core DeiT Components
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 3, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 3, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        mlp_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DeiTStudent(nn.Module):
    """
    DeiT with distillation token.

    Two tokens are prepended:
      - [cls]: used for classification with standard CE loss
      - [distill]: used for knowledge distillation from teacher

    At inference only [cls] is used; [distill] is discarded.

    Default config: DeiT-Tiny adapted for CIFAR-10 (32x32, patch_size=4)
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        n_classes: int = 10,
        dim: int = 192,
        depth: int = 12,
        n_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        n_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.distill_token = nn.Parameter(torch.zeros(1, 1, dim))
        # +2 for [cls] and [distill]
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 2, dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        # Two heads: one for CE, one for distillation
        self.cls_head = nn.Linear(dim, n_classes)
        self.distill_head = nn.Linear(dim, n_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.distill_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * p * p)
        return x

    def forward(self, x: torch.Tensor):
        """
        Returns:
          At training: (cls_logits, distill_logits) — both needed for loss
          At inference: cls_logits only
        """
        B = x.shape[0]
        x = self.patch_embed(self.patchify(x))  # (B, N, dim)

        cls = repeat(self.cls_token, "1 1 d -> b 1 d", b=B)
        dist = repeat(self.distill_token, "1 1 d -> b 1 d", b=B)
        x = torch.cat([cls, dist, x], dim=1)  # (B, N+2, dim)
        x = self.dropout(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = self.cls_head(x[:, 0])
        distill_out = self.distill_head(x[:, 1])

        if self.training:
            return cls_out, distill_out
        # Average the two heads at inference (DeiT paper Section 4)
        return (cls_out + distill_out) / 2


# ---------------------------------------------------------------------------
# Distillation Loss
# ---------------------------------------------------------------------------

class DistillationLoss(nn.Module):
    """
    Combined classification + distillation loss.

    L = (1 - alpha) * CE(cls_out, labels) + alpha * distillation_term

    Hard distillation:
        distillation_term = CE(distill_out, teacher.argmax())

    Soft distillation:
        distillation_term = τ² * KL(distill_out/τ || teacher_out/τ)

    Args:
        alpha: Weight on distillation term (0 = no KD, 1 = only KD)
        tau: Temperature for soft distillation (ignored for hard)
        mode: "hard" or "soft"
    """

    def __init__(self, alpha: float = 0.5, tau: float = 3.0, mode: str = "hard"):
        super().__init__()
        assert mode in ("hard", "soft"), f"mode must be 'hard' or 'soft', got {mode}"
        self.alpha = alpha
        self.tau = tau
        self.mode = mode
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        cls_out: torch.Tensor,
        distill_out: torch.Tensor,
        labels: torch.Tensor,
        teacher_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cls_out: Student [cls] logits (B, n_classes)
            distill_out: Student [distill] logits (B, n_classes)
            labels: Ground-truth class indices (B,)
            teacher_out: Teacher logits (B, n_classes) — no grad
        """
        ce_loss = self.ce(cls_out, labels)

        if self.mode == "hard":
            teacher_labels = teacher_out.argmax(dim=-1)
            distill_loss = self.ce(distill_out, teacher_labels)
        else:
            # Soft: KL divergence at temperature τ
            student_log_prob = F.log_softmax(distill_out / self.tau, dim=-1)
            teacher_prob = F.softmax(teacher_out / self.tau, dim=-1)
            distill_loss = self.kl(student_log_prob, teacher_prob) * (self.tau ** 2)

        return (1 - self.alpha) * ce_loss + self.alpha * distill_loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_with_distillation(
    teacher: nn.Module,
    student: DeiTStudent,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 5e-4,
    alpha: float = 0.5,
    tau: float = 3.0,
    mode: str = "hard",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> DeiTStudent:
    """
    Train DeiT student on CIFAR-10 with distillation from a teacher.

    Args:
        teacher: Pre-trained teacher model (frozen)
        student: DeiTStudent to train
        epochs: Number of training epochs
        alpha: Distillation weight
        tau: Temperature for soft distillation
        mode: "hard" or "soft"

    Returns:
        Trained student model
    """
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    print(f"Training DeiT student on {device} | KD mode={mode} α={alpha} τ={tau}")

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
    train_loader = DataLoader(
        torchvision.datasets.CIFAR10("data/", train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=2,
    )
    test_loader = DataLoader(
        torchvision.datasets.CIFAR10("data/", train=False, download=True, transform=transform_test),
        batch_size=256, shuffle=False,
    )

    teacher = teacher.to(device).eval()
    student = student.to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    criterion = DistillationLoss(alpha=alpha, tau=tau, mode=mode)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch in range(epochs):
        student.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                teacher_logits = teacher(x)
            cls_out, distill_out = student(x)
            loss = criterion(cls_out, distill_out, y, teacher_logits)

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
                    preds = student(x).argmax(dim=-1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            acc = 100 * correct / total
            best_acc = max(best_acc, acc)
            print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.3f} | Acc: {acc:.2f}%")

    print(f"\nBest accuracy: {best_acc:.2f}%")
    return student


if __name__ == "__main__":
    import sys
    from papers.bolt.boba_vit import BobaViT

    # Use BoBa-ViT as teacher (load from checkpoint if available)
    teacher = BobaViT()
    ckpt = "checkpoints/boba_vit_cifar10.pt"
    try:
        teacher.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"Loaded teacher from {ckpt}")
    except FileNotFoundError:
        print(f"Checkpoint not found at {ckpt} — teacher will be random (demo only)")

    student = DeiTStudent(image_size=32, patch_size=4, n_classes=10)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"DeiT-Tiny student: {n_params:,} parameters")

    student = train_with_distillation(teacher, student, epochs=50, mode="hard")
    torch.save(student.state_dict(), "checkpoints/deit_tiny_kd_cifar10.pt")
