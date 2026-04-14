"""
Simple KD baseline: Standard ViT teacher → Polynomial ViT student
==================================================================
Step 1: Train a standard DeiT-Tiny on CIFAR-10 (teacher)
Step 2: Replace GELU/softmax/LayerNorm with polynomial equivalents (student)
Step 3: Distill teacher → student
Step 4: Compare accuracies

Usage: python simple_kd_baseline.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import timm
import time


# ── Data ─────────────────────────────────────────────────────────────
def get_cifar10(batch_size=128):
    t_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    t_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=t_train)
    test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=t_test)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True))


# ── Polynomial replacements ──────────────────────────────────────────

class PolyGELU(nn.Module):
    """ax² + bx + c replacing GELU. Trainable. 1 CKKS level."""
    def __init__(self):
        super().__init__()
        # Least-squares fit to GELU over [-3, 3]
        x = torch.linspace(-3, 3, 10000)
        y = F.gelu(x)
        X = torch.stack([x**2, x, torch.ones_like(x)], dim=1)
        c = torch.linalg.lstsq(X, y).solution
        self.a = nn.Parameter(c[0].clone())
        self.b = nn.Parameter(c[1].clone())
        self.c = nn.Parameter(c[2].clone())

    def forward(self, x):
        return self.a * x * x + self.b * x + self.c


class PolyAttn(nn.Module):
    """ax² + bx + c replacing softmax element-wise. 1 CKKS level.
    Following PolyTransformer (ICML 2024) — softmax eliminated entirely."""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.05))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.25))

    def forward(self, x):
        return self.a * x * x + self.b * x + self.c


# ── Polynomial DeiT-Tiny ────────────────────────────────────────────

class PolyAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.poly_attn = PolyAttn()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.poly_attn(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class PolyBlock(nn.Module):
    def __init__(self, dim, num_heads=3, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim)
        self.attn = PolyAttentionBlock(dim, num_heads)
        self.norm2 = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.act = PolyGELU()
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)

    def forward(self, x):
        # x: (B, N, C) — BatchNorm1d needs (B, C, N)
        x = x + self.attn(self.norm1(x.transpose(1, 2)).transpose(1, 2))
        h = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.fc2(self.act(self.fc1(h)))
        return x


class PolyDeiT(nn.Module):
    def __init__(self, num_classes=10, img_size=32, patch_size=4,
                 embed_dim=192, depth=6, num_heads=3):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([PolyBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.BatchNorm1d(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.head(x[:, 0])


# ── Training functions ───────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += model(imgs).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def train_model(model, train_loader, test_loader, device, epochs,
                teacher=None, kd_alpha=0.1, kd_temp=4.0, label=""):
    """
    Train a model. If teacher is provided, use KD loss.
    KD is simple: alpha * KL(teacher||student) + (1-alpha) * CE
    No T² scaling — just raw KL + CE weighted by alpha.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if teacher is not None:
        teacher = teacher.to(device)
        teacher.eval()

    best_acc = 0.0
    print(f"\n{'─'*60}")
    print(f"  Training: {label}")
    print(f"  KD: {'yes (alpha=' + str(kd_alpha) + ', T=' + str(kd_temp) + ')' if teacher else 'no'}")
    print(f"{'─'*60}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)

            ce_loss = F.cross_entropy(logits, labels)

            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(imgs)
                kd_loss = F.kl_div(
                    F.log_softmax(logits / kd_temp, dim=-1),
                    F.softmax(t_logits / kd_temp, dim=-1),
                    reduction='batchmean'
                )
                loss = kd_alpha * kd_loss + (1 - kd_alpha) * ce_loss
            else:
                loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  loss={total_loss/len(train_loader):.4f}"
                  f"  acc={acc:.2f}%  best={best_acc:.2f}%")

    print(f"  ► Final best: {best_acc:.2f}%")
    return model, best_acc


# ── Main ─────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    epochs = 100
    train_loader, test_loader = get_cifar10()

    # ── Step 1: Train standard teacher ───────────────────────────────
    teacher = timm.create_model('deit_tiny_patch16_224', pretrained=False,
                                num_classes=10, img_size=32, patch_size=4)
    teacher, teacher_acc = train_model(
        teacher, train_loader, test_loader, device, epochs,
        label="Standard DeiT-Tiny teacher (GELU + softmax + LayerNorm)")

    # ── Step 2: Train polynomial student WITH KD ─────────────────────
    student_kd = PolyDeiT(num_classes=10)
    student_kd, kd_acc = train_model(
        student_kd, train_loader, test_loader, device, epochs,
        teacher=teacher, kd_alpha=0.1, kd_temp=4.0,
        label="Polynomial student WITH KD (PolyGELU + PolyAttn + BatchNorm)")

    # ── Step 3: Train polynomial student WITHOUT KD ──────────────────
    student_no_kd = PolyDeiT(num_classes=10)
    student_no_kd, no_kd_acc = train_model(
        student_no_kd, train_loader, test_loader, device, epochs,
        label="Polynomial student WITHOUT KD (baseline)")

    # ── Results ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Teacher (standard):          {teacher_acc:.2f}%")
    print(f"  Student (poly) + KD:         {kd_acc:.2f}%")
    print(f"  Student (poly) no KD:        {no_kd_acc:.2f}%")
    print(f"  ─────────────────────────────────────")
    print(f"  KD recovery:                 {kd_acc - no_kd_acc:+.2f}%")
    print(f"  Gap from teacher:            {teacher_acc - kd_acc:.2f}%")
    print(f"{'='*60}")

    torch.save({
        'teacher_acc': teacher_acc,
        'kd_acc': kd_acc,
        'no_kd_acc': no_kd_acc,
        'teacher_state': teacher.state_dict(),
        'student_kd_state': student_kd.state_dict(),
    }, 'simple_kd_results.pth')
    print("Saved to simple_kd_results.pth")


if __name__ == '__main__':
    main()