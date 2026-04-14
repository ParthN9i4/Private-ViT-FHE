"""
BlindFed-Style Offline Knowledge Distillation for FHE-Compatible Vision Transformers
=====================================================================================

This script implements the core idea from:
  "A Framework for Double-Blind Federated Adaptation of Foundation Models"
  (Tastan & Nandakumar, ICCV 2025)

WHAT THIS DOES:
  1. Takes a standard DeiT-Tiny teacher (with GELU, softmax, LayerNorm)
  2. Creates a polynomial student (GELU→quadratic, softmax→poly, LayerNorm→BatchNorm)
  3. Trains the student via offline KD to mimic the teacher
  4. Compares accuracy: teacher vs student vs student-without-KD

WHY POLYNOMIAL ACTIVATIONS?
  CKKS homomorphic encryption can ONLY do additions and multiplications.
  
  GELU(x) = x · Φ(x) uses the Gaussian CDF Φ — not expressible as finite
  polynomial. But ax² + bx + c IS just multiplications and additions.
  
  Over a bounded range [-3, 3], a well-chosen quadratic closely mimics GELU.
  The accuracy loss from this approximation is recovered via KD.

ARCHITECTURE:
  Teacher: timm's DeiT-Tiny (standard GELU + softmax + LayerNorm)
  Student: Same architecture but with:
    - PolyGELU:    ax² + bx + c  (trainable coefficients, 1 CKKS level)
    - PolyAttnAct: ax² + bx + c  replacing softmax entirely (1 CKKS level)
                   Following PolyTransformer (ICML 2024) scaled-sigmoid approach
    - BatchNorm replacing LayerNorm (fixed stats at inference, ~1 CKKS level)

USAGE:
  python blindfed_offline_kd.py --dataset cifar10 --epochs 50

  For your research, change --dataset to bloodmnist or aptos2019.

Author: Implementation guide for Panchan's PhD research
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import timm
import argparse
import time
import math
from collections import OrderedDict


# =============================================================================
# PART 1: POLYNOMIAL ACTIVATION FUNCTIONS
# =============================================================================
# 
# THE KEY INSIGHT:
# Under CKKS encryption, every operation must decompose into additions and
# multiplications on encrypted numbers. Each multiplication consumes one
# "level" of the encryption budget. When levels run out, you need expensive
# "bootstrapping" to continue. So we want LOW-DEGREE polynomials.
#
# GELU(x) = x · Φ(x)  where Φ is Gaussian CDF
#   - Cannot be computed under CKKS (Φ involves exp, erf, sqrt)
#   - Solution: replace with ax² + bx + c
#   - Cost: 1 multiplication (x·x) + 1 multiplication (a·x²) = 1 CKKS level
#
# The coefficients a, b, c can be:
#   (a) Fixed: BlindFed uses a=0.125, b=0.25, c=0.5
#   (b) Trainable: let backprop find optimal coefficients per layer
#   (c) From CaPriDe: truncate degree-4 fit to degree-2
#
# We implement TRAINABLE coefficients initialized to BlindFed's values.
# This gives the optimizer freedom to adapt per-layer while starting
# from a known-good approximation.
# =============================================================================

class PolyGELU(nn.Module):
    """
    Polynomial approximation of GELU: f(x) = a·x² + b·x + c
    
    IMPORTANT: GELU(0) = 0 exactly. Any polynomial approximation with c != 0
    has a constant offset at the origin. This is a known limitation of
    degree-2 approximations — they cannot simultaneously match GELU's
    value at 0, its slope at 0, and its curvature.
    
    TWO INITIALIZATION OPTIONS:
    
    1. "blindfed": a=0.125, b=0.25, c=0.5 (from BlindFed, ICCV 2025)
       - Large offset at origin: PolyGELU(0) = 0.5 vs GELU(0) = 0.0
       - Decent fit for positive x, poor for negative x
       - Used in published work, so it's a valid baseline
    
    2. "lsq" (default): Least-squares fit of GELU over [-3, 3]
       - Computed by fitting ax²+bx+c to minimize ∫(GELU(x) - poly(x))² dx
       - Much smaller offset at origin
       - Better overall approximation quality
    
    Both sets are TRAINABLE — the optimizer adjusts them during KD training.
    The initialization matters for convergence speed, not final accuracy.
    
    CKKS COST: 1 multiplicative level (the x² term)
    """
    def __init__(self, init='lsq'):
        super().__init__()
        if init == 'blindfed':
            # BlindFed (ICCV 2025) — Quad(x) = 0.125x² + 0.25x + 0.5
            # Note: PolyGELU(0) = 0.5, but GELU(0) = 0.0
            self.a = nn.Parameter(torch.tensor(0.125))
            self.b = nn.Parameter(torch.tensor(0.25))
            self.c = nn.Parameter(torch.tensor(0.5))
        elif init == 'lsq':
            # Least-squares fit of GELU over [-3, 3]
            # Computed offline: minimize Σ(GELU(x_i) - (ax²+bx+c))²
            # for 10,000 uniformly sampled points in [-3, 3]
            self._compute_lsq_init()
        else:
            raise ValueError(f"Unknown init: {init}. Use 'blindfed' or 'lsq'.")
    
    def _compute_lsq_init(self):
        """Compute least-squares degree-2 polynomial fit to GELU over [-3, 3]."""
        # Sample GELU values densely
        x = torch.linspace(-3, 3, 10000)
        y = F.gelu(x)
        # Solve normal equations: [x⁴ x³ x²; x³ x² x; x² x 1] [a;b;c] = [Σx²y; Σxy; Σy]
        X = torch.stack([x**2, x, torch.ones_like(x)], dim=1)
        # Least squares: (X^T X)^{-1} X^T y
        coeffs = torch.linalg.lstsq(X, y).solution
        self.a = nn.Parameter(coeffs[0].clone())
        self.b = nn.Parameter(coeffs[1].clone())
        self.c = nn.Parameter(coeffs[2].clone())
    
    def forward(self, x):
        # Under CKKS:
        #   1. ct_x2 = HMul(ct_x, ct_x)      [1 level consumed]
        #   2. ct_ax2 = PMul(a, ct_x2)         [plaintext multiply, free]
        #   3. ct_bx = PMul(b, ct_x)           [plaintext multiply, free]
        #   4. result = HAdd(ct_ax2, ct_bx, c)  [additions are free]
        # Total: 1 CKKS level
        return self.a * x * x + self.b * x + self.c


class PolyAttentionActivation(nn.Module):
    """
    Polynomial attention activation REPLACING softmax entirely.
    
    Following PolyTransformer (Zimerman et al., ICML 2024):
    Instead of softmax(QK^T/√d), apply a polynomial element-wise to each
    attention score independently. This eliminates softmax COMPLETELY —
    no max subtraction, no exponentiation, no division.
    
    WHY THIS IS BETTER THAN APPROXIMATING SOFTMAX WITH POLYNOMIALS:
    
    Approaches like (1+x/n)^n approximate exp(x), but softmax needs
    exp + max subtraction + division. The full pipeline costs ~38 CKKS levels:
      - Max subtraction costs ~15 CKKS levels (composite sign polynomial)
      - Division costs ~20 CKKS levels (Goldschmidt iteration)
      - Total: ~38 levels, exceeding the entire CKKS budget
    
    Scaled polynomial attention avoids ALL of this:
      - Each score a_{ij} is mapped through f(a_{ij}) = polynomial(a_{ij})
      - No interaction between scores (no max, no sum, no division)
      - Each element processed independently → naturally SIMD-parallel
      - Cost: just the polynomial degree in CKKS levels
    
    The polynomial f(x) = a·x² + b·x + c applied to attention scores
    acts as a "soft gating" function: positive scores get amplified,
    negative scores get suppressed. KD trains the coefficients so that
    the student's attention patterns match the teacher's behavior.
    
    CKKS COST: 1 multiplicative level (degree-2 polynomial)
    vs. ~38 levels for proper polynomial softmax
    
    This is the approach used by:
    - PolyTransformer (Zimerman et al., ICML 2024) — scaled-sigmoid
    - BlindFed (Tastan & Nandakumar, ICCV 2025) — "ASoftmax"
    - Powerformer (Park et al., ACL 2025) — BRP-max
    """
    def __init__(self):
        super().__init__()
        # Trainable polynomial applied element-wise to attention scores
        # Initialized to approximate a sigmoid-like gating:
        #   f(x) ≈ 0 for very negative x, f(x) ≈ x for positive x
        # The optimizer will adjust these during KD
        self.a = nn.Parameter(torch.tensor(0.05))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.25))
    
    def forward(self, x):
        # Element-wise polynomial: no cross-element operations
        # Under CKKS: 1 HMul (x*x) + 2 PMul (a*x², b*x) + adds = 1 level
        return self.a * x * x + self.b * x + self.c


# =============================================================================
# PART 2: POLYNOMIAL VISION TRANSFORMER
# =============================================================================
#
# We modify a standard ViT by replacing three types of operations:
#
# 1. GELU → PolyGELU (in the FFN block)
#    Standard: FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
#    Ours:     FFN(x) = PolyGELU(xW₁ + b₁)W₂ + b₂
#
# 2. Softmax → PolyAttentionActivation (in the attention block)
#    Standard: Attn = softmax(QKᵀ / √d) · V
#    Ours:     Attn = poly(QKᵀ / √d) · V  (element-wise, no max/division)
#    Following PolyTransformer (ICML 2024): softmax eliminated entirely.
#
# 3. LayerNorm → BatchNorm1d (following PolyTransformer, ICML 2024)
#    Standard: LayerNorm computes mean/var per-token, divides by √var
#    BatchNorm: uses FIXED running statistics from training, so at inference
#    it's just: (x - μ_running) * γ / √(σ²_running + ε) + β
#    This is a plaintext-ciphertext affine transform = ~1 CKKS level
#
# Total per encoder layer: ~5 CKKS levels (1 GELU + 1 poly attn + 1 BN + 2 matmul)
# vs. 38+ levels with polynomial softmax (exp + max + division) + polynomial LayerNorm
# =============================================================================

class PolyAttention(nn.Module):
    """Multi-head attention with polynomial activation replacing softmax."""
    
    def __init__(self, dim, num_heads=3, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.poly_attn_act = PolyAttentionActivation()
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # QKᵀ / √d — ciphertext x ciphertext multiplication under CKKS
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Polynomial activation replaces softmax entirely
        # No max subtraction, no division — just element-wise polynomial
        attn = self.poly_attn_act(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class PolyMlp(nn.Module):
    """FFN block with polynomial GELU."""
    
    def __init__(self, in_features, hidden_features=None, gelu_init='lsq'):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = PolyGELU(init=gelu_init)
        self.fc2 = nn.Linear(hidden_features, in_features)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class PolyBlock(nn.Module):
    """
    One transformer encoder block with all polynomial operations.
    
    Standard block:
      x = x + Attention(LayerNorm(x))
      x = x + FFN(LayerNorm(x))
    
    Our block:
      x = x + PolyAttention(BatchNorm(x))
      x = x + PolyFFN(BatchNorm(x))
    """
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, gelu_init='lsq'):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim)
        self.attn = PolyAttention(dim, num_heads)
        self.norm2 = nn.BatchNorm1d(dim)
        self.mlp = PolyMlp(dim, int(dim * mlp_ratio), gelu_init=gelu_init)
    
    def forward(self, x):
        # x shape: (B, N, C) where N=num_tokens, C=embed_dim
        # BatchNorm1d needs (B, C, N), so transpose
        x = x + self.attn(self.norm1(x.transpose(1, 2)).transpose(1, 2))
        x = x + self.mlp(self.norm2(x.transpose(1, 2)).transpose(1, 2))
        return x


class PolyDeiTTiny(nn.Module):
    """
    Polynomial DeiT-Tiny: fully FHE-compatible Vision Transformer.
    
    Architecture (matching DeiT-Tiny):
      - Patch embedding: 16x16 patches → 192-dim (linear projection, FHE-native)
      - Position encoding: learned, added to patches (plaintext add, free)
      - 12 encoder blocks with polynomial activations
      - CLS token → classification head (linear, FHE-native)
    
    Parameters: ~5.5M (same as standard DeiT-Tiny)
    """
    
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, gelu_init='lsq'):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch embedding: Conv2d is equivalent to linear projection of patches
        # Under CKKS: plaintext weight × ciphertext input = standard HE matmul
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, 
                                     kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        
        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Encoder blocks — ALL polynomial
        self.blocks = nn.ModuleList([
            PolyBlock(embed_dim, num_heads, mlp_ratio, gelu_init=gelu_init)
            for _ in range(depth)
        ])
        
        # Final norm + classification head
        self.norm = nn.BatchNorm1d(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding: (B, 3, 32, 32) → (B, 192, 8, 8) → (B, 64, 192)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position encoding (plaintext addition — free under CKKS)
        x = x + self.pos_embed
        
        # Pass through all polynomial encoder blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Normalize and classify from CLS token
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = x[:, 0]  # CLS token only
        x = self.head(x)
        return x


# =============================================================================
# PART 3: KNOWLEDGE DISTILLATION LOSS
# =============================================================================
#
# KD LOSS = α · KL(teacher_soft || student_soft) + (1-α) · CE(student, label)
#
# WHERE:
#   teacher_soft = softmax(teacher_logits / T)   ← soft targets
#   student_soft = log_softmax(student_logits / T)
#   T = temperature (higher → softer distributions → more information)
#   α = weight between KD loss and hard-label loss
#
# WHY TEMPERATURE MATTERS:
#   At T=1 (standard softmax), the teacher might output [0.99, 0.005, 0.005]
#   for a confident prediction. The student learns almost nothing from
#   the non-predicted classes (their probabilities are near zero).
#   
#   At T=4, the same logits become [0.45, 0.28, 0.27] — now the student
#   can see the teacher's "dark knowledge": which wrong classes are
#   LESS wrong than others. This relative ranking information helps
#   the polynomial student learn better internal representations.
#
# FOR YOUR RESEARCH:
#   Your finding that cold-start KD (T=4, α=0.5) outperforms warm-start
#   is novel. To make it publishable, you need:
#   1. Multiple temperatures: T ∈ {1, 2, 4, 8, 16}
#   2. Multiple α values: α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
#   3. Gradient norm logging for both cold-start and warm-start
#   4. Results on CIFAR-100 and APTOS-2019 (not just MedMNIST)
# =============================================================================

class DistillationLoss(nn.Module):
    """
    Combined KD + hard-label loss for offline distillation.
    
    Implements the standard Hinton et al. (2015) distillation loss
    used by BlindFed, Powerformer, and most FHE KD papers.
    """
    
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels):
        # Soft targets from teacher (detached — no gradients through teacher)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Student's soft predictions
        student_log_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # KL divergence between teacher and student soft distributions
        # Multiply by T² to scale gradients correctly (Hinton et al.)
        kd_loss = F.kl_div(
            student_log_soft, teacher_soft, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Standard cross-entropy with hard labels
        ce_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        return total_loss, kd_loss.item(), ce_loss.item()


# =============================================================================
# PART 4: TRAINING LOOPS
# =============================================================================

def get_dataloaders(dataset_name='cifar10', batch_size=128, img_size=32):
    """
    Get train and test dataloaders.
    
    For your research, replace this with:
      - APTOS-2019 (224x224, 5-class DR) for clinical validation
      - BloodMNIST (28x28, 8-class) for quick prototyping
      - Messidor-2 (variable, 4-class DR) for cross-dataset validation
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2470, 0.2435, 0.2616)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader


def create_teacher(num_classes=10, img_size=32):
    """
    Create a standard DeiT-Tiny teacher model.
    
    Uses timm's implementation with standard GELU, softmax, and LayerNorm.
    For CIFAR-10 (32x32), we use patch_size=4 → 64 patches.
    For APTOS-2019 (224x224), use patch_size=16 → 196 patches.
    """
    teacher = timm.create_model(
        'deit_tiny_patch16_224',
        pretrained=False,
        num_classes=num_classes,
        img_size=img_size,
        patch_size=4,  # 4 for 32x32, 16 for 224x224
    )
    return teacher


def create_student(num_classes=10, img_size=32, depth=12, gelu_init='lsq'):
    """
    Create a polynomial student model.
    Same parameter count as teacher, but all activations are polynomial.
    gelu_init: 'lsq' (least-squares fit) or 'blindfed' (BlindFed coefficients)
    """
    student = PolyDeiTTiny(
        img_size=img_size,
        patch_size=4,
        num_classes=num_classes,
        embed_dim=192,
        depth=depth,
        num_heads=3,
        mlp_ratio=4.0,
        gelu_init=gelu_init,
    )
    return student


@torch.no_grad()
def evaluate(model, test_loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def train_teacher(teacher, train_loader, test_loader, device, epochs=50):
    """Train the teacher model with standard cross-entropy."""
    teacher = teacher.to(device)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    print("\n" + "=" * 70)
    print("PHASE 1: Training Teacher (standard DeiT-Tiny)")
    print("  Activations: GELU (standard) + softmax + LayerNorm")
    print("  This is your accuracy CEILING — the best the architecture can do")
    print("=" * 70)
    
    best_acc = 0
    for epoch in range(epochs):
        teacher.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = teacher(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        acc = evaluate(teacher, test_loader, device)
        best_acc = max(best_acc, acc)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {total_loss/len(train_loader):.4f} | "
                  f"Acc: {acc:.2f}% | Best: {best_acc:.2f}%")
    
    print(f"\n  >>> Teacher best accuracy: {best_acc:.2f}%")
    return teacher, best_acc


def train_student_with_kd(student, teacher, train_loader, test_loader, 
                          device, epochs=50, temperature=4.0, alpha=0.5):
    """
    Train polynomial student via offline KD from teacher.
    
    This is the CORE of the BlindFed approach:
    - Teacher is FROZEN (no gradient updates)
    - Student learns to mimic teacher's soft output distributions
    - Temperature T=4 spreads the probability mass so the student
      can learn from the teacher's "dark knowledge"
    - Alpha=0.5 balances KD loss with hard-label CE loss
    """
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()  # Teacher is frozen — inference only
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    kd_criterion = DistillationLoss(temperature=temperature, alpha=alpha)
    
    print(f"\n" + "=" * 70)
    print(f"PHASE 2: Training Polynomial Student via KD")
    print(f"  Activations: PolyGELU + PolyAttnAct + BatchNorm")
    print(f"  KD params: temperature={temperature}, alpha={alpha}")
    print(f"  Teacher is FROZEN — student learns to mimic its outputs")
    print("=" * 70)
    
    best_acc = 0
    grad_norms = []  # Track gradient norms for your ablation study
    
    for epoch in range(epochs):
        student.train()
        total_loss, total_kd, total_ce = 0, 0, 0
        epoch_grad_norm = 0
        num_batches = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Get teacher's soft predictions (no gradient computation)
            with torch.no_grad():
                teacher_logits = teacher(images)
            
            # Get student's predictions
            student_logits = student(images)
            
            # Combined KD + CE loss
            loss, kd_loss, ce_loss = kd_criterion(
                student_logits, teacher_logits, labels
            )
            
            optimizer.zero_grad()
            loss.backward()
            
            # Track gradient norm (for your cold-start vs warm-start analysis)
            grad_norm = 0
            for p in student.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            epoch_grad_norm += grad_norm
            num_batches += 1
            
            optimizer.step()
            total_loss += loss.item()
            total_kd += kd_loss
            total_ce += ce_loss
        
        scheduler.step()
        acc = evaluate(student, test_loader, device)
        best_acc = max(best_acc, acc)
        avg_grad_norm = epoch_grad_norm / num_batches
        grad_norms.append(avg_grad_norm)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Show PolyGELU coefficients to see how they evolve
            poly_gelu = student.blocks[0].mlp.act
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {total_loss/len(train_loader):.4f} | "
                  f"KD: {total_kd/len(train_loader):.4f} | "
                  f"CE: {total_ce/len(train_loader):.4f} | "
                  f"Acc: {acc:.2f}% | "
                  f"GradNorm: {avg_grad_norm:.2f} | "
                  f"PolyGELU[0]: a={poly_gelu.a.item():.4f} "
                  f"b={poly_gelu.b.item():.4f} "
                  f"c={poly_gelu.c.item():.4f}")
    
    print(f"\n  >>> Student (with KD) best accuracy: {best_acc:.2f}%")
    return student, best_acc, grad_norms


def train_student_no_kd(student, train_loader, test_loader, device, epochs=50):
    """
    Train polynomial student WITHOUT KD (baseline comparison).
    
    This shows the accuracy you get if you just train the polynomial
    model from scratch with standard CE loss — no teacher guidance.
    The gap between this and the KD version quantifies KD's value.
    """
    student = student.to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n" + "=" * 70)
    print(f"PHASE 3: Training Polynomial Student WITHOUT KD (baseline)")
    print(f"  Same architecture as Phase 2, but no teacher guidance")
    print(f"  This quantifies how much accuracy KD recovers")
    print("=" * 70)
    
    best_acc = 0
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = student(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        acc = evaluate(student, test_loader, device)
        best_acc = max(best_acc, acc)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {total_loss/len(train_loader):.4f} | "
                  f"Acc: {acc:.2f}% | Best: {best_acc:.2f}%")
    
    print(f"\n  >>> Student (no KD) best accuracy: {best_acc:.2f}%")
    return student, best_acc


# =============================================================================
# PART 5: POLYNOMIAL COEFFICIENT ANALYSIS
# =============================================================================

def analyze_polynomial_coefficients(model):
    """
    Print the learned PolyGELU coefficients for each layer.
    
    This analysis shows whether the optimizer adapts coefficients per-layer
    or keeps them near the BlindFed initialization.
    
    For your paper: if coefficients diverge significantly from initialization,
    it argues for trainable (not fixed) coefficients — a minor but citable
    finding about the importance of per-layer adaptation.
    """
    print("\n" + "=" * 70)
    print("POLYNOMIAL COEFFICIENT ANALYSIS")
    print("  Initial values (BlindFed): a=0.125, b=0.250, c=0.500")
    print("  Learned values per layer after KD training:")
    print("=" * 70)
    
    for i, block in enumerate(model.blocks):
        poly = block.mlp.act
        a_val = poly.a.item()
        b_val = poly.b.item()
        c_val = poly.c.item()
        
        # How far did coefficients move from initialization?
        a_delta = abs(a_val - 0.125)
        b_delta = abs(b_val - 0.25)
        c_delta = abs(c_val - 0.5)
        
        marker = " ***" if max(a_delta, b_delta, c_delta) > 0.05 else ""
        print(f"  Layer {i:2d}: a={a_val:+.4f}  b={b_val:+.4f}  c={c_val:+.4f}"
              f"  (drift: {a_delta:.4f}, {b_delta:.4f}, {c_delta:.4f}){marker}")
    
    print("\n  *** = coefficient drifted >0.05 from initialization")
    print("  Large drift suggests fixed coefficients are suboptimal")


# =============================================================================
# PART 6: MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='BlindFed Offline KD Demo')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--depth', type=int, default=6,
                       help='Number of transformer layers (6 for quick demo, 12 for full)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Data
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
    
    # =========================================================================
    # EXPERIMENT 1: Train teacher (standard DeiT)
    # =========================================================================
    teacher = create_teacher(num_classes=10, img_size=32)
    # For quick demo, use same depth as student
    # In practice, teacher could be deeper/wider
    teacher, teacher_acc = train_teacher(
        teacher, train_loader, test_loader, device, epochs=args.epochs
    )
    
    # =========================================================================
    # EXPERIMENT 2: Train polynomial student WITH KD (cold-start)
    # =========================================================================
    student_kd = create_student(num_classes=10, img_size=32, depth=args.depth)
    student_kd, kd_acc, grad_norms = train_student_with_kd(
        student_kd, teacher, train_loader, test_loader, device,
        epochs=args.epochs, temperature=args.temperature, alpha=args.alpha
    )
    
    # =========================================================================
    # EXPERIMENT 3: Train polynomial student WITHOUT KD (baseline)
    # =========================================================================
    student_no_kd = create_student(num_classes=10, img_size=32, depth=args.depth)
    student_no_kd, no_kd_acc = train_student_no_kd(
        student_no_kd, train_loader, test_loader, device, epochs=args.epochs
    )
    
    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS COMPARISON")
    print("=" * 70)
    print(f"  Teacher (standard GELU+softmax+LN):    {teacher_acc:.2f}%")
    print(f"  Student (polynomial) WITH KD:           {kd_acc:.2f}%")
    print(f"  Student (polynomial) WITHOUT KD:        {no_kd_acc:.2f}%")
    print(f"  ---")
    print(f"  Accuracy drop from polynomial replacement:")
    print(f"    With KD:    {teacher_acc - kd_acc:+.2f}%")
    print(f"    Without KD: {teacher_acc - no_kd_acc:+.2f}%")
    print(f"  KD recovery:  {kd_acc - no_kd_acc:+.2f}% (KD advantage)")
    print("=" * 70)
    
    # Coefficient analysis
    analyze_polynomial_coefficients(student_kd)
    
    # CKKS depth budget analysis
    print("\n" + "=" * 70)
    print("CKKS DEPTH BUDGET ANALYSIS (per encoder layer)")
    print("=" * 70)
    print("  PolyGELU (ax²+bx+c):            1 level  (x*x)")
    print("  PolyAttnAct (ax²+bx+c):          1 level  (replaces softmax entirely)")
    print("  BatchNorm (affine transform):    ~1 level (scale+shift)")
    print("  QKV projection (plaintext×ct):   ~1 level (matmul)")
    print("  FFN projection (plaintext×ct):   ~1 level (matmul)")
    print("  ---")
    print(f"  Total per layer: ~5 levels")
    print(f"  Total for {args.depth} layers: ~{5 * args.depth} levels")
    print(f"  CKKS budget (N=2^16): ~21 levels between bootstraps")
    print(f"  Bootstraps needed: ~{math.ceil(5 * args.depth / 21)}")
    print(f"")
    print(f"  NOTE: Polynomial attention (1 level) replaces softmax entirely")
    print(f"  — no max subtraction, no division, no normalization needed.")
    print("=" * 70)
    
    # Save for later extension
    torch.save({
        'teacher_state': teacher.state_dict(),
        'student_kd_state': student_kd.state_dict(),
        'student_no_kd_state': student_no_kd.state_dict(),
        'teacher_acc': teacher_acc,
        'kd_acc': kd_acc,
        'no_kd_acc': no_kd_acc,
        'grad_norms': grad_norms,
        'args': vars(args),
    }, 'blindfed_kd_results.pth')
    print("\nResults saved to blindfed_kd_results.pth")


if __name__ == '__main__':
    main()