"""
Cold-Start vs Warm-Start KD for Polynomial Vision Transformers
================================================================

RESEARCH QUESTION:
  Does cold-start KD (random init + teacher guidance) outperform
  warm-start KD (pretrained init + teacher guidance) for polynomial ViTs?

HYPOTHESIS:
  Pretrained ReLU/GELU features create out-of-range activations for
  polynomial functions, causing gradient explosion in early training.
  Cold-start avoids this because random init produces small activations
  that stay within the polynomial's well-behaved range.

EXPERIMENT DESIGN:
  3 variants × 3 datasets × 5 temperatures × 3 seeds = 135 runs
  + LR sensitivity check: 3 extra LRs × 3 seeds = 9 runs
  Total: 144 runs, ~5 hours on RTX A6000

  Variant 1 (V1): Cold-start + KD — random init, polynomial activations
  Variant 2 (V2): Warm-start + KD — pretrained init, polynomial activations
  Variant 3 (V3): Warm-start, no KD — pretrained init, CE loss only

DIAGNOSTICS (logged every epoch):
  1. Per-layer gradient norms
  2. KL divergence (student vs teacher)
  3. PolyGELU coefficient trajectories (a, b, c per layer)
  4. Activation range statistics (min/max entering each PolyGELU)

USAGE:
  # Full sweep (~5 hours on A6000)
  python cold_vs_warm_experiment.py

  # Quick test on one dataset (30 min)
  python cold_vs_warm_experiment.py --datasets cifar100 --temps 4 --seeds 42

  # Resume interrupted sweep
  python cold_vs_warm_experiment.py --resume

Author: Panchan's PhD research — AAAI 2027 target
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import timm
import argparse
import os
import json
import time
import copy
import math
import numpy as np
from pathlib import Path
from collections import defaultdict

# Import our polynomial ViT components
from blindfed_offline_kd import (
    PolyGELU, PolyAttentionActivation, PolyDeiTTiny,
    DistillationLoss
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'datasets': ['cifar100', 'bloodmnist', 'pathmnist'],
    'temperatures': [1.0, 2.0, 4.0, 8.0, 16.0],
    'seeds': [42, 123, 456],
    'alpha': 0.1,           # KD loss weight — low to prevent KD dominating CE
                            # With T=4 on 100 classes: KD term has T^2=16 scaling,
                            # so alpha=0.1 gives KD:CE ratio ≈ 0.4:1 (balanced).
    'kd_warmup_fraction': 0.3,  # Train CE-only for first 30% of epochs
                                # V3 shows polynomial models need ~20-30 epochs
                                # to establish basic representations before KD helps
    'kd_rampup_fraction': 0.1,  # Linearly ramp alpha from 0 to target over next 10%
                                # Prevents the sudden KD introduction that caused collapse
    'lr': 1e-3,             # Learning rate
    'weight_decay': 0.05,   # AdamW weight decay
    'epochs': 100,          # Training epochs per run
    'batch_size': 128,
    'depth': 6,             # Transformer layers (6 for speed, 12 for full)
    'embed_dim': 192,
    'num_heads': 3,
    'mlp_ratio': 4.0,
    'img_size': 32,
    'patch_size': 4,
    'gelu_init': 'lsq',    # LSQ initialization for PolyGELU
    # LR sensitivity check for warm-start
    'lr_sensitivity_lrs': [1e-4, 5e-4, 5e-3],
    'lr_sensitivity_temp': 4.0,
    'lr_sensitivity_dataset': 'cifar100',
}


# =============================================================================
# DIAGNOSTIC LOGGER
# =============================================================================

class DiagnosticLogger:
    """
    Logs four diagnostic measurements per epoch for mechanistic analysis.
    
    These measurements transform the experiment from "we measured accuracy"
    to "we understand WHY cold-start beats warm-start."
    """
    
    def __init__(self, num_layers, log_dir):
        self.num_layers = num_layers
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for all diagnostics
        self.data = {
            'epoch_accs': [],
            'epoch_losses': [],
            'gradient_norms': [],        # List of dicts: {layer_i: norm}
            'kl_divergences': [],         # Per-epoch KL div (student vs teacher)
            'poly_gelu_coeffs': [],       # List of dicts: {layer_i: {a, b, c}}
            'activation_ranges': [],      # List of dicts: {layer_i: {min, max, mean, std}}
            'poly_attn_coeffs': [],       # PolyAttnAct coefficients per layer
        }
        
        # Hooks for activation range tracking
        self._activation_stats = {}
        self._hooks = []
    
    def register_activation_hooks(self, model):
        """Register forward hooks on PolyGELU layers to capture input ranges."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        
        for i, block in enumerate(model.blocks):
            layer_idx = i
            def make_hook(idx):
                def hook_fn(module, input, output):
                    x = input[0].detach()
                    self._activation_stats[idx] = {
                        'min': x.min().item(),
                        'max': x.max().item(),
                        'mean': x.mean().item(),
                        'std': x.std().item(),
                        'abs_max': x.abs().max().item(),
                    }
                return hook_fn
            h = block.mlp.act.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(h)
    
    def log_epoch(self, model, teacher, student_logits_all, teacher_logits_all,
                  acc, loss, temperature):
        """Log all diagnostics for one epoch."""
        # 1. Gradient norms per layer
        grad_norms = {}
        for i, block in enumerate(model.blocks):
            block_norm = 0.0
            count = 0
            for p in block.parameters():
                if p.grad is not None:
                    block_norm += p.grad.data.norm(2).item() ** 2
                    count += 1
            grad_norms[f'layer_{i}'] = math.sqrt(block_norm) if count > 0 else 0.0
        self.data['gradient_norms'].append(grad_norms)
        
        # 2. KL divergence
        if teacher_logits_all is not None and student_logits_all is not None:
            with torch.no_grad():
                t_soft = F.softmax(teacher_logits_all / temperature, dim=-1)
                s_log_soft = F.log_softmax(student_logits_all / temperature, dim=-1)
                kl = F.kl_div(s_log_soft, t_soft, reduction='batchmean').item()
        else:
            kl = float('nan')
        self.data['kl_divergences'].append(kl)
        
        # 3. PolyGELU coefficients
        gelu_coeffs = {}
        for i, block in enumerate(model.blocks):
            pg = block.mlp.act
            gelu_coeffs[f'layer_{i}'] = {
                'a': pg.a.item(), 'b': pg.b.item(), 'c': pg.c.item()
            }
        self.data['poly_gelu_coeffs'].append(gelu_coeffs)
        
        # 4. PolyAttnAct coefficients
        attn_coeffs = {}
        for i, block in enumerate(model.blocks):
            pa = block.attn.poly_attn_act
            attn_coeffs[f'layer_{i}'] = {
                'a': pa.a.item(), 'b': pa.b.item(), 'c': pa.c.item()
            }
        self.data['poly_attn_coeffs'].append(attn_coeffs)
        
        # 5. Activation ranges (from hooks)
        self.data['activation_ranges'].append(dict(self._activation_stats))
        self._activation_stats = {}
        
        # Accuracy and loss
        self.data['epoch_accs'].append(acc)
        self.data['epoch_losses'].append(loss)
    
    def save(self, filename):
        """Save all diagnostics to JSON."""
        # Convert any remaining numpy/tensor types
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        filepath = self.log_dir / filename
        with open(filepath, 'w') as f:
            json.dump(convert(self.data), f, indent=2)
    
    def cleanup(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


# =============================================================================
# DATASET LOADING
# =============================================================================

def get_dataset(name, batch_size=128, img_size=32):
    """
    Load dataset with appropriate transforms.
    
    Supported: cifar100, bloodmnist, pathmnist
    """
    norm_mean = (0.4914, 0.4822, 0.4465)
    norm_std = (0.2470, 0.2435, 0.2616)
    
    transform_train = transforms.Compose([
        transforms.Resize(img_size) if name != 'cifar100' else transforms.Lambda(lambda x: x),
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(img_size) if name != 'cifar100' else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    
    if name == 'cifar100':
        train_ds = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        test_ds = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        num_classes = 100
        
    elif name in ('bloodmnist', 'pathmnist'):
        # MedMNIST datasets
        try:
            import medmnist
            from medmnist import BloodMNIST, PathMNIST
        except ImportError:
            print("Installing medmnist...")
            os.system("pip install medmnist --break-system-packages -q")
            import medmnist
            from medmnist import BloodMNIST, PathMNIST
        
        cls = BloodMNIST if name == 'bloodmnist' else PathMNIST
        
        # MedMNIST returns PIL images, need 3-channel conversion
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(norm_mean, norm_std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(norm_mean, norm_std),
        ])
        
        train_ds = cls(split='train', transform=train_transform, download=True)
        test_ds = cls(split='test', transform=test_transform, download=True)
        
        # MedMNIST labels are 2D arrays, need to squeeze
        class MedMNISTWrapper(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
            def __len__(self):
                return len(self.dataset)
            def __getitem__(self, idx):
                img, label = self.dataset[idx]
                return img, int(label.squeeze())
        
        train_ds = MedMNISTWrapper(train_ds)
        test_ds = MedMNISTWrapper(test_ds)
        num_classes = 8 if name == 'bloodmnist' else 9
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, num_classes


# =============================================================================
# TEACHER TRAINING
# =============================================================================

def train_teacher(num_classes, train_loader, test_loader, device, config):
    """
    Train a standard DeiT-Tiny teacher with GELU, softmax, LayerNorm.
    Returns the trained teacher model.
    """
    teacher = timm.create_model(
        'deit_tiny_patch16_224',
        pretrained=False,
        num_classes=num_classes,
        img_size=config['img_size'],
        patch_size=config['patch_size'],
    )
    teacher = teacher.to(device)
    
    optimizer = torch.optim.AdamW(
        teacher.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'])
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(config['epochs']):
        teacher.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(teacher(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        acc = evaluate(teacher, test_loader, device)
        best_acc = max(best_acc, acc)
        
        if (epoch + 1) % 25 == 0:
            print(f"    Teacher epoch {epoch+1}/{config['epochs']}: "
                  f"acc={acc:.2f}% (best={best_acc:.2f}%)")
    
    return teacher, best_acc


# =============================================================================
# WARM-START WEIGHT TRANSFER
# =============================================================================

def transfer_pretrained_weights(teacher, student):
    """
    Transfer compatible weights from trained teacher to polynomial student.
    
    WHAT TRANSFERS:
      - patch_embed (Conv2d weights and bias)
      - cls_token, pos_embed
      - Per-block: attn.qkv, attn.proj, mlp.fc1, mlp.fc2 (all Linear layers)
      - head (classification head)
    
    WHAT DOES NOT TRANSFER:
      - LayerNorm → BatchNorm (different parameterization, different statistics)
      - GELU → PolyGELU (polynomial coefficients stay at initialization)
      - softmax → PolyAttnAct (polynomial coefficients stay at initialization)
    
    This is exactly how BlindFed and Powerformer set up warm-start KD.
    """
    teacher_sd = teacher.state_dict()
    student_sd = student.state_dict()
    
    transferred = []
    skipped = []
    
    # Direct transfers (same key names between timm DeiT and our PolyDeiTTiny)
    direct_keys = ['patch_embed.weight', 'patch_embed.bias',
                   'cls_token', 'pos_embed',
                   'head.weight', 'head.bias']
    
    for key in direct_keys:
        if key in teacher_sd and key in student_sd:
            if teacher_sd[key].shape == student_sd[key].shape:
                student_sd[key] = teacher_sd[key].clone()
                transferred.append(key)
            else:
                skipped.append(f"{key} (shape mismatch: {teacher_sd[key].shape} vs {student_sd[key].shape})")
    
    # Per-block transfers
    # timm DeiT uses: blocks.{i}.attn.qkv, blocks.{i}.attn.proj,
    #                 blocks.{i}.mlp.fc1, blocks.{i}.mlp.fc2
    # Our model uses the same key structure for linear layers
    num_blocks = min(len(teacher.blocks) if hasattr(teacher, 'blocks') else 12,
                     len(student.blocks))
    
    for i in range(num_blocks):
        linear_keys = [
            f'blocks.{i}.attn.qkv.weight', f'blocks.{i}.attn.qkv.bias',
            f'blocks.{i}.attn.proj.weight', f'blocks.{i}.attn.proj.bias',
            f'blocks.{i}.mlp.fc1.weight', f'blocks.{i}.mlp.fc1.bias',
            f'blocks.{i}.mlp.fc2.weight', f'blocks.{i}.mlp.fc2.bias',
        ]
        for key in linear_keys:
            if key in teacher_sd and key in student_sd:
                if teacher_sd[key].shape == student_sd[key].shape:
                    student_sd[key] = teacher_sd[key].clone()
                    transferred.append(key)
                else:
                    skipped.append(f"{key} (shape mismatch)")
            else:
                skipped.append(f"{key} (not found)")
    
    student.load_state_dict(student_sd)
    
    return transferred, skipped


# =============================================================================
# EVALUATION
# =============================================================================

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


@torch.no_grad()
def collect_logits(model, loader, device, max_batches=10):
    """Collect logits from model on a subset of data for KL computation."""
    model.eval()
    all_logits = []
    for i, (images, labels) in enumerate(loader):
        if i >= max_batches:
            break
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0)


# =============================================================================
# SINGLE EXPERIMENT RUNNER
# =============================================================================

def run_single_experiment(variant, teacher, num_classes, train_loader, test_loader,
                          device, config, temperature, seed, log_dir):
    """
    Run one experiment configuration with full diagnostic logging.
    
    variant: 'cold_kd' (V1), 'warm_kd' (V2), or 'warm_no_kd' (V3)
    Returns: best_acc, final_acc, logger
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Create student
    student = PolyDeiTTiny(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        num_classes=num_classes,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        gelu_init=config['gelu_init'],
    ).to(device)
    
    # Warm-start: transfer pretrained weights
    if variant in ('warm_kd', 'warm_no_kd'):
        transferred, skipped = transfer_pretrained_weights(teacher, student)
    
    # Setup optimizer and scheduler
    lr = config.get('override_lr', config['lr'])
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=lr, weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'])
    
    # Loss function
    use_kd = variant in ('cold_kd', 'warm_kd')
    ce_criterion = nn.CrossEntropyLoss()
    target_alpha = config['alpha']
    
    # KD schedule: CE-only warmup → linear ramp-up → full KD
    warmup_end = int(config['epochs'] * config.get('kd_warmup_fraction', 0.3))
    rampup_end = warmup_end + int(config['epochs'] * config.get('kd_rampup_fraction', 0.1))
    
    # Diagnostic logger
    logger = DiagnosticLogger(config['depth'], log_dir)
    logger.register_activation_hooks(student)
    
    teacher.eval()
    best_acc = 0.0
    
    for epoch in range(config['epochs']):
        student.train()
        total_loss = 0.0
        
        # Compute current KD alpha based on schedule
        if not use_kd or epoch < warmup_end:
            current_alpha = 0.0
            kd_status = "CE-only"
        elif epoch < rampup_end:
            # Linear ramp from 0 to target_alpha
            progress = (epoch - warmup_end) / max(rampup_end - warmup_end, 1)
            current_alpha = target_alpha * progress
            kd_status = f"ramp({current_alpha:.3f})"
        else:
            current_alpha = target_alpha
            kd_status = f"KD+CE(a={current_alpha})"
        
        # Create loss function with current alpha
        if current_alpha > 0:
            kd_criterion = DistillationLoss(temperature=temperature, alpha=current_alpha)
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            student_logits = student(images)
            
            if current_alpha > 0:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                loss, _, _ = kd_criterion(student_logits, teacher_logits, labels)
            else:
                loss = ce_criterion(student_logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        # Evaluate
        acc = evaluate(student, test_loader, device)
        best_acc = max(best_acc, acc)
        
        # Collect logits for KL divergence computation
        s_logits = collect_logits(student, test_loader, device)
        t_logits = collect_logits(teacher, test_loader, device) if use_kd else None
        
        # Log diagnostics
        logger.log_epoch(
            model=student,
            teacher=teacher,
            student_logits_all=s_logits,
            teacher_logits_all=t_logits,
            acc=acc,
            loss=total_loss / len(train_loader),
            temperature=temperature,
        )
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            pg = student.blocks[0].mlp.act
            grad_norm = logger.data['gradient_norms'][-1]
            avg_gnorm = np.mean([v for v in grad_norm.values()])
            print(f"      Ep {epoch+1:3d}: acc={acc:.2f}% best={best_acc:.2f}% "
                  f"loss={total_loss/len(train_loader):.4f} "
                  f"gnorm={avg_gnorm:.3f} [{kd_status}] "
                  f"poly[0]=({pg.a.item():.3f},{pg.b.item():.3f},{pg.c.item():.3f})")
    
    logger.cleanup()
    final_acc = acc
    
    return best_acc, final_acc, logger


# =============================================================================
# FULL EXPERIMENT SWEEP
# =============================================================================

def run_full_sweep(config, device):
    """
    Run the complete 135-run experiment sweep.
    
    Structure:
      For each dataset:
        1. Train teacher once
        2. For each temperature × seed:
           Run V1 (cold_kd), V2 (warm_kd), V3 (warm_no_kd)
        3. LR sensitivity check for V2 on cifar100
    """
    results_dir = Path('experiment_results')
    results_dir.mkdir(exist_ok=True)
    
    all_results = {}
    
    total_runs = (len(config['datasets']) * len(config['temperatures']) * 
                  len(config['seeds']) * 3)  # 3 variants
    total_runs += len(config['lr_sensitivity_lrs']) * len(config['seeds'])  # LR check
    run_count = 0
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"COLD-START vs WARM-START KD EXPERIMENT")
    print(f"  Total runs: {total_runs}")
    print(f"  Datasets: {config['datasets']}")
    print(f"  Temperatures: {config['temperatures']}")
    print(f"  Seeds: {config['seeds']}")
    print(f"  Depth: {config['depth']} layers")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")
    
    for dataset_name in config['datasets']:
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*60}")
        
        # Load data
        train_loader, test_loader, num_classes = get_dataset(
            dataset_name, config['batch_size'], config['img_size'])
        
        # Train teacher ONCE per dataset
        print(f"\n  Training teacher (standard DeiT-Tiny)...")
        teacher_seed = 0  # Fixed seed for teacher reproducibility
        torch.manual_seed(teacher_seed)
        torch.cuda.manual_seed_all(teacher_seed)
        teacher, teacher_acc = train_teacher(
            num_classes, train_loader, test_loader, device, config)
        print(f"  Teacher accuracy: {teacher_acc:.2f}%")
        
        # Save teacher
        teacher_path = results_dir / f"teacher_{dataset_name}.pth"
        torch.save(teacher.state_dict(), teacher_path)
        
        dataset_results = {
            'teacher_acc': teacher_acc,
            'num_classes': num_classes,
            'runs': {}
        }
        
        for temp in config['temperatures']:
            for seed in config['seeds']:
                for variant in ['cold_kd', 'warm_kd', 'warm_no_kd']:
                    run_count += 1
                    run_key = f"{variant}_T{temp}_s{seed}"
                    log_dir = results_dir / dataset_name / run_key
                    
                    # Check if already completed (for --resume)
                    result_file = log_dir / 'diagnostics.json'
                    if result_file.exists() and config.get('resume', False):
                        print(f"  [{run_count}/{total_runs}] SKIP {run_key} (already done)")
                        # Load existing result
                        with open(result_file) as f:
                            existing = json.load(f)
                        dataset_results['runs'][run_key] = {
                            'best_acc': max(existing['epoch_accs']),
                            'final_acc': existing['epoch_accs'][-1],
                        }
                        continue
                    
                    elapsed = time.time() - start_time
                    eta = (elapsed / max(run_count - 1, 1)) * (total_runs - run_count)
                    
                    print(f"\n  [{run_count}/{total_runs}] {dataset_name} / {run_key} "
                          f"(elapsed: {elapsed/60:.0f}min, ETA: {eta/60:.0f}min)")
                    
                    best_acc, final_acc, logger = run_single_experiment(
                        variant=variant,
                        teacher=teacher,
                        num_classes=num_classes,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        device=device,
                        config=config,
                        temperature=temp,
                        seed=seed,
                        log_dir=log_dir,
                    )
                    
                    # Save diagnostics
                    logger.save('diagnostics.json')
                    
                    dataset_results['runs'][run_key] = {
                        'best_acc': best_acc,
                        'final_acc': final_acc,
                    }
                    
                    print(f"      Result: best={best_acc:.2f}%, final={final_acc:.2f}%")
        
        # LR sensitivity check (only for cifar100)
        if dataset_name == config.get('lr_sensitivity_dataset', 'cifar100'):
            print(f"\n  --- LR SENSITIVITY CHECK for warm-start ---")
            temp_lr = config.get('lr_sensitivity_temp', 4.0)
            
            for lr_val in config.get('lr_sensitivity_lrs', []):
                for seed in config['seeds']:
                    run_count += 1
                    run_key = f"warm_kd_lr{lr_val}_T{temp_lr}_s{seed}"
                    log_dir = results_dir / dataset_name / run_key
                    
                    print(f"\n  [{run_count}/{total_runs}] LR={lr_val} / {run_key}")
                    
                    # Override LR for this run
                    lr_config = dict(config)
                    lr_config['override_lr'] = lr_val
                    
                    best_acc, final_acc, logger = run_single_experiment(
                        variant='warm_kd',
                        teacher=teacher,
                        num_classes=num_classes,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        device=device,
                        config=lr_config,
                        temperature=temp_lr,
                        seed=seed,
                        log_dir=log_dir,
                    )
                    
                    logger.save('diagnostics.json')
                    dataset_results['runs'][run_key] = {
                        'best_acc': best_acc,
                        'final_acc': final_acc,
                    }
                    print(f"      Result: best={best_acc:.2f}%")
        
        all_results[dataset_name] = dataset_results
    
    # Save all results
    with open(results_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"ALL RUNS COMPLETE in {total_time/3600:.1f} hours")
    print(f"Results saved to {results_dir}/")
    print(f"{'='*70}")
    
    return all_results


# =============================================================================
# RESULTS REPORTING
# =============================================================================

def generate_report(all_results, config):
    """
    Generate the five deliverables for the paper:
    1. Accuracy table (3 variants × 3 datasets × 5 temperatures)
    2. Gradient norm comparison
    3. Activation range comparison
    4. Coefficient drift analysis
    5. KL divergence convergence
    """
    results_dir = Path('experiment_results')
    
    print(f"\n{'='*70}")
    print("RESULTS REPORT")
    print(f"{'='*70}")
    
    # =============================================
    # TABLE 1: Main accuracy results
    # =============================================
    print(f"\n{'='*70}")
    print("TABLE 1: Best accuracy (mean ± std across 3 seeds)")
    print(f"{'='*70}")
    
    header = f"{'Dataset':<12} {'Temp':<6} {'Teacher':<10} {'V1:Cold+KD':<14} {'V2:Warm+KD':<14} {'V3:Warm-noKD':<14} {'V1-V2':<8}"
    print(header)
    print("-" * len(header))
    
    for dataset_name in config['datasets']:
        if dataset_name not in all_results:
            continue
        
        dr = all_results[dataset_name]
        teacher_acc = dr['teacher_acc']
        
        for temp in config['temperatures']:
            accs = {'cold_kd': [], 'warm_kd': [], 'warm_no_kd': []}
            
            for seed in config['seeds']:
                for variant in ['cold_kd', 'warm_kd', 'warm_no_kd']:
                    key = f"{variant}_T{temp}_s{seed}"
                    if key in dr['runs']:
                        accs[variant].append(dr['runs'][key]['best_acc'])
            
            row = f"{dataset_name:<12} T={temp:<4.0f}"
            
            if temp == config['temperatures'][0]:
                row = f"{dataset_name:<12} T={temp:<4.0f} {teacher_acc:>7.2f}%  "
            else:
                row = f"{'':>12} T={temp:<4.0f} {'':>8}  "
            
            for variant in ['cold_kd', 'warm_kd', 'warm_no_kd']:
                if accs[variant]:
                    mean = np.mean(accs[variant])
                    std = np.std(accs[variant])
                    row += f" {mean:>5.2f}±{std:.2f}   "
                else:
                    row += f" {'N/A':>10}   "
            
            # V1 - V2 difference
            if accs['cold_kd'] and accs['warm_kd']:
                diff = np.mean(accs['cold_kd']) - np.mean(accs['warm_kd'])
                marker = "***" if diff > 1.0 else "**" if diff > 0.5 else "*" if diff > 0 else ""
                row += f" {diff:+.2f}{marker}"
            
            print(row)
        print()
    
    # =============================================
    # TABLE 2: LR sensitivity check
    # =============================================
    lr_dataset = config.get('lr_sensitivity_dataset', 'cifar100')
    if lr_dataset in all_results:
        print(f"\n{'='*70}")
        print(f"TABLE 2: LR sensitivity for warm-start (T={config.get('lr_sensitivity_temp', 4.0)}, {lr_dataset})")
        print(f"{'='*70}")
        
        dr = all_results[lr_dataset]
        lr_vals = [config['lr']] + config.get('lr_sensitivity_lrs', [])
        
        for lr_val in lr_vals:
            accs = []
            for seed in config['seeds']:
                if lr_val == config['lr']:
                    key = f"warm_kd_T{config.get('lr_sensitivity_temp', 4.0)}_s{seed}"
                else:
                    key = f"warm_kd_lr{lr_val}_T{config.get('lr_sensitivity_temp', 4.0)}_s{seed}"
                if key in dr['runs']:
                    accs.append(dr['runs'][key]['best_acc'])
            
            if accs:
                mean = np.mean(accs)
                std = np.std(accs)
                default_marker = " (default)" if lr_val == config['lr'] else ""
                print(f"  LR={lr_val:.0e}: {mean:.2f}±{std:.2f}%{default_marker}")
        
        # Compare best warm-start LR against cold-start
        cold_accs = []
        for seed in config['seeds']:
            key = f"cold_kd_T{config.get('lr_sensitivity_temp', 4.0)}_s{seed}"
            if key in dr['runs']:
                cold_accs.append(dr['runs'][key]['best_acc'])
        if cold_accs:
            print(f"\n  Cold-start (LR=1e-3):  {np.mean(cold_accs):.2f}±{np.std(cold_accs):.2f}%")
            print(f"  If warm-start beats cold-start at lower LR → finding is about LR, not init")
            print(f"  If cold-start still wins → finding is genuine init mismatch")
    
    # =============================================
    # SUMMARY VERDICT
    # =============================================
    print(f"\n{'='*70}")
    print("SUMMARY VERDICT")
    print(f"{'='*70}")
    
    cold_wins = 0
    warm_wins = 0
    total_comparisons = 0
    degenerate_runs = 0
    
    for dataset_name in config['datasets']:
        if dataset_name not in all_results:
            continue
        dr = all_results[dataset_name]
        num_classes = dr.get('num_classes', 100)
        random_chance = 100.0 / num_classes  # 1% for 100 classes, 12.5% for 8, etc.
        degenerate_threshold = random_chance * 2  # Below 2x random = likely collapsed
        
        for temp in config['temperatures']:
            cold_accs = []
            warm_accs = []
            for seed in config['seeds']:
                ck = f"cold_kd_T{temp}_s{seed}"
                wk = f"warm_kd_T{temp}_s{seed}"
                if ck in dr['runs']:
                    cold_accs.append(dr['runs'][ck]['best_acc'])
                if wk in dr['runs']:
                    warm_accs.append(dr['runs'][wk]['best_acc'])
            
            if cold_accs and warm_accs:
                cold_mean = np.mean(cold_accs)
                warm_mean = np.mean(warm_accs)
                
                # Check for degenerate training (both near random chance)
                if cold_mean < degenerate_threshold and warm_mean < degenerate_threshold:
                    degenerate_runs += 1
                    print(f"  WARNING: {dataset_name} T={temp} — BOTH variants collapsed "
                          f"(cold={cold_mean:.1f}%, warm={warm_mean:.1f}%, "
                          f"random={random_chance:.1f}%)")
                    continue
                
                total_comparisons += 1
                if cold_mean > warm_mean:
                    cold_wins += 1
                else:
                    warm_wins += 1
    
    if degenerate_runs > 0:
        print(f"\n  DEGENERATE RUNS: {degenerate_runs} configurations collapsed to random chance.")
        print(f"  These are EXCLUDED from the verdict — cannot compare two broken models.")
    
    if total_comparisons == 0:
        print(f"\n  NO VALID COMPARISONS — all runs collapsed. Fix training before interpreting.")
    else:
        print(f"\n  Valid comparisons: {total_comparisons}")
        print(f"  Cold-start wins: {cold_wins}/{total_comparisons}")
        print(f"  Warm-start wins: {warm_wins}/{total_comparisons}")
    
    print(f"\n  Next steps:")
    print(f"  1. Check gradient_norms in diagnostics JSON for early-epoch explosion")
    print(f"  2. Check activation_ranges for out-of-range values in warm-start")
    print(f"  3. Plot coefficient drift to see if warm-start oscillates")
    print(f"  4. Verify finding at depth=12 with a confirmation run")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Cold-Start vs Warm-Start KD Experiment')
    parser.add_argument('--datasets', nargs='+', 
                       default=DEFAULT_CONFIG['datasets'],
                       help='Datasets to test')
    parser.add_argument('--temps', nargs='+', type=float,
                       default=DEFAULT_CONFIG['temperatures'],
                       help='KD temperatures')
    parser.add_argument('--seeds', nargs='+', type=int,
                       default=DEFAULT_CONFIG['seeds'],
                       help='Random seeds')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'])
    parser.add_argument('--depth', type=int, default=DEFAULT_CONFIG['depth'])
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'])
    parser.add_argument('--gelu-init', default='lsq', choices=['lsq', 'blindfed'])
    parser.add_argument('--resume', action='store_true',
                       help='Skip already-completed runs')
    parser.add_argument('--report-only', action='store_true',
                       help='Just generate report from existing results')
    args = parser.parse_args()
    
    config = dict(DEFAULT_CONFIG)
    config.update({
        'datasets': args.datasets,
        'temperatures': args.temps,
        'seeds': args.seeds,
        'epochs': args.epochs,
        'depth': args.depth,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'gelu_init': args.gelu_init,
        'resume': args.resume,
    })
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.report_only:
        results_path = Path('experiment_results/all_results.json')
        if results_path.exists():
            with open(results_path) as f:
                all_results = json.load(f)
            generate_report(all_results, config)
        else:
            print("No results found. Run experiments first.")
        return
    
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Save config
    results_dir = Path('experiment_results')
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run sweep
    all_results = run_full_sweep(config, device)
    
    # Generate report
    generate_report(all_results, config)


if __name__ == '__main__':
    main()