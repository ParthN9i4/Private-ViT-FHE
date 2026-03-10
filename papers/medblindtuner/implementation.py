"""
MedBlindTuner baseline: DeiT backbone (plaintext) + encrypted classification head.

Implements:
  - MedViTFeatureExtractor: run DeiT-Tiny backbone, return [CLS] embedding
  - EncryptedClassificationHead: linear classifier on CKKS ciphertext
  - run_medmnist_benchmark(): reproduce MedBlindTuner results

Paper: MedBlindTuner — Privacy-Preserving Fine-Tuning on Biomedical Images
Dataset: https://medmnist.com/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# MedMNIST Dataset Config
# ---------------------------------------------------------------------------

MEDMNIST_DATASETS = {
    "dermamnist":    {"n_classes": 7,  "task": "multi-class"},
    "breastmnist":   {"n_classes": 2,  "task": "binary"},
    "pneumoniamnist":{"n_classes": 2,  "task": "binary"},
    "bloodmnist":    {"n_classes": 8,  "task": "multi-class"},
    "retinamnist":   {"n_classes": 5,  "task": "multi-class"},
}


def get_medmnist_loaders(
    dataset_name: str,
    batch_size: int = 64,
    image_size: int = 224,
) -> Tuple:
    """
    Load a MedMNIST dataset and return (train_loader, val_loader, test_loader).

    Requires: pip install medmnist

    Args:
        dataset_name: One of the keys in MEDMNIST_DATASETS
        image_size: Resize target (224 for DeiT, 32 for small models)
    """
    try:
        import medmnist
        from medmnist import INFO
        import torchvision.transforms as T
        from torch.utils.data import DataLoader
    except ImportError:
        raise ImportError("Install medmnist: pip install medmnist")

    assert dataset_name in MEDMNIST_DATASETS, \
        f"Unknown dataset. Choose from: {list(MEDMNIST_DATASETS.keys())}"

    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_ds = DataClass(split="train", transform=transform, download=True)
    val_ds   = DataClass(split="val",   transform=transform, download=True)
    test_ds  = DataClass(split="test",  transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# DeiT Backbone Feature Extractor
# ---------------------------------------------------------------------------

class MedViTFeatureExtractor(nn.Module):
    """
    DeiT-Tiny backbone fine-tuned on medical images.

    Runs entirely in plaintext on the client.
    Outputs the [CLS] token embedding which is then encrypted.

    For production: export to ONNX and run on client device.

    Args:
        pretrained: Load pretrained DeiT-Tiny from timm
        n_classes: Number of output classes (for fine-tuning head)
        freeze_backbone: If True, only fine-tune the classifier head
    """

    def __init__(
        self,
        pretrained: bool = True,
        n_classes: int = 7,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("Install timm: pip install timm")

        # Load DeiT-Tiny without the classification head
        self.backbone = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=0,  # no head — returns [CLS] features directly
        )
        self.feature_dim = self.backbone.num_features  # 192 for DeiT-Tiny
        self.head = nn.Linear(self.feature_dim, n_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract [CLS] token features (to be encrypted).

        Args:
            x: (B, 3, 224, 224) images
        Returns:
            features: (B, feature_dim) — plaintext [CLS] embeddings
        """
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full plaintext forward pass (for fine-tuning)."""
        return self.head(self.backbone(x))

    def fine_tune(
        self,
        train_loader,
        val_loader,
        epochs: int = 20,
        lr: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """Fine-tune on a medical imaging dataset."""
        self.to(device)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr, weight_decay=0.05,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        for epoch in range(epochs):
            self.train()
            for x, y in train_loader:
                x = x.to(device)
                y = y.squeeze().long().to(device)
                loss = criterion(self(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validation
            self.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.squeeze().long().to(device)
                    correct += (self(x).argmax(-1) == y).sum().item()
                    total += y.size(0)
            acc = 100 * correct / total
            best_val_acc = max(best_val_acc, acc)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Val Acc: {acc:.2f}%")

        print(f"Best val accuracy: {best_val_acc:.2f}%")


# ---------------------------------------------------------------------------
# Encrypted Classification Head
# ---------------------------------------------------------------------------

class EncryptedClassificationHead:
    """
    Linear classification head evaluated on CKKS-encrypted features.

    The client encrypts the [CLS] embedding; the server evaluates
    the linear layer without ever seeing the plaintext features.

    FHE depth: 1 level (single matrix multiplication).

    Usage:
        head = EncryptedClassificationHead(feature_dim=192, n_classes=7)
        head.set_weights(trained_linear_layer)

        # Client: encrypt features
        ctx = head.make_context()
        enc_features = head.encrypt(features, ctx)

        # Server: evaluate head
        enc_logits = head.evaluate(enc_features)

        # Client: decrypt
        logits = head.decrypt(enc_logits, ctx)
    """

    def __init__(self, feature_dim: int, n_classes: int):
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.weight: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None

    def set_weights(self, linear: nn.Linear) -> None:
        """Extract weights from a trained nn.Linear layer."""
        self.weight = linear.weight.detach().numpy()  # (n_classes, feature_dim)
        self.bias = linear.bias.detach().numpy() if linear.bias is not None else None

    def make_context(self):
        """Create a TenSEAL CKKS context for depth-1 computation."""
        try:
            import tenseal as ts
        except ImportError:
            raise ImportError("Install tenseal: pip install tenseal")
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[40, 40, 40],  # 1 level is enough
        )
        ctx.global_scale = 2 ** 40
        ctx.generate_galois_keys()
        return ctx

    def encrypt(self, features: np.ndarray, context) -> object:
        """Encrypt a feature vector (1D numpy array)."""
        import tenseal as ts
        return ts.ckks_vector(context, features.tolist())

    def evaluate(self, enc_features) -> object:
        """
        Server-side: evaluate linear head on encrypted features.
        Result is encrypted logits.
        """
        assert self.weight is not None, "Call set_weights() first"
        enc_logits = enc_features.mm(self.weight.T.tolist())
        if self.bias is not None:
            enc_logits = enc_logits + self.bias.tolist()
        return enc_logits

    def decrypt(self, enc_logits) -> np.ndarray:
        """Decrypt logits and return as numpy array."""
        return np.array(enc_logits.decrypt())


# ---------------------------------------------------------------------------
# Full Benchmark
# ---------------------------------------------------------------------------

def run_medmnist_benchmark(
    dataset_name: str = "breastmnist",
    image_size: int = 224,
    fine_tune_epochs: int = 20,
    n_fhe_samples: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """
    Reproduce MedBlindTuner pipeline on a MedMNIST dataset.

    Steps:
      1. Fine-tune DeiT-Tiny backbone on the target dataset
      2. Extract [CLS] features for the test set
      3. Run encrypted classification head on a subset (FHE is slow)
      4. Compare plaintext vs. encrypted accuracy

    Args:
        dataset_name: MedMNIST dataset key
        n_fhe_samples: Number of test samples to run through FHE head

    Returns:
        Dict with accuracy metrics and timing
    """
    import time

    n_classes = MEDMNIST_DATASETS[dataset_name]["n_classes"]
    print(f"\n=== MedBlindTuner Baseline: {dataset_name} ({n_classes} classes) ===\n")

    # 1. Load data
    train_loader, val_loader, test_loader = get_medmnist_loaders(
        dataset_name, batch_size=64, image_size=image_size
    )

    # 2. Fine-tune backbone
    print("Step 1: Fine-tuning DeiT-Tiny backbone...")
    extractor = MedViTFeatureExtractor(pretrained=True, n_classes=n_classes)
    extractor.fine_tune(train_loader, val_loader, epochs=fine_tune_epochs, device=device)

    # 3. Plaintext accuracy on test set
    print("\nStep 2: Plaintext test accuracy...")
    extractor.eval().to(device)
    correct = total = 0
    all_features = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y_int = y.squeeze().long()
            preds = extractor(x).argmax(-1).cpu()
            correct += (preds == y_int).sum().item()
            total += y_int.size(0)
            # Save features for FHE eval
            feats = extractor.extract_features(x).cpu().numpy()
            all_features.append(feats)
            all_labels.append(y_int.numpy())

    plaintext_acc = 100 * correct / total
    print(f"Plaintext test accuracy: {plaintext_acc:.2f}%")

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 4. Set up encrypted head
    print(f"\nStep 3: Running FHE head on {n_fhe_samples} samples...")
    enc_head = EncryptedClassificationHead(
        feature_dim=extractor.feature_dim, n_classes=n_classes
    )
    enc_head.set_weights(extractor.head)
    ctx = enc_head.make_context()

    fhe_times = []
    fhe_correct = 0

    for i in range(min(n_fhe_samples, len(all_features))):
        feat = all_features[i]
        label = all_labels[i]

        t0 = time.perf_counter()
        enc_feat = enc_head.encrypt(feat, ctx)
        enc_logits = enc_head.evaluate(enc_feat)
        logits = enc_head.decrypt(enc_logits)
        fhe_times.append(time.perf_counter() - t0)

        pred = np.argmax(logits)
        if pred == label:
            fhe_correct += 1

    fhe_acc = 100 * fhe_correct / n_fhe_samples
    mean_fhe_time = np.mean(fhe_times)

    results = {
        "dataset": dataset_name,
        "n_classes": n_classes,
        "plaintext_accuracy": plaintext_acc,
        "fhe_accuracy": fhe_acc,
        "fhe_n_samples": n_fhe_samples,
        "mean_fhe_latency_s": mean_fhe_time,
        "fhe_depth": 1,
        "protocol": "plaintext backbone + encrypted head",
    }

    print(f"\n{'='*50}")
    print(f"Dataset:           {dataset_name}")
    print(f"Plaintext accuracy:{plaintext_acc:.2f}%")
    print(f"FHE accuracy:      {fhe_acc:.2f}% (n={n_fhe_samples})")
    print(f"Mean FHE latency:  {mean_fhe_time:.3f}s")
    print(f"FHE depth:         1 (encrypted head only)")
    print(f"{'='*50}\n")

    return results


if __name__ == "__main__":
    # Run on BreastMNIST (small, binary — fastest to iterate on)
    results = run_medmnist_benchmark(
        dataset_name="breastmnist",
        fine_tune_epochs=10,
        n_fhe_samples=20,
    )
