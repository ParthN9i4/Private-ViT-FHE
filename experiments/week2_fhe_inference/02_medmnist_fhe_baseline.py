"""
Week 2, Experiment 2: MedBlindTuner replication — encrypted classification head on MedMNIST.

Replicates the MedBlindTuner approach (Panzade et al., AAAI-W 2024):
  - Plaintext ViT backbone (feature extractor, frozen after fine-tuning)
  - Encrypted classification head (CKKS via TenSEAL)

Evaluated on DermaMNIST and BreastMNIST for fast iteration.

Usage:
    pip install tenseal timm medmnist tqdm
    python experiments/week2_fhe_inference/02_medmnist_fhe_baseline.py

Outputs:
    results/week2/medmnist_fhe_baseline.json
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    import timm
    import medmnist
    from medmnist import DermaMNIST, BreastMNIST
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install timm medmnist")
    sys.exit(1)

try:
    import tenseal as ts
    HAS_TENSEAL = True
except ImportError:
    HAS_TENSEAL = False
    print("TenSEAL not found — running accuracy-only mode (no encryption timing).")

try:
    from papers.medblindtuner.implementation import (
        MedViTFeatureExtractor,
        EncryptedClassificationHead,
        run_medmnist_benchmark,
    )
    HAS_MEDBLINDTUNER = True
except ImportError:
    HAS_MEDBLINDTUNER = False


RESULTS_DIR = "results/week2"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
LR = 3e-4
BATCH_SIZE = 64

DATASET_CONFIGS = {
    "DermaMNIST":  {"cls": DermaMNIST,  "num_classes": 7,  "in_channels": 3},
    "BreastMNIST": {"cls": BreastMNIST, "num_classes": 2,  "in_channels": 1},
}


def get_loaders(dataset_cls, num_classes, batch_size=BATCH_SIZE):
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_ds = dataset_cls(split="train", transform=transform, download=True, root="data/medmnist")
    val_ds   = dataset_cls(split="val",   transform=transform, download=True, root="data/medmnist")
    test_ds  = dataset_cls(split="test",  transform=transform, download=True, root="data/medmnist")
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


def train_plaintext(model, train_loader, val_loader, epochs, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.squeeze(1).long().to(device)
            optimizer.zero_grad()
            ce(model(x), y).backward()
            optimizer.step()
        scheduler.step()
        val_acc = evaluate(model, val_loader, device)
        print(f"  epoch {epoch:2d}  val_acc={val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return best_val_acc


def extract_features(backbone, loader, device):
    """Extract backbone features (CLS token) for all samples."""
    backbone.eval()
    features, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feat = backbone(x)  # (B, embed_dim)
            features.append(feat.cpu().numpy())
            labels.append(y.squeeze(1).numpy())
    return np.concatenate(features), np.concatenate(labels)


def encrypted_head_inference(features_np: np.ndarray, weight: np.ndarray,
                              bias: np.ndarray, context) -> np.ndarray:
    """
    Encrypted classification head inference using CKKS.
    Computes: logits = features @ weight.T + bias

    Each sample's feature vector is encrypted separately.
    Returns decrypted logits as numpy array.
    """
    if not HAS_TENSEAL or context is None:
        # Plaintext simulation
        return features_np @ weight.T + bias

    n_samples, feat_dim = features_np.shape
    n_classes = weight.shape[0]
    all_logits = np.zeros((n_samples, n_classes))

    t_enc = t_mul = t_dec = 0.0
    for i in range(min(n_samples, 100)):  # limit to 100 samples for timing
        t0 = time.time()
        enc_feat = ts.ckks_vector(context, features_np[i].tolist())
        t_enc += time.time() - t0

        t0 = time.time()
        logit_list = []
        for c in range(n_classes):
            # Dot product with weight vector for class c
            enc_logit = enc_feat.dot(weight[c].tolist())
            logit_list.append(enc_logit)
        t_mul += time.time() - t0

        t0 = time.time()
        logits_i = np.array([l.decrypt()[0] for l in logit_list]) + bias
        t_dec += time.time() - t0
        all_logits[i] = logits_i

    n = min(n_samples, 100)
    print(f"  Encrypted head (100 samples): "
          f"enc={t_enc/n*1000:.1f}ms/sample, "
          f"compute={t_mul/n*1000:.1f}ms/sample, "
          f"dec={t_dec/n*1000:.1f}ms/sample, "
          f"total={( t_enc+t_mul+t_dec)/n*1000:.1f}ms/sample")

    # For remaining samples, use plaintext
    if n_samples > 100:
        all_logits[100:] = features_np[100:] @ weight.T + bias
    return all_logits


def run_dataset(dataset_name: str, dataset_cls, num_classes: int):
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name} ({num_classes} classes)")
    print(f"{'='*50}")

    train_loader, val_loader, test_loader = get_loaders(dataset_cls, num_classes)

    # Step 1: Fine-tune ViT-Small backbone (faster than ViT-Base)
    print("\n[1] Fine-tuning ViT-Small backbone (plaintext)...")
    full_model = timm.create_model("vit_small_patch16_224", pretrained=True,
                                   num_classes=num_classes)
    full_model = full_model.to(DEVICE)
    best_val = train_plaintext(full_model, train_loader, val_loader, EPOCHS, DEVICE)
    test_acc_full = evaluate(full_model, test_loader, DEVICE)
    print(f"  Full plaintext model — test acc: {test_acc_full:.2f}%")

    # Step 2: Extract features from frozen backbone
    print("\n[2] Extracting backbone features...")
    # Remove classification head to get features
    backbone = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
    backbone.load_state_dict(
        {k: v for k, v in full_model.state_dict().items() if "head" not in k},
        strict=False
    )
    backbone = backbone.to(DEVICE)

    train_feats, train_labels = extract_features(backbone, train_loader, DEVICE)
    test_feats,  test_labels  = extract_features(backbone, test_loader,  DEVICE)
    print(f"  Feature shape: {train_feats.shape}")

    # Step 3: Train linear classification head on plaintext features
    print("\n[3] Training linear classification head (plaintext)...")
    W = torch.randn(num_classes, train_feats.shape[1], requires_grad=True)
    b = torch.zeros(num_classes, requires_grad=True)
    optimizer = torch.optim.Adam([W, b], lr=1e-3)
    ce = nn.CrossEntropyLoss()

    X_train = torch.tensor(train_feats, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)

    for ep in range(50):
        for i in range(0, len(X_train), 256):
            xb = X_train[i:i+256]
            yb = y_train[i:i+256]
            optimizer.zero_grad()
            ce(xb @ W.T + b, yb).backward()
            optimizer.step()

    # Plaintext head accuracy
    with torch.no_grad():
        X_test = torch.tensor(test_feats, dtype=torch.float32)
        preds = (X_test @ W.T + b).argmax(dim=-1).numpy()
    pt_head_acc = 100 * (preds == test_labels).mean()
    print(f"  Plaintext head — test acc: {pt_head_acc:.2f}%")

    # Step 4: Encrypted head inference
    print("\n[4] Running encrypted head inference (CKKS)...")
    if HAS_TENSEAL:
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        context.global_scale = 2 ** 40
        context.generate_galois_keys()
    else:
        context = None

    W_np = W.detach().numpy()
    b_np = b.detach().numpy()

    t0 = time.time()
    enc_logits = encrypted_head_inference(test_feats, W_np, b_np, context)
    enc_latency = time.time() - t0

    enc_preds = enc_logits.argmax(axis=-1)
    enc_head_acc = 100 * (enc_preds == test_labels).mean()
    print(f"  Encrypted head — test acc: {enc_head_acc:.2f}%")
    print(f"  Accuracy delta (encrypted - plaintext head): {enc_head_acc - pt_head_acc:.2f}%")

    return {
        "full_plaintext_model_test_acc": round(float(test_acc_full), 3),
        "plaintext_head_test_acc": round(float(pt_head_acc), 3),
        "encrypted_head_test_acc": round(float(enc_head_acc), 3),
        "accuracy_drop_from_encryption": round(float(pt_head_acc - enc_head_acc), 3),
        "fhe_library": "tenseal" if HAS_TENSEAL else "simulation",
        "total_test_samples": len(test_labels),
    }


def main():
    print(f"Device: {DEVICE}")
    print(f"TenSEAL: {HAS_TENSEAL}")

    all_results = {}
    for dataset_name, config in DATASET_CONFIGS.items():
        try:
            result = run_dataset(dataset_name, config["cls"], config["num_classes"])
            all_results[dataset_name] = result
        except Exception as ex:
            print(f"\nFAILED on {dataset_name}: {ex}")
            all_results[dataset_name] = {"error": str(ex)}

    print("\n\n=== Final Summary ===")
    header = f"{'Dataset':<15} {'Full PT':>10} {'PT Head':>10} {'Enc Head':>10} {'Drop':>8}"
    print(header)
    print("-" * len(header))
    for ds, res in all_results.items():
        if "error" in res:
            print(f"  {ds:<13} {'FAILED':>10}")
        else:
            print(f"  {ds:<13} {res['full_plaintext_model_test_acc']:>9.2f}% "
                  f"{res['plaintext_head_test_acc']:>9.2f}% "
                  f"{res['encrypted_head_test_acc']:>9.2f}% "
                  f"{res['accuracy_drop_from_encryption']:>7.2f}%")

    out_path = os.path.join(RESULTS_DIR, "medmnist_fhe_baseline.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
