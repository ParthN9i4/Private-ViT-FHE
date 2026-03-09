"""
CryptoNets reimplementation using TenSEAL (CKKS).

Original paper used BFV for integer arithmetic.
We use CKKS for floating-point, which is more relevant to our ViT work.

Paper: https://proceedings.mlr.press/v48/gilad-bachrach16.html
"""

import numpy as np
import torch
import torch.nn as nn
import tenseal as ts
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Step 1: Plaintext model with square activation
# ---------------------------------------------------------------------------

class SquareActivation(nn.Module):
    """f(x) = x^2 — polynomial, depth 1."""
    def forward(self, x):
        return x * x


class CryptoNets(nn.Module):
    """
    CryptoNets architecture for MNIST.
    Uses square activation throughout — FHE-friendly.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5, stride=2, padding=0)
        self.act1 = SquareActivation()
        self.conv2 = nn.Conv2d(5, 50, kernel_size=5, stride=2, padding=0)
        self.act2 = SquareActivation()
        self.fc1 = nn.Linear(50 * 4 * 4, 100)  # adjust based on input size
        self.act3 = SquareActivation()
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = x.flatten(1)
        x = self.act3(self.fc1(x))
        return self.fc2(x)


def train_cryptonets(epochs: int = 10, batch_size: int = 128) -> CryptoNets:
    """
    Train CryptoNets on MNIST.

    TODO: Fill in training loop.
    """
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_data = datasets.MNIST("data/", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = CryptoNets()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    return model


# ---------------------------------------------------------------------------
# Step 2: FHE inference using TenSEAL
# ---------------------------------------------------------------------------

def make_fhe_context() -> ts.Context:
    """
    CKKS context for CryptoNets.
    Depth 2 (two square activations) → very small parameters.
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[40, 40, 40, 40],  # 2 levels + head + tail
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context


def fhe_square(enc_vec: ts.CKKSVector) -> ts.CKKSVector:
    """FHE-compatible square activation: consumes 1 multiplicative level."""
    return enc_vec * enc_vec


def fhe_linear(enc_vec: ts.CKKSVector, weight: np.ndarray, bias: np.ndarray) -> ts.CKKSVector:
    """
    FHE matrix-vector product.

    For simplicity uses TenSEAL's built-in mm_ (matrix multiplication).
    In production, use diagonal encoding (BSGS) for better performance.
    """
    return enc_vec.mm(weight.tolist()) + bias.tolist()


def fhe_inference_fc_only(
    model: CryptoNets,
    image: np.ndarray,
    context: ts.Context,
) -> List[float]:
    """
    Run FHE inference on a single MNIST image.

    Simplified: only encrypts the FC portion (conv layers run in plaintext).
    This is a common intermediate step before full FHE.

    Args:
        model: Trained CryptoNets model
        image: 28x28 numpy array (normalized)
        context: TenSEAL context

    Returns:
        Decrypted logits (10 values)
    """
    model.eval()
    with torch.no_grad():
        # Run conv layers in plaintext
        x = torch.tensor(image).float().unsqueeze(0).unsqueeze(0)
        x = model.act1(model.conv1(x))
        x = model.act2(model.conv2(x))
        x = x.flatten().numpy()

    # Encrypt flattened features
    enc_x = ts.ckks_vector(context, x.tolist())

    # FC1: linear + square
    w1 = model.fc1.weight.detach().numpy()
    b1 = model.fc1.bias.detach().numpy()
    enc_x = fhe_linear(enc_x, w1, b1)
    enc_x = fhe_square(enc_x)

    # FC2: linear (no activation — final layer)
    w2 = model.fc2.weight.detach().numpy()
    b2 = model.fc2.bias.detach().numpy()
    enc_x = fhe_linear(enc_x, w2, b2)

    return enc_x.decrypt()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(model: CryptoNets, n_images: int = 10):
    """Time FHE inference vs plaintext inference."""
    import time
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_data = datasets.MNIST("data/", train=False, download=True, transform=transform)
    context = make_fhe_context()

    times = []
    correct_fhe = 0
    correct_plain = 0

    for i in range(n_images):
        image, label = test_data[i]
        image_np = image.squeeze().numpy()

        # Plaintext
        with torch.no_grad():
            logits_plain = model(image.unsqueeze(0)).squeeze().numpy()
        pred_plain = np.argmax(logits_plain)

        # FHE
        t0 = time.time()
        logits_fhe = fhe_inference_fc_only(model, image_np, context)
        t1 = time.time()

        pred_fhe = np.argmax(logits_fhe)
        times.append(t1 - t0)

        if pred_plain == label:
            correct_plain += 1
        if pred_fhe == label:
            correct_fhe += 1

    print(f"\nCryptoNets Benchmark ({n_images} images, FC-only FHE)")
    print(f"  Plaintext accuracy:   {correct_plain}/{n_images}")
    print(f"  FHE accuracy:         {correct_fhe}/{n_images}")
    print(f"  Mean FHE time:        {np.mean(times):.3f}s")
    print(f"  Max FHE time:         {np.max(times):.3f}s")


if __name__ == "__main__":
    print("Training CryptoNets on MNIST...")
    model = train_cryptonets(epochs=5)
    benchmark(model, n_images=10)
