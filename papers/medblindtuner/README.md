# MedBlindTuner (2024)

> "MedBlindTuner: Towards Privacy-Preserving Fine-Tuning on Biomedical Images via Blind Transformers"

## Why This is the Baseline to Beat

MedBlindTuner is the **closest existing work** to our research goal: private ViT inference on medical imaging data. It combines:
1. DeiT backbone (pre-trained)
2. CKKS encryption for privacy
3. MedMNIST benchmark datasets

It uses a **hybrid approach**: plaintext DeiT backbone + encrypted classification head. This is cheaper than full FHE inference but provides weaker privacy (server sees intermediate features, not input).

Our goal: beat MedBlindTuner's accuracy with a stronger privacy guarantee (fully encrypted backbone).

## Protocol

```
Client side:
  1. Run DeiT backbone on raw image (plaintext)
  2. Encrypt the [CLS] token embedding → ciphertext
  3. Send ciphertext to server

Server side:
  4. Compute linear classification head on ciphertext (CKKS matmul)
  5. Return encrypted logits

Client side:
  6. Decrypt logits → prediction
```

**Privacy model**: server sees encrypted [CLS] features. The backbone runs on the client.
This is "feature-level privacy" not "input-level privacy".

**Contrast with full FHE**: full FHE would send the encrypted image and run the entire backbone on the server — stronger privacy, higher cost.

## MedMNIST Datasets Used

| Dataset | Task | Classes | Train | Test |
|---------|------|---------|-------|------|
| DermaMNIST | Skin lesion classification | 7 | 7007 | 2005 |
| BreastMNIST | Breast ultrasound malignancy | 2 | 546 | 156 |
| PneumoniaMNIST | Chest X-ray pneumonia | 2 | 4708 | 624 |
| BloodMNIST | Blood cell type | 8 | 11959 | 3421 |
| RetinaMNIST | Diabetic retinopathy | 5 | 1080 | 400 |

## Results (from paper)

| Dataset | DeiT-Tiny (plaintext) | MedBlindTuner | Our target |
|---------|-----------------------|---------------|------------|
| DermaMNIST | 73.2% | 71.8% | > 72% (full FHE) |
| BreastMNIST | 83.1% | 82.1% | > 82% (full FHE) |
| PneumoniaMNIST | 93.2% | 92.8% | > 92% (full FHE) |

## NAG Optimizer for Encrypted Training

MedBlindTuner uses **NAG (Nesterov Accelerated Gradient)** in the encrypted domain for the classification head:
```
v_{t+1} = μ · v_t + ∇L(w_t + μ · v_t)    (look-ahead step, encrypted)
w_{t+1} = w_t - α · v_{t+1}
```

This allows fine-tuning the encrypted head weights without ever decrypting — useful for on-device personalization.

For our implementation: we use standard plaintext NAG (simplified), since we focus on inference rather than encrypted training.

## Implementation

See [`implementation.py`](implementation.py) for:
- `MedViTFeatureExtractor` — DeiT backbone in plaintext, outputs [CLS] features
- `EncryptedClassificationHead` — linear head applied to CKKS-encrypted features (TenSEAL)
- `run_medmnist_benchmark()` — reproduce MedBlindTuner results on DermaMNIST + BreastMNIST

## Path Beyond MedBlindTuner

Our research extends MedBlindTuner by:
1. Moving encryption boundary from features to input patches
2. Using KD-trained polynomial ViT (DeiT-Tiny + PolyGELU) for full FHE backbone
3. Targeting same MedMNIST accuracy with stronger privacy

Concretely:
```
MedBlindTuner (this paper):
  client: DeiT backbone (plaintext) → encrypted features
  server: linear head on ciphertext

Our target:
  client: encrypt raw patches
  server: poly-DeiT full forward pass on ciphertexts
```

## Open Questions

1. How much accuracy is lost moving from feature-level to input-level privacy?
2. Can KD from a full DeiT teacher close the accuracy gap for the polynomial backbone?
3. Which MedMNIST datasets are most sensitive to polynomial activation approximation error?
4. Is BreastMNIST (binary, small dataset) a good first target for full FHE inference?
