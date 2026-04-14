"""
Quick verification: Test all polynomial replacements before full run.
Run this FIRST — it takes 30 seconds and catches any timm version issues.

Usage: CUDA_VISIBLE_DEVICES=2 python3 verify_poly_replacements.py
"""

import torch
import torch.nn as nn
import timm
import types
import sys

print("=" * 60)
print("  Verifying polynomial replacements on your timm version")
print(f"  timm version: {timm.__version__}")
print(f"  PyTorch version: {torch.__version__}")
print("=" * 60)

# Create model
model = timm.create_model("deit_tiny_patch16_224", pretrained=True, num_classes=5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Test 1: Verify attention structure
print("\n── Test 1: Attention module structure ──")
attn = model.blocks[0].attn
print(f"  Type: {type(attn).__name__}")
print(f"  num_heads: {attn.num_heads}")
print(f"  head_dim: {attn.head_dim if hasattr(attn, 'head_dim') else 'N/A'}")
print(f"  scale: {attn.scale:.6f}")
print(f"  fused_attn: {attn.fused_attn if hasattr(attn, 'fused_attn') else 'N/A'}")
print(f"  qkv: {attn.qkv}")
print(f"  proj: {attn.proj}")

# Determine head_dim
if hasattr(attn, 'head_dim'):
    head_dim = attn.head_dim
else:
    head_dim = attn.qkv.in_features // attn.num_heads
print(f"  Resolved head_dim: {head_dim}")

# Test 2: Forward pass with original model
print("\n── Test 2: Original model forward pass ──")
x = torch.randn(2, 3, 224, 224).to(device)
with torch.no_grad():
    out = model(x)
print(f"  Input: {list(x.shape)} → Output: {list(out.shape)}")
print(f"  Output sample: {out[0, :3].cpu().tolist()}")

# Test 3: Replace GELU
print("\n── Test 3: GELU replacement ──")
sys.path.insert(0, '.')
from step34_full_poly_kd import PolyActivation, replace_gelu
n_gelu = replace_gelu(model, degree=2)
print(f"  Replaced {n_gelu} GELU modules")
assert n_gelu == 12, f"Expected 12 GELU replacements, got {n_gelu}"
print(f"  ✓ GELU replacement OK")

# Test 4: Replace softmax
print("\n── Test 4: Softmax replacement ──")
from step34_full_poly_kd import replace_attention_softmax, poly_softmax
n_softmax = replace_attention_softmax(model, depth=3)
print(f"  Replaced {n_softmax} attention softmax")
assert n_softmax == 12, f"Expected 12 softmax replacements, got {n_softmax}"
print(f"  ✓ Softmax replacement OK")

# Test 5: Replace LayerNorm
print("\n── Test 5: LayerNorm replacement ──")
from step34_full_poly_kd import replace_layernorms, PolyLayerNorm
n_ln = replace_layernorms(model)
print(f"  Replaced {n_ln} LayerNorm modules")
assert n_ln >= 25, f"Expected ≥25 LayerNorm replacements, got {n_ln}"
print(f"  ✓ LayerNorm replacement OK")

# Test 6: Verify no non-polynomial ops remain
print("\n── Test 6: Verify fully polynomial ──")
from step34_full_poly_kd import verify_no_nonpoly_ops
is_poly = verify_no_nonpoly_ops(model)
assert is_poly, "Non-polynomial operations still present!"
print(f"  ✓ All non-polynomial ops replaced")

# Test 7: Forward pass with fully polynomial model
print("\n── Test 7: Polynomial model forward pass ──")
model.eval()
with torch.no_grad():
    out_poly = model(x)
print(f"  Input: {list(x.shape)} → Output: {list(out_poly.shape)}")
print(f"  Output sample: {out_poly[0, :3].cpu().tolist()}")
print(f"  Output is finite: {torch.isfinite(out_poly).all().item()}")
print(f"  Output has no NaN: {not torch.isnan(out_poly).any().item()}")

# Test 8: Backward pass (training works)
print("\n── Test 8: Backward pass (gradient flow) ──")
model.train()
out_train = model(x)
loss = out_train.sum()
loss.backward()
# Check at least some gradients are non-zero
grad_count = 0
nonzero_grad = 0
for name, p in model.named_parameters():
    if p.requires_grad and p.grad is not None:
        grad_count += 1
        if p.grad.abs().max() > 0:
            nonzero_grad += 1
print(f"  Parameters with gradients: {grad_count}")
print(f"  Parameters with non-zero gradients: {nonzero_grad}")
assert nonzero_grad > 0, "No non-zero gradients!"
print(f"  ✓ Gradient flow OK")

# Test 9: poly_softmax correctness
print("\n── Test 9: poly_softmax accuracy ──")
test_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]]).to(device)
real_sm = torch.softmax(test_input, dim=-1)
for depth in [2, 3, 4]:
    poly_sm = poly_softmax(test_input, depth=depth, dim=-1)
    max_err = (real_sm - poly_sm).abs().max().item()
    print(f"  Depth {depth}: max error = {max_err:.6f}")
print(f"  ✓ poly_softmax working")

print("\n" + "=" * 60)
print("  ALL TESTS PASSED — safe to run full experiment")
print("=" * 60)
print(f"\n  Run: CUDA_VISIBLE_DEVICES=2 python3 step34_full_poly_kd.py --dataset all --baseline_dir results")