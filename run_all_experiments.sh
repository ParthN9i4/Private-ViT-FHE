#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Complete Experiment Pipeline: KD + Polynomial ViT + FHE
# Run inside tmux with: bash run_all_experiments.sh
# ═══════════════════════════════════════════════════════════════

set -e  # Stop on any error

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Complete Experiment Pipeline                        ║"
echo "║  Step 1: Plaintext baseline (DeiT-Tiny, 6 datasets) ║"
echo "║  Step 2: Polynomial GELU cold-start                  ║"
echo "║  Step 2v2: Polynomial GELU warm-start (bug-fixed)    ║"
echo "║  Step 5: Knowledge Distillation + Polynomial GELU    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Estimated total time: ~3-4 hours on A6000"
echo "Started at: $(date)"
echo ""

# ── Step 1: Plaintext Baselines ──
echo "═══════════════════════════════════════════════════"
echo "  STEP 1: Plaintext DeiT-Tiny baselines"
echo "  Output: results/"
echo "═══════════════════════════════════════════════════"
python3 baseline_deit_improved.py --dataset all --output_dir results --epochs 30

echo ""
echo "Step 1 complete. Results in results/"
echo "Timestamp: $(date)"
echo ""

# ── Step 2 v1: Cold-Start Polynomial GELU ──
echo "═══════════════════════════════════════════════════"
echo "  STEP 2 v1: Cold-start polynomial GELU"
echo "  Output: results_step2/"
echo "═══════════════════════════════════════════════════"
python3 step2_poly_gelu.py --dataset all --output_dir results_step2 --baseline_dir results --epochs 30

echo ""
echo "Step 2 v1 complete. Results in results_step2/"
echo "Timestamp: $(date)"
echo ""

# ── Step 2 v2: Warm-Start Polynomial GELU (bug-fixed) ──
echo "═══════════════════════════════════════════════════"
echo "  STEP 2 v2: Warm-start polynomial GELU"
echo "  Output: results_step2v2/"
echo "═══════════════════════════════════════════════════"
python3 step2v2_poly_gelu_warmstart.py --dataset all --output_dir results_step2v2 --baseline_dir results --epochs 30

echo ""
echo "Step 2 v2 complete. Results in results_step2v2/"
echo "Timestamp: $(date)"
echo ""

# ── Step 5: Knowledge Distillation ──
echo "═══════════════════════════════════════════════════"
echo "  STEP 5: Knowledge Distillation + Polynomial GELU"
echo "  Output: results_step5_kd/"
echo "═══════════════════════════════════════════════════"
python3 step5_kd_poly_gelu.py --dataset all --output_dir results_step5_kd --baseline_dir results --step2_dir results_step2v2 --epochs 30

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Finished at: $(date)"
echo "═══════════════════════════════════════════════════"
echo ""
echo "Directory structure:"
echo "  results/           → Step 1 plaintext baselines"
echo "  results_step2/     → Step 2 v1 cold-start poly GELU"
echo "  results_step2v2/   → Step 2 v2 warm-start poly GELU"
echo "  results_step5_kd/  → Step 5 KD + poly GELU"