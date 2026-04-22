"""
replot_bloodmnist_mechanism.py
===============================
Reads the existing JSON from investigate_collapse_bloodmnist.py and
produces publication-ready figures. No retraining.

Adjusted for the BloodMNIST result: Config E shows activation COLLAPSE
(not explosion). The figure structure now highlights:
  (a) per-layer norms at epoch 10 -- shows the collapse-to-constant shape
  (b) layer-5 norm over training -- shows the frozen state
  (c) gradient norms -- shows the "gradients die" signature

USAGE
-----
  pip install matplotlib
  python replot_bloodmnist_mechanism.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_PATH = "./experiment_results/investigate_collapse_bloodmnist.json"
FIG_DIR = "./figures"


def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"[ERROR] JSON not found: {RESULTS_PATH}")
        return

    with open(RESULTS_PATH) as f:
        data = json.load(f)
    logs = data["logs"]

    os.makedirs(FIG_DIR, exist_ok=True)

    colors = {"Config_D_LN": "tab:green",
              "Config_E_BN": "tab:red",
              "Fix2_Normalized": "tab:blue"}
    labels = {"Config_D_LN": "Config D (LayerNorm, works)",
              "Config_E_BN": "Config E (BatchNorm, collapses)",
              "Fix2_Normalized": "Fix 2 (row-normalized)"}
    markers = {"Config_D_LN": "s", "Config_E_BN": "o", "Fix2_Normalized": "^"}

    # ------------------------------------------------------------------
    # MAIN FIGURE: 3-panel mechanism figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Find closest available epoch to 10
    target_ep = 10
    epochs_available = logs["Config_D_LN"]["epoch"]
    probe_idx = epochs_available.index(target_ep) if target_ep in epochs_available \
                else len(epochs_available) // 2
    actual_probe_ep = epochs_available[probe_idx]

    # (a) per-layer norms at epoch 10
    ax = axes[0]
    for name, log in logs.items():
        y = log["attn_mean"][probe_idx]
        ax.plot(range(len(y)), y, marker=markers[name], color=colors[name],
                label=labels[name], linewidth=2, markersize=8)
    ax.set_xlabel("Transformer block (layer index)", fontsize=11)
    ax.set_ylabel(f"Mean attention output L2 norm", fontsize=11)
    ax.set_yscale("log")
    ax.set_title(f"(a) Through-depth norms at epoch {actual_probe_ep}", fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower left")

    # (b) L5 norm over training
    ax = axes[1]
    for name, log in logs.items():
        y = [layer_norms[-1] for layer_norms in log["attn_mean"]]
        ax.plot(log["epoch"], y, marker=markers[name], color=colors[name],
                label=labels[name], linewidth=2, markersize=6)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("L2 norm at layer 5", fontsize=11)
    ax.set_yscale("log")
    ax.set_title("(b) Layer 5 attention over training", fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="center right")

    # (c) L5 gradient norm over training -- captures the "gradients die" claim
    ax = axes[2]
    for name, log in logs.items():
        # Skip epoch 0 (init, grads are zero)
        epochs = log["epoch"][1:]
        y = [gn[-1] for gn in log["grad_norm"][1:]]
        y = [max(v, 1e-8) for v in y]  # floor for log scale
        ax.plot(epochs, y, marker=markers[name], color=colors[name],
                label=labels[name], linewidth=2, markersize=6)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Gradient L2 norm at layer 5", fontsize=11)
    ax.set_yscale("log")
    ax.set_title("(c) Gradients follow activations", fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="center right")

    fig.suptitle("Collapse mechanism on BloodMNIST", fontsize=13, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "collapse_mechanism_bloodmnist.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Main figure -> {out_path}")

    # ------------------------------------------------------------------
    # SUPPLEMENTARY: val_bal and train_loss trajectories
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for name, log in logs.items():
        epochs = log["epoch"][1:]
        y = [v * 100 for v in log["val_bal"][1:]]
        ax.plot(epochs, y, marker=markers[name], color=colors[name],
                label=labels[name], linewidth=2, markersize=5)
    ax.axhline(12.5, color="gray", linestyle=":", alpha=0.6,
               label="Random chance (12.5%)")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Val balanced accuracy (%)", fontsize=11)
    ax.set_title("Learning dynamics", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    for name, log in logs.items():
        epochs = log["epoch"][1:]
        y = log["train_loss"][1:]
        ax.plot(epochs, y, marker=markers[name], color=colors[name],
                label=labels[name], linewidth=2, markersize=5)
    ax.axhline(np.log(8), color="gray", linestyle=":", alpha=0.6,
               label=r"$\ln(8) \approx 2.08$ (8-class random)")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Train loss", fontsize=11)
    ax.set_title("Training loss", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "collapse_dynamics_bloodmnist.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Dynamics figure -> {out_path}")

    # ------------------------------------------------------------------
    # DIAGNOSTIC TABLE (printed): per-layer numbers at key epochs
    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("Attention norm ratio: Config E vs Config D (higher E/D = worse)")
    print("=" * 74)
    for ep_target in [1, 5, 10, 20]:
        if ep_target not in epochs_available:
            continue
        idx = epochs_available.index(ep_target)
        print(f"\nEpoch {ep_target}:")
        print(f"  Layer  |  D (LN)    |  E (BN)    |  E/D    |  Fix2  |  Fix2/D")
        print(f"  -------+------------+------------+---------+--------+--------")
        for L in range(6):
            d = logs["Config_D_LN"]["attn_mean"][idx][L]
            e = logs["Config_E_BN"]["attn_mean"][idx][L]
            f2 = logs["Fix2_Normalized"]["attn_mean"][idx][L]
            ed = e / d if d > 0 else float("inf")
            fd = f2 / d if d > 0 else float("inf")
            print(f"  {L}      |  {d:8.1f}  |  {e:8.1f}  |  {ed:6.3f} |  {f2:5.1f} |  {fd:6.3f}")


if __name__ == "__main__":
    main()