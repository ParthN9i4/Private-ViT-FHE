"""
Experiment: GELU approximation accuracy vs multiplicative depth trade-off.

This is one of the core experiments for private ViT inference.
Higher degree → better accuracy but more levels consumed.

Run this before implementing FHE inference to understand
what degree is acceptable for your accuracy target.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from utils.poly_approx import chebyshev_approx, minimax_approx, gelu, approx_error, poly_depth


DEGREES = [3, 4, 5, 7, 10, 15, 20, 27]
DOMAIN = (-6, 6)  # typical range of pre-activation values in transformers


def run_approximation_study():
    print("GELU Polynomial Approximation Study")
    print("=" * 60)
    print(f"Domain: {DOMAIN[0]} to {DOMAIN[1]}")
    print(f"{'Degree':>8} {'Depth':>6} {'Max Error':>12} {'Mean Error':>12}")
    print("-" * 60)

    results = []
    for degree in DEGREES:
        coeffs = chebyshev_approx(gelu, degree, DOMAIN)
        metrics = approx_error(gelu, coeffs, DOMAIN)
        results.append({
            "degree": degree,
            "depth": poly_depth(degree),
            "max_abs_error": metrics["max_abs_error"],
            "mean_abs_error": metrics["mean_abs_error"],
            "coeffs": coeffs,
        })
        print(
            f"{degree:>8} {poly_depth(degree):>6} "
            f"{metrics['max_abs_error']:>12.2e} "
            f"{metrics['mean_abs_error']:>12.2e}"
        )

    print("=" * 60)
    print("\nNote: Max error < 0.01 is typically acceptable for classification tasks.")
    print("Degree 4 (depth 2) is used in BOLT; Degree 27 (depth 5) in Iron.\n")

    return results


def plot_approximations(results, save_path="gelu_approx_comparison.png"):
    """Plot true GELU vs polynomial approximations."""
    x = np.linspace(*DOMAIN, 1000)
    y_true = gelu(x)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, res in enumerate(results):
        ax = axes[i]
        y_approx = sum(c * x**j for j, c in enumerate(res["coeffs"]))
        error = np.abs(y_true - y_approx)

        ax.plot(x, y_true, "b-", label="True GELU", alpha=0.7)
        ax.plot(x, y_approx, "r--", label=f"Degree {res['degree']}", alpha=0.9)
        ax.set_title(
            f"Degree {res['degree']} | Depth {res['depth']}\n"
            f"Max err: {res['max_abs_error']:.2e}"
        )
        ax.legend(fontsize=7)
        ax.set_xlim(DOMAIN)
        ax.grid(True, alpha=0.3)

    plt.suptitle("GELU Polynomial Approximations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")


def impact_on_accuracy():
    """
    Theoretical analysis: how much does approximation error
    propagate through a 6-layer ViT?

    Assumes error accumulates additively per layer (conservative).
    """
    print("\nError propagation through 6 ViT layers")
    print("(assumes additive accumulation — conservative estimate)")
    print("=" * 50)

    for degree in [4, 7, 15, 27]:
        coeffs = chebyshev_approx(gelu, degree, DOMAIN)
        metrics = approx_error(gelu, coeffs, DOMAIN)
        per_layer_err = metrics["max_abs_error"]
        total_err = per_layer_err * 6 * 2  # 2 activations per block

        print(
            f"Degree {degree:2d}: per-activation err={per_layer_err:.2e}, "
            f"accumulated over 12 activations: {total_err:.2e}"
        )


if __name__ == "__main__":
    results = run_approximation_study()
    impact_on_accuracy()
    plot_approximations(results)
