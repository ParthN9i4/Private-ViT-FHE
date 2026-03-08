"""
Polynomial approximation utilities for FHE-friendly activation functions.

The core problem: CKKS only supports polynomial operations.
Non-linear activations (GELU, ReLU, softmax, LayerNorm) must be
approximated by polynomials of bounded degree.

Higher degree = more accuracy but more multiplicative depth.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Callable, Tuple


# ---------------------------------------------------------------------------
# Minimax polynomial approximation via Remez / scipy
# ---------------------------------------------------------------------------

def minimax_approx(
    fn: Callable,
    degree: int,
    domain: Tuple[float, float],
    n_samples: int = 1000,
) -> np.ndarray:
    """
    Compute minimax polynomial approximation of fn on [a, b].

    Uses Chebyshev nodes as initial sample points and minimizes
    the maximum absolute error (L-infinity norm).

    Args:
        fn: Target function (e.g., np.tanh, gelu)
        degree: Polynomial degree
        domain: (a, b) interval to approximate over
        n_samples: Number of evaluation points

    Returns:
        Coefficients [c0, c1, ..., cd] of the approximating polynomial
        such that p(x) = c0 + c1*x + c2*x^2 + ... + cd*x^d
    """
    a, b = domain
    # Chebyshev nodes for better numerical conditioning
    k = np.arange(1, n_samples + 1)
    x = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k - 1) * np.pi / (2 * n_samples))
    y = fn(x)

    def max_error(coeffs):
        p = np.polyval(coeffs[::-1], x)  # numpy uses descending order
        return np.max(np.abs(p - y))

    # Least squares initialization
    coeffs_init = np.polyfit(x, y, degree)[::-1]  # ascending order

    result = minimize(
        max_error,
        coeffs_init,
        method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-10, "fatol": 1e-10},
    )
    return result.x


def chebyshev_approx(
    fn: Callable,
    degree: int,
    domain: Tuple[float, float],
) -> np.ndarray:
    """
    Chebyshev polynomial approximation (fast, good starting point).

    Less optimal than minimax but closed-form and fast.
    Use this to get initial coefficients, then refine with minimax_approx.
    """
    a, b = domain
    # Sample at Chebyshev nodes
    k = np.arange(degree + 1)
    nodes = np.cos((2 * k + 1) * np.pi / (2 * (degree + 1)))
    x_nodes = 0.5 * (a + b) + 0.5 * (b - a) * nodes
    y_nodes = fn(x_nodes)

    # Fit polynomial
    return np.polyfit(x_nodes, y_nodes, degree)[::-1]  # ascending order


# ---------------------------------------------------------------------------
# Standard FHE-friendly activations
# ---------------------------------------------------------------------------

def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit: x * Phi(x)"""
    return x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def approx_gelu_degree4(x: np.ndarray) -> np.ndarray:
    """
    Degree-4 GELU approximation from BOLT paper.
    Valid on domain [-5, 5].
    """
    return 0.5 * x + 0.1972 * x**3 + 0.0012 * x**4


def approx_gelu_degree27(x: np.ndarray) -> np.ndarray:
    """
    Higher-accuracy degree-27 approximation (Iron paper).
    Better accuracy, more multiplicative depth.
    """
    # These coefficients are from the Iron paper
    # Re-derive with minimax_approx(gelu, 27, (-8, 8)) for exact values
    return gelu(x)  # placeholder — use minimax_approx in practice


# ---------------------------------------------------------------------------
# Softmax approximation
# ---------------------------------------------------------------------------

def approx_softmax_domain_decomp(
    x: np.ndarray, degree: int = 7
) -> np.ndarray:
    """
    Piecewise polynomial approximation of softmax.

    Iron paper approach: decompose the domain into segments where
    exp can be accurately approximated by low-degree polynomials.

    Args:
        x: Input logits (already scaled / centered)
        degree: Polynomial degree per segment
    """
    # Subtract max for numerical stability (same as standard softmax)
    x_shifted = x - np.max(x)

    # Approximate exp on each segment
    exp_x = np.exp(x_shifted)  # placeholder: replace with polynomial eval
    return exp_x / np.sum(exp_x)


# ---------------------------------------------------------------------------
# Multiplicative depth counter
# ---------------------------------------------------------------------------

def poly_depth(degree: int) -> int:
    """
    Minimum multiplicative depth to evaluate a polynomial of given degree.

    Uses Paterson-Stockmeyer algorithm: depth = ceil(log2(degree)) for
    standard evaluation; optimal baby-step/giant-step is better.

    Args:
        degree: Polynomial degree

    Returns:
        Minimum number of multiplicative levels needed
    """
    if degree <= 1:
        return 0
    return int(np.ceil(np.log2(degree)))


def eval_polynomial_ckks(coeffs: np.ndarray, x):
    """
    Evaluate polynomial using Horner's method.
    Minimizes the number of multiplications.

    p(x) = c0 + x*(c1 + x*(c2 + ... + x*cd))

    In FHE: each multiplication costs one level.
    Horner's method uses degree multiplications.

    Args:
        coeffs: [c0, c1, ..., cd] in ascending order
        x: Input (float array or CKKS ciphertext)

    Returns:
        Polynomial evaluation result (same type as x)
    """
    result = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        result = result * x + c
    return result


# ---------------------------------------------------------------------------
# Approximation quality metrics
# ---------------------------------------------------------------------------

def approx_error(
    fn: Callable,
    coeffs: np.ndarray,
    domain: Tuple[float, float],
    n_eval: int = 10000,
) -> dict:
    """
    Evaluate approximation quality.

    Returns max absolute error, mean absolute error, and relative error.
    """
    a, b = domain
    x = np.linspace(a, b, n_eval)
    y_true = fn(x)
    y_approx = eval_polynomial_ckks(coeffs, x)

    abs_err = np.abs(y_true - y_approx)
    return {
        "max_abs_error": float(np.max(abs_err)),
        "mean_abs_error": float(np.mean(abs_err)),
        "max_rel_error": float(np.max(abs_err / (np.abs(y_true) + 1e-10))),
        "degree": len(coeffs) - 1,
        "depth": poly_depth(len(coeffs) - 1),
    }


if __name__ == "__main__":
    # Quick demo: approximate GELU on [-5, 5] with degree 4 and degree 15
    print("GELU approximation comparison")
    print("=" * 50)
    for degree in [4, 7, 15, 27]:
        coeffs = chebyshev_approx(gelu, degree, (-5, 5))
        metrics = approx_error(gelu, coeffs, (-5, 5))
        print(
            f"Degree {degree:2d} | Depth {metrics['depth']} | "
            f"Max err: {metrics['max_abs_error']:.2e} | "
            f"Mean err: {metrics['mean_abs_error']:.2e}"
        )
