"""
Unified benchmarking harness for FHE inference experiments.

Records latency, throughput, and noise statistics across experiments
and outputs structured JSON for comparison.
"""

import json
import time
import platform
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional


@dataclass
class BenchmarkResult:
    name: str
    latency_s: float
    n_runs: int
    mean_latency_s: float
    min_latency_s: float
    max_latency_s: float
    metadata: Dict[str, Any]


def benchmark(
    fn: Callable,
    name: str,
    n_warmup: int = 1,
    n_runs: int = 5,
    metadata: Optional[Dict[str, Any]] = None,
) -> BenchmarkResult:
    """
    Time a callable over multiple runs.

    Args:
        fn: Zero-argument callable to benchmark
        name: Human-readable name for this benchmark
        n_warmup: Number of warmup runs (not recorded)
        n_runs: Number of timed runs
        metadata: Optional dict of extra info (model config, params, etc.)

    Returns:
        BenchmarkResult with timing statistics
    """
    for _ in range(n_warmup):
        fn()

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        latencies.append(time.perf_counter() - t0)

    return BenchmarkResult(
        name=name,
        latency_s=latencies[0],
        n_runs=n_runs,
        mean_latency_s=sum(latencies) / len(latencies),
        min_latency_s=min(latencies),
        max_latency_s=max(latencies),
        metadata=metadata or {},
    )


def get_hardware_info() -> Dict[str, str]:
    """Collect basic hardware and software version info."""
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import tenseal as ts
        info["tenseal_version"] = ts.__version__
    except ImportError:
        info["tenseal_version"] = "not installed"
    try:
        import torch
        info["torch_version"] = torch.__version__
    except ImportError:
        info["torch_version"] = "not installed"
    return info


def save_results(
    results: List[BenchmarkResult],
    path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save benchmark results to a JSON file.

    Args:
        results: List of BenchmarkResult objects
        path: Output file path
        extra: Optional extra fields to include in the top-level JSON
    """
    output = {
        "hardware": get_hardware_info(),
        "results": [asdict(r) for r in results],
    }
    if extra:
        output.update(extra)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {path}")


def print_results(results: List[BenchmarkResult]) -> None:
    """Print a formatted table of benchmark results."""
    header = f"{'Name':<40} {'Mean (s)':>10} {'Min (s)':>10} {'Max (s)':>10} {'Runs':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<40} {r.mean_latency_s:>10.3f} {r.min_latency_s:>10.3f} "
            f"{r.max_latency_s:>10.3f} {r.n_runs:>6}"
        )
