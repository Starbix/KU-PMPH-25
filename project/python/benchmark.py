#!/usr/bin/env python3
# benchmark.py - Benchmarking script for attention mechanisms

import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import attention_benchmark as ab
from typing import Dict, Tuple, List, Any


def generate_test_data(
    batch_size: int, seq_length: int, num_heads: int, head_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate random test data for attention benchmarking."""
    # Generate query, key, and value tensors with appropriate shapes
    q = np.random.normal(0, 1, (batch_size, seq_length, num_heads, head_dim)).astype(
        np.float32
    )
    k = np.random.normal(0, 1, (batch_size, seq_length, num_heads, head_dim)).astype(
        np.float32
    )
    v = np.random.normal(0, 1, (batch_size, seq_length, num_heads, head_dim)).astype(
        np.float32
    )

    return q, k, v


def run_benchmark(
    seq_lengths: List[int],
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    num_runs: int = 10,
    verify: bool = True,
) -> Dict[str, List[Any]]:
    """
    Run benchmarks for different sequence lengths and collect results.

    Args:
        seq_lengths: List of sequence lengths to test
        batch_size: Batch size for the test data
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        num_runs: Number of runs for each benchmark for averaging
        verify: Whether to verify correctness between implementations

    Returns:
        Dictionary containing benchmark results
    """
    results = {
        "seq_lengths": seq_lengths,
        "std_times": [],
        "flash_times": [],
        "speedups": [],
        "max_errors": [],
        "mean_errors": [],
    }

    for seq_length in seq_lengths:
        print(f"Benchmarking sequence length: {seq_length}")

        # Generate data
        q, k, v = generate_test_data(batch_size, seq_length, num_heads, head_dim)

        # Reshape to [batch_size*num_heads, seq_len, head_dim] for the C++ interface
        q_reshaped = q.reshape(batch_size * num_heads, seq_length, head_dim)
        k_reshaped = k.reshape(batch_size * num_heads, seq_length, head_dim)
        v_reshaped = v.reshape(batch_size * num_heads, seq_length, head_dim)

        # Compare implementations
        benchmark_results = ab.compare_implementations(
            q_reshaped, k_reshaped, v_reshaped, num_runs
        )

        # Store results
        results["std_times"].append(benchmark_results["standard_time_ms"])
        results["flash_times"].append(benchmark_results["flash_time_ms"])
        results["speedups"].append(benchmark_results["speedup"])
        results["max_errors"].append(benchmark_results["max_abs_error"])
        results["mean_errors"].append(benchmark_results["mean_abs_error"])

        print(f"  Standard Attention: {benchmark_results['standard_time_ms']:.4f} ms")
        print(f"  Flash Attention:    {benchmark_results['flash_time_ms']:.4f} ms")
        print(f"  Speedup:            {benchmark_results['speedup']:.2f}x")

        if verify:
            print(f"  Max Error:          {benchmark_results['max_abs_error']:.6e}")
            print(f"  Mean Error:         {benchmark_results['mean_abs_error']:.6e}")

        print()

    return results


def plot_results(results: Dict[str, List[Any]], output_path: str = None):
    """Plot benchmark results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot execution times
    ax1.plot(
        results["seq_lengths"], results["std_times"], "o-", label="Standard Attention"
    )
    ax1.plot(
        results["seq_lengths"], results["flash_times"], "s-", label="Flash Attention"
    )
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Execution Time (ms)")
    ax1.set_title("Attention Performance Comparison")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    ax1.legend()

    # Plot speedup
    ax2.plot(results["seq_lengths"], results["speedups"], "D-", color="green")
    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Speedup (x)")
    ax2.set_title("Flash Attention Speedup")
    ax2.set_xscale("log", base=2)
    ax2.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Benchmark attention mechanisms")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--head_dim", type=int, default=64, help="Dimension of each attention head"
    )
    parser.add_argument(
        "--num_runs", type=int, default=10, help="Number of runs for each benchmark"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify correctness between implementations",
    )
    parser.add_argument("--output", type=str, help="Output path for the benchmark plot")
    args = parser.parse_args()

    # Define sequence lengths to benchmark
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]

    # Run benchmarks
    results = run_benchmark(
        seq_lengths=seq_lengths,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        num_runs=args.num_runs,
        verify=args.verify,
    )

    # Plot results
    plot_results(results, args.output)

    # Print summary
    print("Summary:")
    print(
        f"Maximum speedup: {max(results['speedups']):.2f}x at sequence length {results['seq_lengths'][results['speedups'].index(max(results['speedups']))]}"
    )


if __name__ == "__main__":
    main()
