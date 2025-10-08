#!/usr/bin/env python3
# run_benchmark.py - Python script to run the C++ benchmark and visualize results

import argparse
import subprocess
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict, List, Any, Optional, Tuple


def parse_csv_output(output: str) -> Dict[str, List[Any]]:
    """Parse CSV output from the C++ benchmark."""
    lines = output.strip().split("\n")
    if len(lines) < 2:  # Need at least header and one data row
        raise ValueError("Invalid CSV output format")

    header = lines[0].split(",")

    results = {
        "seq_lengths": [],
        "std_times": [],
        "flash_times": [],
        "speedups": [],
        "max_errors": [],
        "mean_errors": [],
    }

    for i in range(1, len(lines)):
        values = lines[i].split(",")
        if len(values) < len(header):
            continue  # Skip incomplete lines

        results["seq_lengths"].append(int(values[0]))
        results["std_times"].append(float(values[1]))
        results["flash_times"].append(float(values[2]))
        results["speedups"].append(float(values[3]))

        if len(values) > 4 and "max_abs_error" in header:
            max_error_idx = header.index("max_abs_error")
            results["max_errors"].append(float(values[max_error_idx]))

        if len(values) > 5 and "mean_abs_error" in header:
            mean_error_idx = header.index("mean_abs_error")
            results["mean_errors"].append(float(values[mean_error_idx]))

    return results


def run_benchmark(args: argparse.Namespace) -> Dict[str, List[Any]]:
    """Run the C++ benchmark executable with given arguments."""
    # Build command
    cmd = [os.path.join(args.bin_dir, "benchmark")]

    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])

    if args.num_heads:
        cmd.extend(["--num_heads", str(args.num_heads)])

    if args.head_dim:
        cmd.extend(["--head_dim", str(args.head_dim)])

    if args.num_runs:
        cmd.extend(["--num_runs", str(args.num_runs)])

    if args.seq_lengths:
        cmd.extend(["--seq_lengths", args.seq_lengths])

    if args.test_kernel_only:
        cmd.append("--test_kernel_only")

    if args.verify:
        cmd.append("--verify")

    # Always get CSV output for easier parsing
    cmd.append("--csv")

    # Run benchmark and capture output
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return parse_csv_output(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark: {e}")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)


def plot_results(results: Dict[str, List[Any]], output_path: Optional[str] = None):
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
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run attention benchmarks")
    parser.add_argument(
        "--bin_dir",
        type=str,
        default="../build",
        help="Directory containing the benchmark executable",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size for the test")
    parser.add_argument("--num_heads", type=int, help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, help="Dimension of each attention head")
    parser.add_argument(
        "--num_runs", type=int, help="Number of runs for each benchmark"
    )
    parser.add_argument(
        "--seq_lengths",
        type=str,
        help="Comma-separated list of sequence lengths to benchmark",
    )
    parser.add_argument(
        "--test_kernel_only",
        action="store_true",
        help="Only test the fill_ones kernel with a small matrix",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify correctness between implementations",
    )
    parser.add_argument("--output", type=str, help="Output path for the benchmark plot")
    args = parser.parse_args()

    if args.test_kernel_only:
        # Just run the test and exit
        subprocess.run([os.path.join(args.bin_dir, "benchmark"), "--test_kernel_only"])
        return

    # Run benchmark
    results = run_benchmark(args)

    # Plot results
    plot_results(results, args.output)

    # Print summary
    print("\nSummary:")
    print(
        f"Maximum speedup: {max(results['speedups']):.2f}x at sequence length "
        f"{results['seq_lengths'][results['speedups'].index(max(results['speedups']))]}"
    )


if __name__ == "__main__":
    main()
