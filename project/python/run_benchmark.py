#!/usr/bin/env python3
"""
Python benchmark script for attention implementations.
Compares standard attention and flash attention implementations.
"""

import argparse
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import profiler

# Set CUDA architecture for compilation
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"


def load_attention_modules():
    """Load the attention and flash_attention CUDA modules."""
    print("Loading attention module...")
    attention = load(
        name="attention",
        sources=[
            "src/attention/attention.cpp",
            "src/attention/attention.cu",
        ],
        verbose=False,
        with_cuda=True,
        extra_cflags=["-DENABLE_PYBIND -O3"],
    )

    print("Loading flash_attention module...")
    flash_attention = load(
        name="flash_attention",
        sources=[
            "src/flash_attention/flash_attention.cpp",
            "src/flash_attention/flash_attention_kernel.cu",
        ],
        verbose=False,
        with_cuda=True,
        extra_cflags=["-DENABLE_PYBIND -O3"],
    )

    return attention, flash_attention


def torch_reference_attention(Q, K, V):
    """
    Reference PyTorch attention implementation for verification.
    Uses standard scaled dot-product attention (without scaling for now).
    """
    # Q, K, V shape: (seq_len, head_dim)
    # TODO: Add scaling factor 1/sqrt(head_dim)
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (seq_len, seq_len)
    probs = F.softmax(scores, dim=-1)  # (seq_len, seq_len)
    output = torch.matmul(probs, V)  # (seq_len, head_dim)
    return output


def create_test_tensors(seq_len, head_dim, device="cuda", dtype=torch.float32):
    """Create random test tensors Q, K, V."""
    return (
        torch.randn(seq_len, head_dim, device=device, dtype=dtype),
        torch.randn(seq_len, head_dim, device=device, dtype=dtype),
        torch.randn(seq_len, head_dim, device=device, dtype=dtype),
    )


def create_known_test_tensors(seq_len, head_dim, device="cuda", dtype=torch.float32):
    print("Creating known test tensors for verification...")
    q = torch.triu(torch.ones(seq_len, head_dim)).cuda()
    k = torch.tensor(
        [
            [i + j if (i + j) % 2 == 0 else -(i + j) for j in range(head_dim)]
            for i in range(seq_len)
        ],
        dtype=torch.float32,
    ).cuda()
    v = torch.ones(seq_len, head_dim, device=device, dtype=dtype)
    print("q:", q)
    print("k:", k)
    print("v:", v)
    return (
        q,
        k,
        v,
    )


def benchmark_implementation(
    impl_func,
    Q,
    K,
    V,
    num_runs=10,
    warmup_runs=0,
    name="Implementation",
    enable_profiler=False,
    enable_memory_tracking=False,
):
    """Benchmark a single attention implementation."""
    print(f"Benchmarking {name}...")

    # Warmup runs
    for _ in range(warmup_runs):
        _ = impl_func(Q, K, V)
        torch.cuda.synchronize()

    # Actual timing
    torch.cuda.synchronize()

    times = []
    prof_trace = None

    for run in range(num_runs):
        if enable_profiler and run == 0:  # Profile only the first run
            with profiler.profile(use_cuda=True, record_shapes=True) as prof:
                start = time.time()
                output = impl_func(Q, K, V)
                torch.cuda.synchronize()
                end = time.time()
            prof_trace = prof
        elif enable_memory_tracking and run == 0:  # Memory tracking on first run
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            output = impl_func(Q, K, V)
            torch.cuda.synchronize()
            end = time.time()

            # Print memory stats
            print(f"  Current memory usage: {torch.cuda.memory_usage() / 1e6:.2f} MB")
            print(
                f"  Peak memory allocated: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB"
            )
            print(
                f"  Peak memory reserved: {torch.cuda.max_memory_reserved() / 1e6:.2f} MB"
            )
        else:
            start = time.time()
            output = impl_func(Q, K, V)
            torch.cuda.synchronize()
            end = time.time()

        run_time = (end - start) * 1000  # Convert to milliseconds
        times.append(run_time)
        print(f"  Run {run:2d}: {run_time:.3f} ms")

    avg_time = sum(times) / len(times)
    std_time = np.std(times)
    min_time = min(times)
    max_time = max(times)

    print(f"  Average: {avg_time:.3f} ± {std_time:.3f} ms")
    print(f"  Min/Max: {min_time:.3f} / {max_time:.3f} ms")

    # Print profiler statistics if enabled
    if enable_profiler and prof_trace is not None:
        print(f"  Profiler summary for {name}:")
        print(prof_trace.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    return avg_time, times, output


def profile_implementation_detailed(
    impl_func, Q, K, V, name="Implementation", enable_memory_tracking=False
):
    """Run detailed profiling analysis for a single implementation."""
    print(f"\nDetailed profiling for {name}...")

    # Warmup
    for _ in range(3):
        _ = impl_func(Q, K, V)
        torch.cuda.synchronize()

    # Profile with detailed settings
    if enable_memory_tracking:
        torch.cuda.reset_peak_memory_stats()

    with profiler.profile(use_cuda=True, record_shapes=True) as prof:
        output = impl_func(Q, K, V)
        torch.cuda.synchronize()

    if enable_memory_tracking:
        print(f"Current memory usage: {torch.cuda.memory_usage() / 1e6:.2f} MB")
        print(
            f"Peak memory allocated: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB"
        )
        print(f"Peak memory reserved: {torch.cuda.max_memory_reserved() / 1e6:.2f} MB")

    # Print detailed profiler information
    print(f"\nProfiler Results for {name}:")
    print("=" * 80)

    # CUDA time breakdown
    print("\nTop 15 operations by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    # CPU time breakdown
    print("\nTop 10 operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    return output, prof


def verify_implementations(attention_mod, flash_attention_mod, Q, K, V, tolerance=1e-4):
    """Verify that all implementations produce similar results."""
    print("\nRunning verification...")

    # Get outputs from all implementations
    torch_output = torch_reference_attention(Q, K, V)
    attention_output = attention_mod.forward(Q, K, V)
    flash_output = flash_attention_mod.forward(Q, K, V)

    print(f"Output shapes:")
    print(f"  PyTorch reference: {torch_output.shape}")
    print(f"  Standard attention: {attention_output.shape}")
    print(f"  Flash attention: {flash_output.shape}")

    # Compare attention vs reference
    attention_close = torch.allclose(
        attention_output, torch_output, atol=tolerance, rtol=tolerance
    )
    if attention_close:
        print("✓ Standard attention matches PyTorch reference")
    else:
        max_diff = torch.max(torch.abs(attention_output - torch_output)).item()
        mean_diff = torch.mean(torch.abs(attention_output - torch_output)).item()
        print(
            f"✗ Standard attention differs from reference (max: {max_diff:.6f}, mean: {mean_diff:.6f})"
        )

    # Compare flash vs reference
    flash_close = torch.allclose(
        flash_output, torch_output, atol=tolerance, rtol=tolerance
    )
    if flash_close:
        print("✓ Flash attention matches PyTorch reference")
    else:
        max_diff = torch.max(torch.abs(flash_output - torch_output)).item()
        mean_diff = torch.mean(torch.abs(flash_output - torch_output)).item()
        print(
            f"✗ Flash attention differs from reference (max: {max_diff:.6f}, mean: {mean_diff:.6f})"
        )

    # Compare flash vs attention
    flash_attention_close = torch.allclose(
        flash_output, attention_output, atol=tolerance, rtol=tolerance
    )
    if flash_attention_close:
        print("✓ Flash attention matches standard attention")
    else:
        max_diff = torch.max(torch.abs(flash_output - attention_output)).item()
        mean_diff = torch.mean(torch.abs(flash_output - attention_output)).item()
        print(
            f"✗ Flash attention differs from standard attention (max: {max_diff:.6f}, mean: {mean_diff:.6f})"
        )

    return attention_close and flash_close and flash_attention_close


def plot_results(results, output_path=None):
    """Plot benchmark results."""
    implementations = list(results.keys())
    avg_times = [results[impl]["avg_time"] for impl in implementations]

    plt.figure(figsize=(10, 6))

    # Bar plot of average times
    plt.subplot(1, 2, 1)
    bars = plt.bar(implementations, avg_times)
    plt.ylabel("Average Time (ms)")
    plt.title("Attention Implementation Comparison")
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, time in zip(bars, avg_times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(avg_times) * 0.01,
            f"{time:.2f}ms",
            ha="center",
            va="bottom",
        )

    # Box plot showing distribution
    plt.subplot(1, 2, 2)
    times_data = [results[impl]["times"] for impl in implementations]
    plt.boxplot(times_data, tick_labels=implementations)
    plt.ylabel("Time (ms)")
    plt.title("Timing Distribution")
    plt.xticks(rotation=45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def run_sequence_length_sweep(attention_mod, flash_attention_mod, args):
    """Run benchmarks across different sequence lengths."""
    seq_lengths = [int(x) for x in args.seq_lengths.split(",")]
    results = {
        "seq_lengths": seq_lengths,
        "torch_times": [],
        "attention_times": [],
        "flash_times": [],
    }

    print(f"\nRunning sequence length sweep: {seq_lengths}")

    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")

        Q, K, V = create_test_tensors(seq_len, args.head_dim)

        # Benchmark each implementation
        torch_time, _, _ = benchmark_implementation(
            torch_reference_attention,
            Q,
            K,
            V,
            args.num_runs,
            name="PyTorch Reference",
            enable_profiler=args.profile,
            enable_memory_tracking=args.memory,
        )
        attention_time, _, _ = benchmark_implementation(
            attention_mod.forward,
            Q,
            K,
            V,
            args.num_runs,
            name="Standard Attention",
            enable_profiler=args.profile,
            enable_memory_tracking=args.memory,
        )
        flash_time, _, _ = benchmark_implementation(
            flash_attention_mod.forward,
            Q,
            K,
            V,
            args.num_runs,
            name="Flash Attention",
            enable_profiler=args.profile,
            enable_memory_tracking=args.memory,
        )

        results["torch_times"].append(torch_time)
        results["attention_times"].append(attention_time)
        results["flash_times"].append(flash_time)

    # Plot sequence length sweep results
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, results["torch_times"], "o-", label="PyTorch Reference")
    plt.plot(seq_lengths, results["attention_times"], "s-", label="Standard Attention")
    plt.plot(seq_lengths, results["flash_times"], "^-", label="Flash Attention")
    plt.xlabel("Sequence Length")
    plt.ylabel("Average Time (ms)")
    plt.title("Attention Performance vs Sequence Length")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if args.output:
        sweep_path = args.output.replace(".png", "_sweep.png")
        plt.savefig(sweep_path, dpi=300, bbox_inches="tight")
        print(f"Sequence length sweep plot saved to {sweep_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Benchmark attention implementations")

    parser.add_argument(
        "--seq_lengths",
        type=str,
        default="1024",
        help="Comma-separated list of sequence lengths (default: 1024)",
    )
    parser.add_argument(
        "--head_dim", type=int, default=64, help="Head dimension (default: 64)"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of benchmark runs (default: 10)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify correctness between implementations",
    )
    parser.add_argument("--output", type=str, help="Output path for benchmark plot")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run sequence length sweep instead of single benchmark",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable CUDA profiling with torch.autograd.profiler",
    )
    parser.add_argument(
        "--profile_output",
        type=str,
        default="./profiler_traces",
        help="Output directory for profiler traces (default: ./profiler_traces)",
    )
    parser.add_argument(
        "--profile_only",
        action="store_true",
        help="Run detailed profiling analysis only (no benchmarking)",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Enable CUDA memory tracking and snapshot generation",
    )

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        return 1

    print("Python Attention Benchmark")
    print("=" * 30)
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch Version: {torch.__version__}")
    print()

    # Load attention modules
    try:
        attention_mod, flash_attention_mod = load_attention_modules()
        print("Successfully loaded attention modules\n")
    except Exception as e:
        print(f"Error loading attention modules: {e}")
        return 1

    if args.sweep:
        run_sequence_length_sweep(attention_mod, flash_attention_mod, args)
    else:
        # Single benchmark run
        seq_len = int(args.seq_lengths.split(",")[0])  # Use first sequence length

        print(f"Configuration:")
        print(f"  Sequence Length: {seq_len}")
        print(f"  Head Dimension: {args.head_dim}")
        print(f"  Number of Runs: {args.num_runs}")
        print(
            f"  Profiling: {'Enabled' if args.profile or args.profile_only else 'Disabled'}"
        )
        print(f"  Memory Tracking: {'Enabled' if args.memory else 'Disabled'}")
        print()

        # Create test tensors
        Q, K, V = create_known_test_tensors(seq_len, args.head_dim)
        print(f"Created test tensors with shape: {Q.shape}")

        # Verification
        if args.verify:
            verify_implementations(attention_mod, flash_attention_mod, Q, K, V)

        if args.profile_only:
            # Run detailed profiling analysis only
            print("\nRunning detailed profiling analysis...")
            profile_implementation_detailed(
                torch_reference_attention,
                Q,
                K,
                V,
                "PyTorch Reference",
                enable_memory_tracking=args.memory,
            )
            profile_implementation_detailed(
                attention_mod.forward,
                Q,
                K,
                V,
                "Standard Attention",
                enable_memory_tracking=args.memory,
            )
            profile_implementation_detailed(
                flash_attention_mod.forward,
                Q,
                K,
                V,
                "Flash Attention",
                enable_memory_tracking=args.memory,
            )
            print("\nDetailed profiling completed.")
        else:
            # Benchmark all implementations
            results = {}

            torch_time, torch_times, _ = benchmark_implementation(
                torch_reference_attention,
                Q,
                K,
                V,
                args.num_runs,
                name="PyTorch Reference",
                enable_profiler=args.profile,
                enable_memory_tracking=args.memory,
            )
            results["PyTorch Reference"] = {
                "avg_time": torch_time,
                "times": torch_times,
            }

            attention_time, attention_times, _ = benchmark_implementation(
                attention_mod.forward,
                Q,
                K,
                V,
                args.num_runs,
                name="Standard Attention",
                enable_profiler=args.profile,
                enable_memory_tracking=args.memory,
            )
            results["Standard Attention"] = {
                "avg_time": attention_time,
                "times": attention_times,
            }

            flash_time, flash_times, _ = benchmark_implementation(
                flash_attention_mod.forward,
                Q,
                K,
                V,
                args.num_runs,
                name="Flash Attention",
                enable_profiler=args.profile,
                enable_memory_tracking=args.memory,
            )
            results["Flash Attention"] = {"avg_time": flash_time, "times": flash_times}

            # Print summary
            print(f"\n{'=' * 50}")
            print("BENCHMARK SUMMARY")
            print(f"{'=' * 50}")
            for impl, data in results.items():
                print(f"{impl:20s}: {data['avg_time']:8.3f} ms")

            if attention_time > 0 and flash_time > 0:
                speedup = attention_time / flash_time
                print(f"{'Speedup (Flash/Std)':20s}: {speedup:8.2f}x")

            # Plot results
            if args.output or len(results) > 1:
                plot_results(results, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
