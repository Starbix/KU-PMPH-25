#!/usr/bin/env python3
import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import matplotlib.pyplot as plt
import numpy as np

# compile for Ampere (A100)
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

DISABLE_FLASH_ATTENTION = False
VERBOSE_PROFILING = False


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
    print("(this can take a while), compiling...")
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
    # Q, K, V shape: (seq_len, head_dim)
    # TODO: Add scaling factor 1/sqrt(head_dim)
    S = torch.matmul(Q, K.transpose(-2, -1)).clone()
    A = F.softmax(S, dim=-1).clone()
    output = torch.matmul(A, V)

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
    print("q:", q.shape)
    print("k:", k.shape)
    print("v:", v.shape)
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
    is_custom_kernel=False,
):
    """Benchmark a single attention implementation."""
    print(f"Benchmarking {name}...")

    # Warmup runs
    try:
        for _ in range(warmup_runs):
            _ = impl_func(Q, K, V)
            torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
        print(f"  CUDA out of memory during warmup: {e}")
        return -1, []

    torch.cuda.synchronize()

    times = []

    for run in range(num_runs):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            try:
                output = impl_func(Q, K, V)
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError as e:
                print(f"  CUDA out of memory during run {run}: {e}")
                return -1, []

        # Print detailed profiling info if enabled
        if VERBOSE_PROFILING:
            print(f"\n  --- Profiling details for run {run} ---")
            print(prof.key_averages().table(sort_by="cuda_time", row_limit=10))
            print()

        # Extract time from profiler
        if is_custom_kernel:
            # For custom kernels, count CPU + CUDA time for all operations
            # Include aten:: kernels but exclude CUDA API calls and memory copies
            kernel_time = 0
            if VERBOSE_PROFILING:
                print(f"\n  --- Counting kernels for run {run} ---")
            for evt in prof.key_averages():
                name = evt.key
                # Count CPU + CUDA time including aten:: operations and memory allocations
                # Exclude CUDA API calls (except malloc/free), memory copies, and profiler overhead
                if (
                    (
                        not name.startswith("cuda")
                        or name.startswith("cudaMalloc")
                        or name.startswith("cudaFree")
                        or name.startswith("cudaLaunchKernel")
                    )
                    and not name.startswith("Memcpy")
                    and not "Activity Buffer" in name
                    and (evt.device_time > 0 or evt.cpu_time_total > 0)
                ):
                    evt_time = evt.cpu_time_total + evt.device_time
                    if VERBOSE_PROFILING:
                        print(
                            f"    COUNTED: {name[:60]:60s} CPU: {evt.cpu_time_total:8.3f} us  CUDA: {evt.device_time:8.3f} us  Total: {evt_time:10.3f} us"
                        )
                    kernel_time += evt_time
                elif (
                    evt.device_time > 0 or evt.cpu_time_total > 0
                ) and VERBOSE_PROFILING:
                    evt_time = evt.cpu_time_total + evt.device_time
                    print(
                        f"    SKIPPED: {name[:60]:60s} CPU: {evt.cpu_time_total:8.3f} us  CUDA: {evt.device_time:8.3f} us  Total: {evt_time:10.3f} us"
                    )

            run_time = kernel_time / 1000.0  # Convert to ms
            if VERBOSE_PROFILING:
                print(f"  Total kernel time: {kernel_time:.3f} us = {run_time:.3f} ms")
        else:
            # For PyTorch implementations, count everything (CPU + CUDA)
            run_time = (
                sum(
                    [
                        evt.cpu_time_total + evt.device_time
                        for evt in prof.key_averages()
                    ]
                )
                / 1000.0
            )  # Convert to ms

        times.append(run_time)
        print(f"  Run {run:2d}: {run_time:.3f} ms")

    # drop first run, usually much longer than subsequent runs
    stats_times = times[1:] if len(times) > 1 else times
    avg_time = sum(stats_times) / len(stats_times)
    std_time = np.std(stats_times)
    min_time = min(stats_times)
    max_time = max(stats_times)

    print(f"  Average: {avg_time:.3f} ± {std_time:.3f} ms")
    print(f"  Min/Max: {min_time:.3f} / {max_time:.3f} ms")

    return avg_time, times


def verify_implementations(attention_mod, flash_attention_mod, Q, K, V, tolerance=1e-4):
    """Verify that all implementations produce similar results."""
    print("\nRunning verification...")

    torch_output = torch_reference_attention(Q, K, V)
    attention_output = attention_mod.forward(Q, K, V)

    attention_close = torch.allclose(
        attention_output, torch_output, atol=tolerance, rtol=tolerance
    )
    if attention_close:
        print("✅ Standard attention matches PyTorch reference")
    else:
        max_diff = torch.max(torch.abs(attention_output - torch_output)).item()
        mean_diff = torch.mean(torch.abs(attention_output - torch_output)).item()
        print(
            f"💀 Standard attention differs from reference (max: {max_diff:.6f}, mean: {mean_diff:.6f})"
        )

    flash_output = flash_attention_mod.forward(Q, K, V)

    flash_close = torch.allclose(
        flash_output, torch_output, atol=tolerance, rtol=tolerance
    )
    if flash_close:
        print("✅ Flash attention matches PyTorch reference")
    else:
        max_diff = torch.max(torch.abs(flash_output - torch_output)).item()
        mean_diff = torch.mean(torch.abs(flash_output - torch_output)).item()
        print(
            f"💀 Flash attention differs from reference (max: {max_diff:.6f}, mean: {mean_diff:.6f})"
        )

    return attention_close and flash_close


def plot_results(results, output_path=None):
    print(results)
    # Filter out implementations with -1 (didn't compute anything)
    valid_results = {
        impl: data for impl, data in results.items() if data["avg_time"] != -1
    }

    if not valid_results:
        print("No valid results to plot (all implementations returned -1)")
        return

    implementations = list(valid_results.keys())
    avg_times = [valid_results[impl]["avg_time"] for impl in implementations]

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
    times_data = [valid_results[impl]["times"] for impl in implementations]
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
        torch_time, _ = benchmark_implementation(
            torch_reference_attention,
            Q,
            K,
            V,
            args.num_runs,
            name="PyTorch Reference",
        )

        # Skip standard attention if PyTorch reference failed due to OOM
        if torch_time == -1:
            attention_time = -1
            print("  Skipping Standard Attention due to PyTorch Reference OOM")
        else:
            attention_time, _ = benchmark_implementation(
                attention_mod.forward,
                Q,
                K,
                V,
                args.num_runs,
                name="Standard Attention",
            )

        if not DISABLE_FLASH_ATTENTION:
            flash_time, _ = benchmark_implementation(
                flash_attention_mod.forward,
                Q,
                K,
                V,
                args.num_runs,
                name="Flash Attention",
            )
        else:
            flash_time = 0

        results["torch_times"].append(torch_time)
        results["attention_times"].append(attention_time)
        results["flash_times"].append(flash_time)

    # Filter out -1 values for each dataset
    torch_valid = [
        (s, t) for s, t in zip(seq_lengths, results["torch_times"]) if t != -1
    ]
    attention_valid = [
        (s, t) for s, t in zip(seq_lengths, results["attention_times"]) if t != -1
    ]
    flash_valid = (
        [(s, t) for s, t in zip(seq_lengths, results["flash_times"]) if t != -1]
        if not DISABLE_FLASH_ATTENTION
        else []
    )

    # Unzip for plotting
    torch_seq, torch_times = zip(*torch_valid) if torch_valid else ([], [])
    attention_seq, attention_times = (
        zip(*attention_valid) if attention_valid else ([], [])
    )
    flash_seq, flash_times = zip(*flash_valid) if flash_valid else ([], [])

    plt.figure(figsize=(10, 6))
    plt.plot(torch_seq, torch_times, "o-", label="PyTorch Reference")
    plt.plot(attention_seq, attention_times, "s-", label="Standard Attention")
    if not DISABLE_FLASH_ATTENTION:
        plt.plot(flash_seq, flash_times, "^-", label="Flash Attention")

    # Add OOM lines
    for name, times, color in [
        ("PyTorch", results["torch_times"], "r"),
        ("Standard Attention", results["attention_times"], "g"),
        ("Flash Attention", results["flash_times"], "b"),
    ]:
        oom = next((i for i, t in enumerate(times) if t == -1), None)
        if oom is not None:
            plt.axvline(
                x=seq_lengths[oom],
                color=color,
                linestyle="--",
                label=f"First {name} OOM",
            )

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


def run_hyperparameter_search(flash_attention_mod, args):
    """Search for optimal hyperparameters for flash attention across different configurations."""

    # Sequence lengths to test: 2^6 to 2^14
    seq_lengths = [2**i for i in range(6, 15)]
    head_dims = [64, 128]

    # Hyperparameter search space
    block_sizes = [8, 16, 24, 32, 48]
    thread_dims = [8, 12, 16, 32, 48]

    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH FOR FLASH ATTENTION")
    print("=" * 80)
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Head dimensions: {head_dims}")
    print(f"Block size candidates: {block_sizes}")
    print(f"Thread dimension candidates: {thread_dims}")
    print("=" * 80 + "\n")

    all_results = []

    for head_dim in head_dims:
        for seq_len in seq_lengths:
            print(f"\n{'=' * 80}")
            print(f"Testing seq_len={seq_len}, head_dim={head_dim}")
            print(f"{'=' * 80}")

            # Create test tensors
            Q, K, V = create_test_tensors(seq_len, head_dim)

            best_time = float("inf")
            best_params = None
            config_results = []

            # Test all hyperparameter combinations
            for B_c in block_sizes:
                if B_c > seq_len:
                    continue

                for B_r in block_sizes:
                    if B_r > head_dim:
                        continue

                    for bdim_x in thread_dims:
                        for bdim_y in thread_dims:
                            # blocks can't have more threads than 1024
                            if bdim_x * bdim_y > 1024:
                                continue
                            params_str = f"B_c={B_c:3d}, B_r={B_r:3d}, bdim_x={bdim_x:2d}, bdim_y={bdim_y:2d}"
                            print(f"Testing {params_str}", flush=True)
                            try:
                                # Warmup run
                                _ = flash_attention_mod.forward_with_parameters(
                                    Q, K, V, B_c, B_r, bdim_x, bdim_y
                                )
                                torch.cuda.synchronize()

                                # Timing runs using profiler (consistent with benchmark_implementation)
                                times = []
                                num_runs = 5  # Fewer runs for hyperparameter search

                                for _ in range(num_runs):
                                    with torch.profiler.profile(
                                        activities=[
                                            torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA,
                                        ],
                                        record_shapes=False,
                                        profile_memory=False,
                                        with_stack=False,
                                    ) as prof:
                                        output = (
                                            flash_attention_mod.forward_with_parameters(
                                                Q, K, V, B_c, B_r, bdim_x, bdim_y
                                            )
                                        )
                                        torch.cuda.synchronize()

                                    # Extract time from profiler (same as custom kernel timing)
                                    kernel_time = 0
                                    for evt in prof.key_averages():
                                        name = evt.key
                                        if (
                                            (
                                                not name.startswith("cuda")
                                                or name.startswith("cudaMalloc")
                                                or name.startswith("cudaFree")
                                                or name.startswith("cudaLaunchKernel")
                                            )
                                            and not name.startswith("Memcpy")
                                            and not "Activity Buffer" in name
                                            and (
                                                evt.device_time > 0
                                                or evt.cpu_time_total > 0
                                            )
                                        ):
                                            kernel_time += (
                                                evt.cpu_time_total + evt.device_time
                                            )

                                    run_time = kernel_time / 1000.0  # Convert to ms
                                    times.append(run_time)

                                avg_time = sum(times) / len(times)

                                config_results.append(
                                    {
                                        "B_c": B_c,
                                        "B_r": B_r,
                                        "bdim_x": bdim_x,
                                        "bdim_y": bdim_y,
                                        "time": avg_time,
                                        "status": "success",
                                    }
                                )

                                if avg_time < best_time:
                                    best_time = avg_time
                                    best_params = (B_c, B_r, bdim_x, bdim_y)

                                print(f"  ✓ {params_str} -> {avg_time:7.3f} ms")

                            except RuntimeError as e:
                                error_msg = str(e)
                                print(f"  ✗ {params_str} -> FAILED ({error_msg})")
                                if (
                                    "out of memory" in error_msg.lower()
                                    or "shared memory" in error_msg.lower()
                                ):
                                    config_results.append(
                                        {
                                            "B_c": B_c,
                                            "B_r": B_r,
                                            "bdim_x": bdim_x,
                                            "bdim_y": bdim_y,
                                            "time": None,
                                            "status": "failed_memory",
                                        }
                                    )
                                else:
                                    config_results.append(
                                        {
                                            "B_c": B_c,
                                            "B_r": B_r,
                                            "bdim_x": bdim_x,
                                            "bdim_y": bdim_y,
                                            "time": None,
                                            "status": "failed_other",
                                        }
                                    )
                                torch.cuda.empty_cache()

            # Print summary for this configuration
            print(f"\n{'-' * 80}")
            if best_params:
                B_c, B_r, bdim_x, bdim_y = best_params
                print(f"BEST PARAMS for seq_len={seq_len}, head_dim={head_dim}:")
                print(f"  B_c={B_c}, B_r={B_r}, bdim_x={bdim_x}, bdim_y={bdim_y}")
                print(f"  Time: {best_time:.3f} ms")
            else:
                print(f"NO SUCCESSFUL RUNS for seq_len={seq_len}, head_dim={head_dim}")
            print(f"{'-' * 80}\n")

            all_results.append(
                {
                    "seq_len": seq_len,
                    "head_dim": head_dim,
                    "best_params": best_params,
                    "best_time": best_time if best_time != float("inf") else None,
                    "configs": config_results,
                }
            )

    # Print final summary
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH RESULTS SUMMARY")
    print("=" * 80)
    print(
        f"{'seq_len':>8} | {'head_dim':>8} | {'B_c':>5} | {'B_r':>5} | {'bdim_x':>7} | {'bdim_y':>7} | {'Time (ms)':>10}"
    )
    print("-" * 80)

    for result in all_results:
        seq_len = result["seq_len"]
        head_dim = result["head_dim"]
        if result["best_params"]:
            B_c, B_r, bdim_x, bdim_y = result["best_params"]
            time = result["best_time"]
            print(
                f"{seq_len:8d} | {head_dim:8d} | {B_c:5d} | {B_r:5d} | {bdim_x:7d} | {bdim_y:7d} | {time:10.3f}"
            )
        else:
            print(
                f"{seq_len:8d} | {head_dim:8d} | {'N/A':>5} | {'N/A':>5} | {'N/A':>7} | {'N/A':>7} | {'FAILED':>10}"
            )

    print("=" * 80 + "\n")
    return all_results


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
        "--hyperparam-search",
        action="store_true",
        help="Run hyperparameter search to find optimal block sizes and thread dimensions",
    )

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        return 1

    print("Python Attention Benchmark")
    print("=" * 30)
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print()

    try:
        attention_mod, flash_attention_mod = load_attention_modules()
        print("Successfully loaded attention modules\n")
    except Exception as e:
        print(f"Error loading attention modules: {e}")
        return 1

    if args.hyperparam_search:
        run_hyperparameter_search(flash_attention_mod, args)
    elif args.sweep:
        run_sequence_length_sweep(attention_mod, flash_attention_mod, args)
    else:
        # Single benchmark run
        seq_len = int(args.seq_lengths.split(",")[0])  # Use first sequence length

        print(f"Configuration:")
        print(f"  Sequence Length: {seq_len}")
        print(f"  Head Dimension: {args.head_dim}")
        print(f"  Number of Runs: {args.num_runs}")
        print()

        # Create test tensors
        Q, K, V = create_test_tensors(seq_len, args.head_dim)
        print(f"Created test tensors with shape: {Q.shape}")

        # Verification
        if args.verify:
            verify_implementations(attention_mod, flash_attention_mod, Q, K, V)

        # Benchmark all implementations
        results = {}

        torch_time, torch_times, _ = benchmark_implementation(
            torch_reference_attention,
            Q,
            K,
            V,
            args.num_runs,
            name="PyTorch Reference",
        )
        results["PyTorch Reference"] = {
            "avg_time": torch_time,
            "times": torch_times,
        }

        # Skip standard attention if PyTorch reference failed due to OOM
        if torch_time == -1:
            attention_time = -1
            attention_times = []
            print("  Skipping Standard Attention due to PyTorch Reference OOM")
        else:
            attention_time, attention_times = benchmark_implementation(
                attention_mod.forward,
                Q,
                K,
                V,
                args.num_runs,
                name="Standard Attention",
                is_custom_kernel=True,
            )
        results["Standard Attention"] = {
            "avg_time": attention_time,
            "times": attention_times,
        }

        if not DISABLE_FLASH_ATTENTION:
            flash_time, flash_times, _ = benchmark_implementation(
                flash_attention_mod.forward,
                Q,
                K,
                V,
                args.num_runs,
                name="Flash Attention",
                is_custom_kernel=True,
            )
            results["Flash Attention"] = {"avg_time": flash_time, "times": flash_times}

        print(f"\n{'=' * 50}")
        print("BENCHMARK SUMMARY")
        print(f"{'=' * 50}")
        for impl, data in results.items():
            print(f"{impl:20s}: {data['avg_time']:8.3f} ms")

        if not DISABLE_FLASH_ATTENTION and attention_time > 0 and flash_time > 0:
            speedup = attention_time / flash_time
            print(f"{'Speedup (Flash/Std)':20s}: {speedup:8.2f}x")

        if not DISABLE_FLASH_ATTENTION and torch_time > 0 and flash_time > 0:
            speedup_torch = torch_time / flash_time
            print(f"{'Speedup (Flash/PyTorch)':20s}: {speedup_torch:8.2f}x")

        # Plot results
        if args.output or len(results) > 1:
            plot_results(results, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
