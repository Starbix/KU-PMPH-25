"""
Python optimization script for attention implementations.
Runs flash attention and minimizes runtime.
"""

import argparse
import os
import math
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import profiler
import argparse

# Set CUDA architecture for compilation
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"


def main():
    parser = argparse.ArgumentParser(description="A simple greeting program.")
    parser.add_argument(
        "--seq_len",
        type=int,
        nargs="+",
        help=f"Sequence length(s) (default [1024, 4096, 16384])",
        default=[2**10, 2**12, 2**14],
    )
    parser.add_argument(
        "--head_dim", type=int, help=f"Head dimension (default {64})", default=64
    )
    parser.add_argument(
        "--with_standard",
        action="store_true",
        help=f"Whether or not optimization should be done with comparison of standard attention.",
    )
    parser.add_argument(
        "--no_pytorch",
        action="store_true",
        help=f"Disable comparison with PyTorch's matmul-based attention implementation.",
    )
    args = parser.parse_args()

    seq_lens = args.seq_len if isinstance(args.seq_len, list) else [args.seq_len]
    head_dim = args.head_dim

    print(
        f"Running optimization with sequence length {seq_lens} and head dimension {head_dim}"
    )

    if not torch.cuda.is_available():
        print("⚠️ CUDA not available — exiting.")
        return

    print("Loading attention modules")
    attention, flash_attention = load_attention_modules()
    flash_attention_func = flash_attention.forward_duration
    standard_attention_func = attention.forward_duration

    grid = FlashAttentionParameterGrid(
        Bcs=[48, 32, 24, 16],
        Brs=[48, 32, 24, 16],
        bdim_xs=[32, 48, 64],
        bdim_ys=[8, 16],
    )

    all_results = {}

    for seq_len in seq_lens:
        print(f"\n{'=' * 60}")
        print(
            f"Running optimization with sequence length {seq_len} and head dimension {head_dim}"
        )
        print(f"{'=' * 60}")

        print("Creating test matrices")
        Q, K, V = create_random_test_tensors(seq_len, head_dim)

        if args.with_standard:
            print("Running optimization with standard attention as baseline")
            optimal_parameters = optimize_with_standard_attention(
                grid,
                flash_attention_func,
                standard_attention_func,
                Q,
                K,
                V,
                num_runs=50,
                warmup_runs=1,
            )
        elif args.no_pytorch:
            print("Running optimization without baseline comparison")
            optimal_parameters = optimize(
                grid, flash_attention_func, Q, K, V, num_runs=50, warmup_runs=1
            )
        else:
            print(
                "Running optimization with PyTorch's matmul-based attention as baseline"
            )
            optimal_parameters = optimize_with_pytorch_attention(
                grid, flash_attention_func, Q, K, V, num_runs=50, warmup_runs=1
            )

        all_results[seq_len] = optimal_parameters
        print(f"\nOptimization done for seq_len={seq_len}. Result:")
        print(optimal_parameters)

    print(f"\n{'=' * 60}")
    print("SUMMARY OF ALL RESULTS")
    print(f"{'=' * 60}")
    for seq_len, result in all_results.items():
        print(f"\nseq_len={seq_len}:")
        for key, value in result.items():
            if key != "failing_configuations":
                print(f"  {key}: {value}")


def load_attention_modules():
    """Load the attention and flash_attention CUDA modules."""
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


def create_random_test_tensors(
    seq_len: int, head_dim: int, device="cuda", dtype=torch.float32
):
    q = torch.randn(seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(seq_len, head_dim, dtype=dtype, device=device)
    return q, k, v


def benchmark_standard_attention(impl_func, Q, K, V, num_runs=10, warmup_runs=0):
    """Benchmark a single attention implementation."""
    # Warmup runs
    for _ in range(warmup_runs):
        _ = impl_func(Q, K, V)
        torch.cuda.synchronize()

    # Actual timing
    torch.cuda.synchronize()

    times = []

    for _ in range(num_runs):
        runtime = impl_func(Q, K, V)
        torch.cuda.synchronize()
        times.append(runtime)

    avg_time = sum(times) / len(times)
    std_time = np.std(times)
    min_time = min(times)
    max_time = max(times)

    return avg_time, std_time, min_time, max_time


def benchmark_pytorch_attention(Q, K, V, num_runs=10, warmup_runs=0):
    """Benchmark PyTorch's manual attention implementation using matmul."""
    # Warmup runs
    for _ in range(warmup_runs):
        with torch.no_grad():
            scores = torch.matmul(Q, K.transpose(-2, -1))
            probs = torch.softmax(scores, dim=-1)
            output = torch.matmul(probs, V)
        torch.cuda.synchronize()

    # Actual timing
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    for i in range(num_runs):
        start_events[i].record()
        with torch.no_grad():
            scores = torch.matmul(Q, K.transpose(-2, -1))
            probs = torch.softmax(scores, dim=-1)
            output = torch.matmul(probs, V)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [start_events[i].elapsed_time(end_events[i]) for i in range(num_runs)]

    avg_time = sum(times) / len(times)
    std_time = np.std(times)
    min_time = min(times)
    max_time = max(times)

    return avg_time, std_time, min_time, max_time


def benchmark_flash_attention(
    impl_func,
    Q,
    K,
    V,
    B_c,
    B_r,
    bdim_x,
    bdim_y,
    num_runs=10,
    warmup_runs=0,
):
    """Benchmark a single attention implementation."""

    # Warmup runs
    for _ in range(warmup_runs):
        _ = impl_func(Q, K, V, B_c, B_r, bdim_x, bdim_y)
        torch.cuda.synchronize()

    # Actual timing
    torch.cuda.synchronize()

    times = []

    for _ in range(num_runs):
        runtime = impl_func(Q, K, V, B_c, B_r, bdim_x, bdim_y)
        times.append(runtime)

    avg_time = sum(times) / len(times)
    std_time = np.std(times)
    min_time = min(times)
    max_time = max(times)

    return avg_time, std_time, min_time, max_time


class FlashAttentionParameterGrid:
    def __init__(self, Bcs: list, Brs: list, bdim_xs: list, bdim_ys: list):
        self.B_cs = Bcs
        self.B_rs = Brs
        self.bdim_xs = bdim_xs
        self.bdim_ys = bdim_ys


def optimize(
    grid: FlashAttentionParameterGrid, impl_func, Q, K, V, num_runs=10, warmup_runs=0
) -> dict:
    optimal_parameters = {"B_c": 0, "B_r": 0, "bdim_x": 0, "bdim_y": 0}
    best_avg_time = math.inf
    for B_c in grid.B_cs:
        for B_r in grid.B_rs:
            for bdim_x in grid.bdim_xs:
                for bdim_y in grid.bdim_ys:
                    avg_time, _, _, _ = benchmark_flash_attention(
                        impl_func,
                        Q,
                        K,
                        V,
                        B_c,
                        B_r,
                        bdim_x,
                        bdim_y,
                        num_runs=num_runs,
                        warmup_runs=warmup_runs,
                    )
                    if avg_time < best_avg_time:
                        best_avg_time = avg_time
                        optimal_parameters["B_c"] = B_c
                        optimal_parameters["B_r"] = B_r
                        optimal_parameters["bdim_x"] = bdim_x
                        optimal_parameters["bdim_y"] = bdim_y
    return optimal_parameters


def optimize_with_standard_attention(
    grid: FlashAttentionParameterGrid,
    impl_func_flash,
    impl_func_standard,
    Q,
    K,
    V,
    num_runs=10,
    warmup_runs=0,
) -> dict:
    optimal_parameters = {
        "B_c": 0,
        "B_r": 0,
        "bdim_x": 0,
        "bdim_y": 0,
        "avg_time_flash (ms)": 0,
        "avg_time_standard (ms)": 0,
        "speedup": 0,
        "failing_configuations": [],
    }
    best_avg_time = math.inf
    for B_c in grid.B_cs:
        for B_r in grid.B_rs:
            for bdim_x in grid.bdim_xs:
                for bdim_y in grid.bdim_ys:
                    try:
                        avg_time_flash, _, _, _ = benchmark_flash_attention(
                            impl_func_flash,
                            Q,
                            K,
                            V,
                            B_c,
                            B_r,
                            bdim_x,
                            bdim_y,
                            num_runs=num_runs,
                            warmup_runs=warmup_runs,
                        )
                        avg_time_standard, _, _, _ = benchmark_standard_attention(
                            impl_func_standard,
                            Q,
                            K,
                            V,
                            num_runs=num_runs,
                            warmup_runs=warmup_runs,
                        )
                        if (
                            avg_time_flash < avg_time_standard
                            and avg_time_flash < best_avg_time
                        ):
                            best_avg_time = avg_time_flash
                            optimal_parameters["B_c"] = B_c
                            optimal_parameters["B_r"] = B_r
                            optimal_parameters["bdim_x"] = bdim_x
                            optimal_parameters["bdim_y"] = bdim_y
                            optimal_parameters["avg_time_flash (ms)"] = avg_time_flash
                            optimal_parameters["avg_time_standard (ms)"] = (
                                avg_time_standard
                            )
                            optimal_parameters["speedup"] = (
                                avg_time_standard / avg_time_flash
                            )
                    except Exception as e:
                        optimal_parameters["failing_configuations"] = {
                            "B_c": B_c,
                            "B_r": B_r,
                            "bdim_x": bdim_x,
                            "bdim_y": bdim_y,
                        }
                        # skipping this configuration
                        continue
    return optimal_parameters


def optimize_with_pytorch_attention(
    grid: FlashAttentionParameterGrid,
    impl_func_flash,
    Q,
    K,
    V,
    num_runs=10,
    warmup_runs=0,
) -> dict:
    """Optimize flash attention parameters using PyTorch's matmul-based attention as baseline."""
    optimal_parameters = {
        "B_c": 0,
        "B_r": 0,
        "bdim_x": 0,
        "bdim_y": 0,
        "avg_time_flash (ms)": 0,
        "avg_time_pytorch (ms)": 0,
        "speedup": 0,
        "failing_configuations": [],
    }

    # Benchmark PyTorch's implementation once (it doesn't change)
    avg_time_pytorch, _, _, _ = benchmark_pytorch_attention(
        Q, K, V, num_runs=num_runs, warmup_runs=warmup_runs
    )

    best_avg_time = math.inf
    for B_c in grid.B_cs:
        for B_r in grid.B_rs:
            for bdim_x in grid.bdim_xs:
                for bdim_y in grid.bdim_ys:
                    try:
                        avg_time_flash, _, _, _ = benchmark_flash_attention(
                            impl_func_flash,
                            Q,
                            K,
                            V,
                            B_c,
                            B_r,
                            bdim_x,
                            bdim_y,
                            num_runs=num_runs,
                            warmup_runs=warmup_runs,
                        )
                        if avg_time_flash < best_avg_time:
                            best_avg_time = avg_time_flash
                            optimal_parameters["B_c"] = B_c
                            optimal_parameters["B_r"] = B_r
                            optimal_parameters["bdim_x"] = bdim_x
                            optimal_parameters["bdim_y"] = bdim_y
                            optimal_parameters["avg_time_flash (ms)"] = avg_time_flash
                            optimal_parameters["avg_time_pytorch (ms)"] = (
                                avg_time_pytorch
                            )
                            optimal_parameters["speedup"] = (
                                avg_time_pytorch / avg_time_flash
                            )
                    except Exception as e:
                        optimal_parameters["failing_configuations"].append(
                            {"B_c": B_c, "B_r": B_r, "bdim_x": bdim_x, "bdim_y": bdim_y}
                        )
                        # skipping this configuration
                        continue
    return optimal_parameters


if __name__ == "__main__":
    main()
