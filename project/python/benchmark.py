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
import parameter_optimization as opt


def main():
    parser = argparse.ArgumentParser(description="A simple greeting program.")
    parser.add_argument("--seq_len", type=int, help=f"Sequence length (default {128})", default=128)
    parser.add_argument("--head_dim", type=int, help=f"Head dimension (default {64})", default=64)
    parser.add_argument("--Bc", type=int, help=f"B_c (default {32})", default=32)
    parser.add_argument("--Br", type=int, help=f"B_r (default {16})", default=16)
    parser.add_argument("--bdimx", type=int, help=f"bdim_x (default {32})", default=32)
    parser.add_argument("--bdimy", type=int, help=f"bdim_y (default {16})", default=16)
    args = parser.parse_args()

    seq_len = args.seq_len
    head_dim = args.head_dim
    B_c = args.Bc
    B_r = args.Br
    bdim_x = args.bdimx
    bdim_y = args.bdimy
   
    print(f"Running benchmarking with sequence length {seq_len} and head dimension {head_dim}")

    if not torch.cuda.is_available():
        print("⚠️ CUDA not available — exiting.")
        return    

    print("Loading flash_attention module")
    attention, flash_attention = opt.load_attention_modules()
    flash_attention_func = flash_attention.forward_duration
    standard_attention_func = attention.forward_duration

    print("Creating test matrices")
    Q, K, V = opt.create_random_test_tensors(seq_len, head_dim)

    print("Running benchmarking...")

    avg_time_pytorch, std_time_pytorch, min_time_pytorch, max_time_pytorch = opt.benchmark_standard_attention(
        torch_reference_attention,
        Q, K, V,
        num_runs=50,
        warmup_runs=1
    )
    avg_time_standard, std_time_standard, min_time_standard, max_time_standard = opt.benchmark_standard_attention(
        standard_attention_func,
        Q, K, V,
        num_runs=50,
        warmup_runs=1
    )
    avg_time_flash, std_time_flash, min_time_flash, max_time_flash = opt.benchmark_flash_attention(
        flash_attention_func,
         Q, K, V,
         B_c, B_r, bdim_x, bdim_y,
         num_runs=50,
         warmup_runs=1
    )

    print("# Benchmark results:")
    print("## Pytorch reference:")
    print(f"Avg. runtime: {avg_time_pytorch}\t\nStd. runtime: {std_time_pytorch}\t\nMin. runtime: {min_time_pytorch}\t\nMax. runtime: {max_time_pytorch}")
    print("## Standard attention:")
    print(f"Avg. runtime: {avg_time_standard}\t\nStd. runtime: {std_time_standard}\t\nMin. runtime: {min_time_standard}\t\nMax. runtime: {max_time_standard}")
    print("## Flash attention:")
    print(f"Avg. runtime: {avg_time_flash}\t\nStd. runtime: {std_time_flash}\t\nMin. runtime: {min_time_flash}\t\nMax. runtime: {max_time_flash}")
    print("")
    print(f"Pytorch/Flash speedup: {avg_time_pytorch/avg_time_flash}")
    print(f"Standard/Flash speedup: {avg_time_standard/avg_time_flash}")

    print("Running validation...")

    O_torch = torch_reference_attention(Q, K, V, measure_time=False)
    O_standard = None
    O_flash = None

    print("# Validation results:")




def torch_reference_attention(Q, K, V, measure_time=True):
    """
    Reference PyTorch attention implementation for verification.
    Uses standard scaled dot-product attention (without scaling for now).
    If measure_time=True, returns runtime in milliseconds.
    """
    if measure_time:
        # Create CUDA events for timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        
        # Perform attention
        scores = torch.matmul(Q, K.transpose(-2, -1))
        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, V)
        
        end.record()
        # Wait for the events to be recorded
        torch.cuda.synchronize()
        # Compute elapsed time in milliseconds
        runtime_ms = start.elapsed_time(end)
        return runtime_ms
    else:
        # Just return the output (for correctness checking)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, V)
        return output


if __name__ == "__main__": main()
    
