import matplotlib.pyplot as plt
import numpy as np
import argparse
import parameter_optimization as opt
import torch
import plotting_benchmarks


def main():
    parser = argparse.ArgumentParser(description="Plotting out of memory")
    parser.add_argument(
        "--head_dim", type=int, help=f"Head dimension (default {64})", default=64
    )
    parser.add_argument(
       "--seq_lens",
        type=int,
        nargs='+',  # Accept multiple values
        help="The list of seq. lengths to plot runtime against",
        default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    )
    parser.add_argument("--Bc", type=int, help=f"B_c (default {32})", default=32)
    parser.add_argument("--Br", type=int, help=f"B_r (default {16})", default=16)
    parser.add_argument("--bdimx", type=int, help=f"bdim_x (default {32})", default=32)
    parser.add_argument("--bdimy", type=int, help=f"bdim_y (default {16})", default=16)
    parser.add_argument("--file_path", type=str, help=f"The filepath of the resulting plot")
    args = parser.parse_args()

    head_dim = args.head_dim
    B_c = args.Bc
    B_r = args.Br
    bdim_x = args.bdimx
    bdim_y = args.bdimy
    seq_lens = args.seq_lens
    file_path = f"python/plots/oom_visualization_{head_dim}_{B_c}_{B_r}_{bdim_x}_{bdim_y}.png"
    if args.file_path:
        file_path = args.file_path

    print(
        f"Plotting with the following configuration: head_dim: {head_dim}, B_c: {B_c}, B_r: {B_r}, bdim_x: {bdim_x}, bdim_y: {bdim_y}"
    )

    if not torch.cuda.is_available():
        print("⚠️ CUDA not available — exiting.")
        return

    print("Loading attention modules")
    attention, flash_attention = opt.load_attention_modules()
    flash_attention_func = flash_attention.forward_duration
    standard_attention_func = attention.forward_duration

    print("Running benchmarks")
    runtimes_flash = [
        call_flash_attention(
            N, head_dim, 
            B_c, B_r, bdim_x, bdim_y, 
            flash_attention_func
        )
        for N in seq_lens
    ]
    runtimes_standard = [
        call_standard_attention(
            N, head_dim,
            standard_attention_func
        )
        for N in seq_lens
    ]

    print("Plotting")

    if contains_oom_err(runtimes_flash):
        valid_seq_lens_flash, valid_runtimes_flash, x_oom_err_flash, y_oom_err_flash = find_first_oom_error(seq_lens, runtimes_flash) 
        plt.plot(
            valid_seq_lens_flash, 
            valid_runtimes_flash, 
            label="Flash Attention", 
            marker='s', 
            linestyle='-'
        )
        plt.plot(
            x_oom_err_flash, 
            y_oom_err_flash, 
            marker='X', 
            color="orange", 
            markersize=12,
            label="OOM Flash Attention"
        )
        plt.text(x_oom_err_flash - 0.1, y_oom_err_flash - 0.1, fr"$N$ = {x_oom_err_flash}")
        plt.axvline(x=x_oom_err_flash, color='orange', linestyle='--', linewidth=1)
    else:
        plt.plot(
            seq_lens, 
            runtimes_flash, 
            label="Flash Attention", 
            marker='s', 
            linestyle='-'
        )
    if contains_oom_err(runtimes_standard):
        valid_seq_lens_standard, valid_runtimes_standard, x_oom_err_standard, y_oom_err_standard = find_first_oom_error(seq_lens, runtimes_standard) 
        plt.plot(
            valid_seq_lens_standard, 
            valid_runtimes_standard, 
            label="Standard Attention", 
            marker='^', 
            linestyle='-',
            color='g'
        )
        plt.plot(
            x_oom_err_standard, 
            y_oom_err_standard, 
            marker='X', 
            color='r', 
            markersize=12,
            label="OOM Standard Attention"
        )
        plt.text(x_oom_err_standard - 1.2, y_oom_err_standard - 1.2, fr"$N$ = {x_oom_err_standard}")
        plt.axvline(x=x_oom_err_standard, color='r', linestyle='--', linewidth=1)
    else:
        plt.plot(
            seq_lens, 
            runtimes_standard, 
            label="Standard Attention", 
            marker='^', 
            linestyle='-',
            color='g'
        )

    plt.xlabel(r"$N$")
    plt.ylabel("Runtime (ms)")
    plt.ylabel("Runtime (ms)")

    info_text = (
        rf"$d$: {head_dim}" + "\n"
        rf"$B_c$: {B_c}" + "\n"
        rf"$B_r$: {B_r}" + "\n"
        rf"$bdim_x$: {bdim_x}" + "\n"
        rf"$bdim_y$: {bdim_y}" 
    )

    legend = plt.legend(
        loc='upper left',
        frameon=True,
        facecolor='white',
        edgecolor='gray'
    )

    fig = plt.gcf()
    renderer = fig.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer=renderer)
    bbox_fig = bbox.transformed(fig.transFigure.inverted())

    x0 = bbox_fig.x0
    y0 = bbox_fig.y0 - 0.02  # shift down a bit

    plt.figtext(
        x0 + 0.01, y0,
        info_text,
        ha='left', va='top',
        fontsize=9,
        bbox=dict(
            boxstyle='round,pad=0.4',
            facecolor='white',
            edgecolor='gray',
            alpha=0.8
        )
    )

    plt.savefig(file_path)


def call_flash_attention(seq_len, head_dim, B_c, B_r, bdim_x, bdim_y, flash_func):
    Q, K, V = opt.create_random_test_tensors(seq_len, head_dim)
    try:
        avg_time, std_time, min_time, max_time = opt.benchmark_flash_attention(
            flash_func,
            Q, K, V, 
            B_c, B_r, bdim_x, bdim_y,
            num_runs=50, warmup_runs=1
        )
        return avg_time
    except Exception:
        return -1

def call_standard_attention(seq_len, head_dim, standard_func):
    Q, K, V = opt.create_random_test_tensors(seq_len, head_dim)
    try:
        avg_time, std_time, min_time, max_time = opt.benchmark_standard_attention(
            standard_func,
            Q, K, V,
            num_runs=50, warmup_runs=1
        )
        return avg_time
    except Exception:
        return -1

def find_first_oom_error(seq_lens, runtimes):
    # Returns 
    if contains_oom_err(runtimes):
        idx = runtimes.index(-1)
        return seq_lens[:idx], runtimes[:idx], seq_lens[idx], runtimes[idx - 1]


def contains_oom_err(runtimes):
    return -1 in runtimes


if __name__ == "__main__":
    main()
