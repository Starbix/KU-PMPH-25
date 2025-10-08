# PMPH 2025 Project: FlashAttention

## Authors: Daniel Sommer and Cédric Laubacher

This project implements standard Attention and FlashAttention in CUDA, with Python bindings for benchmarking and comparison.

## Project Structure

```
project/
├── CMakeLists.txt        # Main CMake configuration file
├── include/              # Header files
│   ├── attention.cuh     # Header for standard attention
│   └── flash_attention.cuh # Header for flash attention
├── python/               # Python scripts for calling C++ benchmarks
│   └── run_benchmark.py  # Python script for running C++ benchmarks and visualizing results
└── src/                  # Source files
    ├── attention/        # Standard attention implementation
    │   └── attention.cu  # CUDA implementation of standard attention
    ├── flash_attention/  # FlashAttention implementation
    │   └── flash_attention.cu # CUDA implementation of FlashAttention
    └── benchmark_main.cpp # Standalone C++ benchmark executable
```

## Prerequisites

- CUDA (12.8)
- CMake (4.0)
- Python (3.12)

## Setting Up the Virtual Environment


###
```bash
# on hendrix, load modules
module load cuda/12.8 cmake/4.0.3 python/3.12.8 gcc/13.2.0
# Make the script executable
chmod +x setup_env.sh

# Run the setup script
./setup_env.sh
```

After setting up, activate the virtual environment: `source venv/bin/activate`

Install the required Python dependencies:

```bash
pip3 install -r requirements.txt
```

## Building the Project

### Build Instructions

```bash
# Make sure you're in the project root directory
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

This will compile the CUDA code and create the Python bindings.

Running the benchmarks:
```bash
# From the project root directory, using C++ executable directly
./build/benchmark --batch_size 2 --num_heads 8 --head_dim 64 --verify

# Or using the Python script for visualization (recommended for plots)
python3 python/run_benchmark.py --batch_size 2 --num_heads 8 --verify --output benchmark_results.png
```

### Benchmarking Options

#### C++ Benchmark Executable
```
Usage: benchmark [options]
Options:
  --batch_size <size>       Batch size for the test (default: 2)
  --num_heads <num>         Number of attention heads (default: 8)
  --head_dim <dim>          Dimension of each attention head (default: 64)
  --num_runs <runs>         Number of runs for each benchmark (default: 10)
  --seq_lengths <list>      Comma-separated list of sequence lengths (default: 128,256,512,1024,2048,4096,8192)
  --test_kernel_only        Only test the fill_ones kernel with a small matrix
  --output <file>           Output file path for benchmark results (default: stdout)
  --csv                     Output results in CSV format
  --verify                  Verify correctness between implementations
  --help                    Display this help message and exit
```

#### Python Visualization Script
```
usage: run_benchmark.py [-h] [--bin_dir BIN_DIR] [--batch_size BATCH_SIZE] [--num_heads NUM_HEADS]
                        [--head_dim HEAD_DIM] [--num_runs NUM_RUNS] [--seq_lengths SEQ_LENGTHS]
                        [--test_kernel_only] [--verify] [--output OUTPUT]

Run attention benchmarks

optional arguments:
  -h, --help            show this help message and exit
  --bin_dir BIN_DIR     Directory containing the benchmark executable
  --batch_size BATCH_SIZE
                        Batch size for the test
  --num_heads NUM_HEADS
                        Number of attention heads
  --head_dim HEAD_DIM   Dimension of each attention head
  --num_runs NUM_RUNS   Number of runs for each benchmark
  --seq_lengths SEQ_LENGTHS
                        Comma-separated list of sequence lengths to benchmark
  --test_kernel_only    Only test the fill_ones kernel with a small matrix
  --verify              Verify correctness between implementations
  --output OUTPUT       Output path for the benchmark plot
```

## Implementation Details

TODO
