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
├── python/               # Python binding and benchmarking code
│   ├── attention_benchmark.cpp # C++ bindings for Python
│   └── benchmark.py      # Python benchmarking script
└── src/                  # Source files
    ├── attention/        # Standard attention implementation
    │   └── attention.cu  # CUDA implementation of standard attention
    └── flash_attention/  # FlashAttention implementation
        └── flash_attention.cu # CUDA implementation of FlashAttention
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
# From the project root directory
# export PYTHONPATH=$PYTHONPATH:$(pwd)/build
python3 python/benchmark.py
```

### Benchmarking Options

```
usage: benchmark.py [-h] [--batch_size BATCH_SIZE] [--num_heads NUM_HEADS]
                   [--head_dim HEAD_DIM] [--num_runs NUM_RUNS] [--verify]
                   [--output OUTPUT]

Benchmark attention mechanisms

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Batch size
  --num_heads NUM_HEADS
                        Number of attention heads
  --head_dim HEAD_DIM   Dimension of each attention head
  --num_runs NUM_RUNS   Number of runs for each benchmark
  --verify              Verify correctness between implementations
  --output OUTPUT       Output path for the benchmark plot
```

## Implementation Details

TODO
