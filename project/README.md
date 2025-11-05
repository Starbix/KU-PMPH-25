# PMPH 2025 Project: FlashAttention

## Authors: Daniel Sommer and Cédric Laubacher

This project implements standard Attention and FlashAttention in CUDA, with Python bindings for benchmarking and comparison.

To run our benchmarking scripts you first have to set up a Python environment, then you can run the different python files within the `python` folder.

## Setting up the virtual environment

1. **Load modules.** Run the following in the root of the project:
```bash
module load cuda/12.8 python/3.12.8 gcc/13.2.0 ninja/1.8.2
```

2. **Setting up virtual Python environment.** Run the following in the root of the project to create a Python virtual environment and load all requirements from `requirements.txt`:
```bash
./setup_env.sh
```

## Running the Python scripts:

You have to be in the root of the project folder to run the python scripts. The following is an example:

```bash
python3 python/run_benchmark.py --verify
```


## Python Script Options

The following are descriptions of the options you can pass to our Python scripts.

### Run_benchmark.py
```
usage: run_benchmark.py [-h] [--seq_lengths SEQ_LENGTHS] [--head_dim HEAD_DIM] [--num_runs NUM_RUNS] [--verify] [--output OUTPUT] [--sweep] [--hyperparam-search]

Benchmark attention implementations

options:
  -h, --help                 show this help message and exit
  --seq_lengths SEQ_LENGTHS  Comma-separated list of sequence lengths (default: 1024)
  --head_dim HEAD_DIM        Head dimension (default: 64)
  --num_runs NUM_RUNS        Number of benchmark runs (default: 10)
  --verify                   Verify correctness between implementations
  --output OUTPUT            Output path for benchmark plot
  --sweep                    Run sequence length sweep instead of single benchmark
  --hyperparam-search        Run hyperparameter search to find optimal block sizes and  thread dimensions

```

