# PMPH 2025 Project: FlashAttention

## Authors: Daniel Sommer and Cédric Laubacher

This project implements standard Attention and FlashAttention in CUDA, with Python bindings for benchmarking and comparison.

To run our benchmarking scripts you first have to set up a Python environment, then you can run the different python files within the `python` folder.

## Setting up the virtual environment

1. **Load modules.** Run the following in the root of the project:
```bash
module load cuda/12.8 python/3.12.8 gcc/13.2.0 ninja/1.8.2
```

2. **Load Python requirements.** Run the following in the root of the project:
```bash
pip3 install -r requirements.txt
```

## Running the Python scripts:

You have to be in the root of the project folder to run the python scripts. The following is an example:

```bash
python python/plotting_benchmarks.py 
```


## Building the attention modules

If you want to, you can build our C++ and Cuda code that implements the attention algorithms. However, this is not necessary to test our code or run our benchmarking - the Python scripts described above both benchmarks and validates implementations.

To build the attention modules do the following. 

1. **Set up the environment:** Run the following in the root of the project directory:
```bash
# on hendrix, load modules
module load cuda/12.8 cmake/4.0.3 python/3.12.8 gcc/13.2.0 ninja/1.8.2
# Make the script executable
chmod +x setup_env.sh

# Run the setup script
./setup_env.sh
```
2. **Activate the environment.**
After setting up, activate the virtual environment: `source venv/bin/activate`

3. **Build the project.** Build the Cuda and C++ files by running the following in the root of the project directory:

```bash
# Make sure you're in the project root directory
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```



## Python Script Options

The following are descriptions of the options you can pass to our Python scripts.

### Benchmarks.py
```
usage: benchmark.py [-h] [--seq_len SEQ_LEN] [--head_dim HEAD_DIM] [--Bc BC] [--Br BR] [--bdimx BDIMX] [--bdimy BDIMY]

Benchmarking

options:
  -h, --help           show this help message and exit
  --seq_len SEQ_LEN    Sequence length (default 128)
  --head_dim HEAD_DIM  Head dimension (default 64)
  --Bc BC              B_c (default 32)
  --Br BR              B_r (default 16)
  --bdimx BDIMX        bdim_x (default 32)
  --bdimy BDIMY        bdim_y (default 16)

```

### Plotting_benchmarks.py

```
usage: plotting_benchmarks.py [-h] [--head_dim HEAD_DIM] [--seq_lens SEQ_LENS [SEQ_LENS ...]] [--Bc BC] [--Br BR] [--bdimx BDIMX] [--bdimy BDIMY] [--file_path FILE_PATH]

Plotting benchmarks

options:
  -h, --help                         show this help message and exit
  --head_dim HEAD_DIM                Head dimension (default 64)
  --seq_lens SEQ_LENS [SEQ_LENS ...] The list of seq. lengths to plot runtime against
  --Bc BC                            B_c (default 32)
  --Br BR                            B_r (default 16)
  --bdimx BDIMX                      bdim_x (default 32)
  --bdimy BDIMY                      bdim_y (default 16)
  --file_path FILE_PATH              The filepath of the resulting plot
```

### Plotting_out_of_memory.py
```
usage: plotting_out_of_memory.py [-h] [--head_dim HEAD_DIM] [--seq_lens SEQ_LENS [SEQ_LENS ...]] [--Bc BC] [--Br BR] [--bdimx BDIMX] [--bdimy BDIMY] [--file_path FILE_PATH]

Plotting out of memory

options:
  -h, --help                         show this help message and exit
  --head_dim HEAD_DIM                Head dimension (default 64)
  --seq_lens SEQ_LENS [SEQ_LENS ...] The list of seq. lengths to plot runtime against
  --Bc BC                            B_c (default 32)
  --Br BR                            B_r (default 16)
  --bdimx BDIMX                      bdim_x (default 32)
  --bdimy BDIMY                      bdim_y (default 16)
  --file_path FILE_PATH              The filepath of the resulting plot
```

### Parameter_optimization.py
```
usage: parameter_optimization.py [-h] [--seq_len SEQ_LEN [SEQ_LEN ...]] [--head_dim HEAD_DIM] [--with_standard] [--no_pytorch]

Parameter optimization

options:
  -h, --help                      show this help message and exit
  --seq_len SEQ_LEN [SEQ_LEN ...] Sequence length(s) (default [1024, 4096, 16384])
  --head_dim HEAD_DIM             Head dimension (default 64)
  --with_standard                 Whether or not optimization should be done with comparison of standard attention.
  --no_pytorch                    Disable comparison with PyTorch's matmul-based  attention implementation.
```


