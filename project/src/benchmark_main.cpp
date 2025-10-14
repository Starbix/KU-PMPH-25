#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include "attention.cuh"
#include "flash_attention.cuh"

// Utility function to print help message
void print_help() {
    std::cout << "Usage: benchmark [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --batch_size <size>       Batch size for the test (default: 2)" << std::endl;
    std::cout << "  --num_heads <num>         Number of attention heads (default: 8)" << std::endl;
    std::cout << "  --head_dim <dim>          Dimension of each attention head (default: 64)" << std::endl;
    std::cout << "  --num_runs <runs>         Number of runs for each benchmark (default: 10)" << std::endl;
    std::cout << "  --seq_lengths <list>      Comma-separated list of sequence lengths to benchmark (default: 128,256,512,1024,2048,4096,8192)" << std::endl;
    std::cout << "  --test_kernel_only        Only test the fill_ones kernel with a small matrix" << std::endl;
    std::cout << "  --output <file>           Output file path for benchmark results (default: stdout)" << std::endl;
    std::cout << "  --csv                     Output results in CSV format" << std::endl;
    std::cout << "  --verify                  Verify correctness between implementations" << std::endl;
    std::cout << "  --help                    Display this help message and exit" << std::endl;
}

// Parse command-line arguments
class Arguments {
public:
    int batch_size = 2;
    int num_heads = 8;
    int head_dim = 64;
    int num_runs = 10;
    std::vector<int> seq_lengths = {128, 256, 512, 1024, 2048, 4096, 8192};
    bool test_kernel_only = false;
    std::string output_file;
    bool csv_format = false;
    bool verify = false;

    Arguments(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];

            if (arg == "--help") {
                print_help();
                exit(0);
            } else if (arg == "--batch_size" && i + 1 < argc) {
                batch_size = std::stoi(argv[++i]);
            } else if (arg == "--num_heads" && i + 1 < argc) {
                num_heads = std::stoi(argv[++i]);
            } else if (arg == "--head_dim" && i + 1 < argc) {
                head_dim = std::stoi(argv[++i]);
            } else if (arg == "--num_runs" && i + 1 < argc) {
                num_runs = std::stoi(argv[++i]);
            } else if (arg == "--seq_lengths" && i + 1 < argc) {
                seq_lengths.clear();
                std::string lengths_str = argv[++i];
                size_t pos = 0;
                std::string token;
                while ((pos = lengths_str.find(',')) != std::string::npos) {
                    token = lengths_str.substr(0, pos);
                    seq_lengths.push_back(std::stoi(token));
                    lengths_str.erase(0, pos + 1);
                }
                if (!lengths_str.empty()) {
                    seq_lengths.push_back(std::stoi(lengths_str));
                }
            } else if (arg == "--test_kernel_only") {
                test_kernel_only = true;
            } else if (arg == "--output" && i + 1 < argc) {
                output_file = argv[++i];
            } else if (arg == "--csv") {
                csv_format = true;
            } else if (arg == "--verify") {
                verify = true;
            } else {
                std::cerr << "Unknown argument: " << arg << std::endl;
                print_help();
                exit(1);
            }
        }
    }
};

// Utility function to print matrix for debugging
void print_matrix(const float* matrix, int batch_size, int rows, int cols, const std::string& name,
                  int max_batch = 1, int max_rows = 5, int max_cols = 5) {
    std::cout << name << " shape: [" << batch_size << ", " << rows << ", " << cols << "]" << std::endl;

    max_batch = std::min(max_batch, batch_size);
    max_rows = std::min(max_rows, rows);
    max_cols = std::min(max_cols, cols);

    for (int b = 0; b < max_batch; ++b) {
        std::cout << "Batch " << b << ":" << std::endl;
        for (int r = 0; r < max_rows; ++r) {
            std::cout << "  ";
            for (int c = 0; c < max_cols; ++c) {
                int idx = b * (rows * cols) + r * cols + c;
                std::cout << std::fixed << std::setprecision(2) << std::setw(6) << matrix[idx];
            }
            if (cols > max_cols) {
                std::cout << " ...";
            }
            std::cout << std::endl;
        }
        if (rows > max_rows) {
            std::cout << "  ..." << std::endl;
        }
        std::cout << std::endl;
    }
}

// Generate random test data
void generate_test_data(float* q, float* k, float* v,
                        int batch_size, int seq_length, int num_heads, int head_dim) {
    // Set up random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Calculate total elements
    int elements_per_tensor = batch_size * num_heads * seq_length * head_dim;

    // Fill tensors with random values
    for (int i = 0; i < elements_per_tensor; ++i) {
        q[i] = dist(gen);
        k[i] = dist(gen);
        v[i] = dist(gen);
    }
}

// Reshape tensors from [batch_size, seq_length, num_heads, head_dim] to [batch_size*num_heads, seq_length, head_dim]
void reshape_for_attention(float* src, float* dst,
                           int batch_size, int seq_length, int num_heads, int head_dim) {
    // For each element in the source tensor
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_length; ++s) {
            for (int h = 0; h < num_heads; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    // Calculate source index [b, s, h, d]
                    int src_idx = b * (seq_length * num_heads * head_dim) +
                                s * (num_heads * head_dim) +
                                h * head_dim + d;

                    // Calculate destination index [b*num_heads + h, s, d]
                    int dst_idx = (b * num_heads + h) * (seq_length * head_dim) +
                                s * head_dim + d;

                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

// Benchmark standard attention
double benchmark_attention(float* q, float* k, float* v, float* output,
                        int batch_size, int seq_length, int head_dim, int num_runs) {
    double total_time = 0.0;

    for (int i = 0; i < num_runs; ++i) {
        // Allocate device memory½
        float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_output = nullptr;
        size_t tensor_size = seq_length * head_dim * sizeof(float);

        cudaMalloc(&d_q, tensor_size);
        cudaMalloc(&d_k, tensor_size);
        cudaMalloc(&d_v, tensor_size);
        cudaMalloc(&d_output, tensor_size);

        // Copy data to device
        cudaMemcpy(d_q, q, tensor_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_k, k, tensor_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, v, tensor_size, cudaMemcpyHostToDevice);

        // Start timing
        auto start = std::chrono::high_resolution_clock::now(); 

        // Call the attention kernel
        // attention::fill_ones(d_k, batch_size, seq_length, head_dim);
        attention::compute_attention<float>(
            d_q, 
            d_k, 
            d_v, 
            seq_length, 
            head_dim, 
            d_output
        );

        // Wait for kernel to finish
        cudaDeviceSynchronize();

        // End timing
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        total_time += duration.count();

        // Copy result back to host
        cudaMemcpy(output, d_output, tensor_size, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_output);
    }

    return total_time / num_runs;
}

// Benchmark flash attention
double benchmark_flash_attention(float* q, float* k, float* v, float* output,
                              int batch_size, int seq_length, int head_dim, int num_runs) {
    double total_time = 0.0;

    for (int i = 0; i < num_runs; ++i) {
        // Allocate device memory
        float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_output = nullptr;
        size_t tensor_size = batch_size * seq_length * head_dim * sizeof(float);

        cudaMalloc(&d_q, tensor_size);
        cudaMalloc(&d_k, tensor_size);
        cudaMalloc(&d_v, tensor_size);
        cudaMalloc(&d_output, tensor_size);

        // Copy data to device
        cudaMemcpy(d_q, q, tensor_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_k, k, tensor_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, v, tensor_size, cudaMemcpyHostToDevice);

        // Start timing
        auto start = std::chrono::high_resolution_clock::now();

        // Call the flash attention kernel
        //flash_attention::compute(d_q, d_k, d_v, d_output, batch_size, seq_length, head_dim);

        // Wait for kernel to finish
        cudaDeviceSynchronize();

        // End timing
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        total_time += duration.count();

        // Copy result back to host
        cudaMemcpy(output, d_output, tensor_size, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_output);
    }

    return total_time / num_runs;
}

// Compare implementations and calculate error metrics
void compare_implementations(float* std_output, float* flash_output, int total_elements,
                           double& max_abs_error, double& mean_abs_error) {
    max_abs_error = 0.0;
    mean_abs_error = 0.0;

    for (int i = 0; i < total_elements; ++i) {
        double abs_err = std::abs(std_output[i] - flash_output[i]);
        max_abs_error = std::max(max_abs_error, abs_err);
        mean_abs_error += abs_err;
    }

    mean_abs_error /= total_elements;
}

// Test the fill_ones kernel
void test_fill_ones_kernel() {
    std::cout << "\n=== Testing fill_ones kernel on K matrix ===" << std::endl;

    // Create small test matrices
    const int batch_size = 1;
    const int seq_len = 4;
    const int head_dim = 4;
    const int total_elements = batch_size * seq_len * head_dim;

    // Allocate host memory
    float* k = new float[total_elements];

    // Generate random data
    for (int i = 0; i < total_elements; ++i) {
        k[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Print original K matrix
    std::cout << "Original K matrix:" << std::endl;
    print_matrix(k, batch_size, seq_len, head_dim, "K");

    // Allocate device memory
    float* d_k = nullptr;
    cudaMalloc(&d_k, total_elements * sizeof(float));
    cudaMemcpy(d_k, k, total_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Execute kernel
    std::cout << "Running fill_ones kernel..." << std::endl;
    attention::fill_ones(d_k, batch_size, seq_len, head_dim);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(k, d_k, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_k);

    // Print K matrix after kernel execution
    std::cout << "K matrix after fill_ones kernel:" << std::endl;
    print_matrix(k, batch_size, seq_len, head_dim, "K");

    // Verify all elements are 1.0
    bool all_ones = true;
    for (int i = 0; i < total_elements; ++i) {
        if (std::abs(k[i] - 1.0f) > 1e-5) {
            all_ones = false;
            break;
        }
    }

    std::cout << "Matrix contains all ones: " << (all_ones ? "YES" : "NO") << std::endl;
    if (!all_ones) {
        std::cerr << "ERROR: The fill_ones kernel did not properly fill the matrix with ones!" << std::endl;
    }

    std::cout << "===================================" << std::endl;

    // Clean up
    delete[] k;
}

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    Arguments args(argc, argv);

    // Handle test_kernel_only mode
    if (args.test_kernel_only) {
        test_fill_ones_kernel();
        return 0;
    }

    // Open output file if specified
    std::ofstream output_file;
    std::streambuf* original_cout = nullptr;

    if (!args.output_file.empty()) {
        output_file.open(args.output_file);
        if (!output_file.is_open()) {
            std::cerr << "Error: Could not open output file: " << args.output_file << std::endl;
            return 1;
        }
        // Redirect cout to file
        original_cout = std::cout.rdbuf();
        std::cout.rdbuf(output_file.rdbuf());
    }

    // Print header for CSV format if requested
    if (args.csv_format) {
        std::cout << "seq_length,std_time_ms,flash_time_ms,speedup";
        if (args.verify) {
            std::cout << ",max_abs_error,mean_abs_error";
        }
        std::cout << std::endl;
    } else {
        std::cout << "Attention Benchmarking" << std::endl;
        std::cout << "======================" << std::endl;
        std::cout << "Batch size: " << args.batch_size << std::endl;
        std::cout << "Number of heads: " << args.num_heads << std::endl;
        std::cout << "Head dimension: " << args.head_dim << std::endl;
        std::cout << "Number of runs per test: " << args.num_runs << std::endl;
        std::cout << std::endl;
    }

    // Run benchmarks for each sequence length
    double max_speedup = 0.0;
    int max_speedup_seq_len = 0;

    for (int seq_length : args.seq_lengths) {
        if (!args.csv_format) {
            std::cout << "Benchmarking sequence length: " << seq_length << std::endl;
        }

        // Compute total elements per tensor
        const int total_elements_bhsd = args.batch_size * seq_length * args.num_heads * args.head_dim;
        const int total_elements_bsd = args.batch_size * args.num_heads * seq_length * args.head_dim;

        // Allocate host memory for input tensors in BHSD format (batch, seq, heads, dim)
        float* q_bhsd = new float[total_elements_bhsd];
        float* k_bhsd = new float[total_elements_bhsd];
        float* v_bhsd = new float[total_elements_bhsd];

        // Generate random test data
        generate_test_data(q_bhsd, k_bhsd, v_bhsd, args.batch_size, seq_length, args.num_heads, args.head_dim);

        // Reshape to BSD format (batch*heads, seq, dim) for the kernels
        float* q_bsd = new float[total_elements_bsd];
        float* k_bsd = new float[total_elements_bsd];
        float* v_bsd = new float[total_elements_bsd];

        reshape_for_attention(q_bhsd, q_bsd, args.batch_size, seq_length, args.num_heads, args.head_dim);
        reshape_for_attention(k_bhsd, k_bsd, args.batch_size, seq_length, args.num_heads, args.head_dim);
        reshape_for_attention(v_bhsd, v_bsd, args.batch_size, seq_length, args.num_heads, args.head_dim);

        // Allocate memory for output
        float* std_output = new float[total_elements_bsd];
        float* flash_output = new float[total_elements_bsd];

        // Run benchmarks
        const int effective_batch_size = args.batch_size * args.num_heads;
        double std_time = benchmark_attention(q_bsd, k_bsd, v_bsd, std_output,
                                            effective_batch_size, seq_length, args.head_dim,
                                            args.num_runs);

        double flash_time = benchmark_flash_attention(q_bsd, k_bsd, v_bsd, flash_output,
                                                   effective_batch_size, seq_length, args.head_dim,
                                                   args.num_runs);

        double speedup = std_time / flash_time;

        // Update max speedup info
        if (speedup > max_speedup) {
            max_speedup = speedup;
            max_speedup_seq_len = seq_length;
        }

        // Calculate error metrics if verification is requested
        double max_abs_error = 0.0;
        double mean_abs_error = 0.0;
        if (args.verify) {
            compare_implementations(std_output, flash_output, total_elements_bsd,
                                   max_abs_error, mean_abs_error);
        }

        // Output results
        if (args.csv_format) {
            std::cout << seq_length << "," << std_time << "," << flash_time << "," << speedup;
            if (args.verify) {
                std::cout << "," << max_abs_error << "," << mean_abs_error;
            }
            std::cout << std::endl;
        } else {
            std::cout << "  Standard Attention: " << std::fixed << std::setprecision(4) << std_time << " ms" << std::endl;
            std::cout << "  Flash Attention:    " << std::fixed << std::setprecision(4) << flash_time << " ms" << std::endl;
            std::cout << "  Speedup:            " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;

            if (args.verify) {
                std::cout << "  Max Error:          " << std::scientific << std::setprecision(6) << max_abs_error << std::endl;
                std::cout << "  Mean Error:         " << std::scientific << std::setprecision(6) << mean_abs_error << std::endl;
            }
            std::cout << std::endl;
        }

        // Clean up
        delete[] q_bhsd;
        delete[] k_bhsd;
        delete[] v_bhsd;
        delete[] q_bsd;
        delete[] k_bsd;
        delete[] v_bsd;
        delete[] std_output;
        delete[] flash_output;
    }

    // Output summary
    if (!args.csv_format) {
        std::cout << "Summary:" << std::endl;
        std::cout << "Maximum speedup: " << std::fixed << std::setprecision(2)
                  << max_speedup << "x at sequence length " << max_speedup_seq_len << std::endl;
    }

    // Restore cout if redirected
    if (original_cout) {
        std::cout.rdbuf(original_cout);
        output_file.close();
    }

    return 0;
}
