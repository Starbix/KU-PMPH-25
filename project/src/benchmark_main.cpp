#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstring>
#include <torch/torch.h>
#include <cuda_runtime.h>
<<<<<<< Updated upstream
#include "attention.cuh"
#include "flash_attention.cuh"
#include "utils.cu"

int gpuAssert(cudaError_t code) {
  if(code != cudaSuccess) {
    printf("GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
  }
  return 0;
}

=======
>>>>>>> Stashed changes

#include "../include/attention.h"
#include "../include/flash_attention.h"

struct BenchmarkConfig {
    int batch_heads = 8;
    int seq_len = 128;
    int head_dim = 64;
    int num_runs = 10;
    bool verify = false;
    bool verbose = false;
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --batch_heads N    Number of batch * heads (default: 8)\n";
    std::cout << "  --seq_len N        Sequence length (default: 128)\n";
    std::cout << "  --head_dim N       Head dimension (default: 64)\n";
    std::cout << "  --num_runs N       Number of benchmark runs (default: 10)\n";
    std::cout << "  --verify           Verify correctness between implementations\n";
    std::cout << "  --verbose          Print detailed timing information\n";
    std::cout << "  --help             Show this help message\n";
}

<<<<<<< Updated upstream
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
                        int seq_length, int head_dim, int num_runs) {
    double total_time = 0.0;

    // Allocate device memory
    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_output = nullptr; // TODO: substitute placeholders
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

    for (int i = 0; i < num_runs; ++i) {
        // Call the attention kernel
        cudaError_t error = attention::compute<float>(
            d_q, 
            d_k, 
            d_v, 
            seq_length, 
            head_dim, 
            d_output
        );
    }

    // Wait for kernel to finish
    cudaDeviceSynchronize();
    
    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    
    gpuAssert(cudaPeekAtLastError());

    std::chrono::duration<double, std::milli> duration = end - start;
    total_time += duration.count();

    // Copy result back to host
    cudaMemcpy(output, d_output, tensor_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_output);

    return total_time / num_runs;
}

// Benchmark flash attention
double benchmark_flash_attention(float* q, float* k, float* v, float* output,
                              int batch_size, int seq_length, int head_dim, int num_runs) {
    double total_time = 0.0;

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

    for (int i = 0; i < num_runs; ++i) {
        // Call the flash attention kernel
        cudaError_t error = flash_attention::compute<float>(
            d_q, 
            d_k, 
            d_v, 
            seq_length, 
            head_dim, 
            d_output
        );
    }

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // end timing
    auto end = std::chrono::high_resolution_clock::now();
    
    gpuAssert(cudaPeekAtLastError());

    std::chrono::duration<double, std::milli> duration = end - start;
    total_time += duration.count();

    // Copy result back to host
    cudaMemcpy(output, d_output, tensor_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_output);

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
                                            seq_length, args.head_dim,
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
=======
bool parse_args(int argc, char** argv, BenchmarkConfig& config) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            return false;
        } else if (arg == "--batch_heads" && i + 1 < argc) {
            config.batch_heads = std::atoi(argv[++i]);
        } else if (arg == "--seq_len" && i + 1 < argc) {
            config.seq_len = std::atoi(argv[++i]);
        } else if (arg == "--head_dim" && i + 1 < argc) {
            config.head_dim = std::atoi(argv[++i]);
        } else if (arg == "--num_runs" && i + 1 < argc) {
            config.num_runs = std::atoi(argv[++i]);
        } else if (arg == "--verify") {
            config.verify = true;
        } else if (arg == "--verbose") {
            config.verbose = true;
>>>>>>> Stashed changes
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return false;
        }
    }
    
    // Basic validation
    if (config.batch_heads <= 0 || config.seq_len <= 0 || config.head_dim <= 0 || config.num_runs <= 0) {
        std::cerr << "Error: All dimensions and num_runs must be positive" << std::endl;
        return false;
    }
    
    return true;
}

torch::Tensor create_random_tensor(int batch_heads, int seq_len, int head_dim) {
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA)
        .requires_grad(false);
    
    return torch::randn({batch_heads, seq_len, head_dim}, options);
}

double benchmark_implementation(
    torch::Tensor (*impl)(torch::Tensor, torch::Tensor, torch::Tensor),
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int num_runs, const std::string& name, bool verbose
) {
    // Warmup
    auto warmup_result = impl(Q, K, V);
    torch::cuda::synchronize();
    
    std::vector<double> times;
    times.reserve(num_runs);
    
    for (int run = 0; run < num_runs; run++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = impl(Q, K, V);
        torch::cuda::synchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(duration);
        
        if (verbose) {
            std::cout << name << " run " << std::setw(2) << run 
                     << ": " << std::fixed << std::setprecision(3) 
                     << duration << " ms" << std::endl;
        }
    }
    
    // Calculate average
    double total = 0.0;
    for (double time : times) {
        total += time;
    }
    
    return total / num_runs;
}

torch::Tensor torch_reference_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // Reference PyTorch attention implementation
    // TODO: Add scaling factor 1/sqrt(head_dim)
    auto scores = torch::matmul(Q, K.transpose(-2, -1));
    auto probs = torch::softmax(scores, -1);
    auto output = torch::matmul(probs, V);
    return output;
}

bool verify_correctness(torch::Tensor output1, torch::Tensor output2, const std::string& name1, const std::string& name2) {
    // Check if outputs are close (allowing for numerical differences)
    bool is_close = torch::allclose(output1, output2, 1e-4, 1e-4);
    
    if (is_close) {
        std::cout << "✓ Verification PASSED: " << name1 << " and " << name2 << " outputs match" << std::endl;
    } else {
        std::cout << "✗ Verification FAILED: " << name1 << " and " << name2 << " outputs differ" << std::endl;
        
        // Print some statistics about the differences
        auto diff = (output1 - output2).abs();
        auto max_diff = torch::max(diff).item<float>();
        auto mean_diff = torch::mean(diff).item<float>();
        
        std::cout << "  Max difference: " << max_diff << std::endl;
        std::cout << "  Mean difference: " << mean_diff << std::endl;
    }
    
    return is_close;
}

int main(int argc, char** argv) {
    BenchmarkConfig config;
    
    if (!parse_args(argc, argv, config)) {
        return 1;
    }
    
    // Check CUDA availability
    if (!torch::cuda::is_available()) {
        std::cerr << "Error: CUDA is not available" << std::endl;
        return 1;
    }
    
    std::cout << "Attention Implementations Benchmark" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch * Heads: " << config.batch_heads << std::endl;
    std::cout << "  Sequence Length: " << config.seq_len << std::endl;
    std::cout << "  Head Dimension: " << config.head_dim << std::endl;
    std::cout << "  Number of Runs: " << config.num_runs << std::endl;
    std::cout << "  Verification: " << (config.verify ? "enabled" : "disabled") << std::endl;
    std::cout << std::endl;
    
    // Create input tensors
    std::cout << "Creating input tensors..." << std::endl;
    auto Q = create_random_tensor(config.batch_heads, config.seq_len, config.head_dim);
    auto K = create_random_tensor(config.batch_heads, config.seq_len, config.head_dim);
    auto V = create_random_tensor(config.batch_heads, config.seq_len, config.head_dim);
    
    std::cout << "Input tensor shape: [" << config.batch_heads << ", " 
              << config.seq_len << ", " << config.head_dim << "]" << std::endl;
    std::cout << std::endl;
    
    // Benchmark PyTorch reference
    std::cout << "Benchmarking PyTorch reference..." << std::endl;
    double avg_time_torch = benchmark_implementation(
        torch_reference_attention, Q, K, V, config.num_runs, "PyTorch", config.verbose
    );
    
    std::cout << std::endl;
    
    // Benchmark standard attention
    std::cout << "Benchmarking standard attention..." << std::endl;
    double avg_time_attention = benchmark_implementation(
        attention::forward, Q, K, V, config.num_runs, "Attention", config.verbose
    );
    
    std::cout << std::endl;
    
    // Benchmark flash attention
    std::cout << "Benchmarking flash attention..." << std::endl;
    double avg_time_flash = benchmark_implementation(
        flash_attention::forward, Q, K, V, config.num_runs, "Flash", config.verbose
    );
    
    std::cout << std::endl;
    
    // Print results
    std::cout << "Results:" << std::endl;
    std::cout << "--------" << std::endl;
    std::cout << "PyTorch Reference:  " << std::fixed << std::setprecision(3) 
              << avg_time_torch << " ms (avg)" << std::endl;
    std::cout << "Standard Attention: " << std::fixed << std::setprecision(3) 
              << avg_time_attention << " ms (avg)" << std::endl;
    std::cout << "Flash Attention:    " << std::fixed << std::setprecision(3) 
              << avg_time_flash << " ms (avg)" << std::endl;
    
    if (avg_time_attention > 0 && avg_time_flash > 0) {
        double speedup = avg_time_attention / avg_time_flash;
        std::cout << "Speedup (Flash/Std): " << std::fixed << std::setprecision(2) 
                  << speedup << "x" << std::endl;
    }
    
    std::cout << std::endl;
    
    // Verification if requested
    if (config.verify) {
        std::cout << "Running verification..." << std::endl;
        
        auto output_torch = torch_reference_attention(Q, K, V);
        auto output_attention = attention::forward(Q, K, V);
        auto output_flash = flash_attention::forward(Q, K, V);
        
        verify_correctness(output_attention, output_torch, "Standard Attention", "PyTorch Reference");
        verify_correctness(output_flash, output_torch, "Flash Attention", "PyTorch Reference");
        verify_correctness(output_flash, output_attention, "Flash Attention", "Standard Attention");
        std::cout << std::endl;
    }
    
    std::cout << "Benchmark completed successfully!" << std::endl;
    
    return 0;
}