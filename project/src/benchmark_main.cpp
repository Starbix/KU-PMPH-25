#include <iostream>
#include <vector>
#include <sys/time.h>
#include <iomanip>
#include <string>
#include <cstring>
#include <torch/torch.h>
#include <cuda_runtime.h>

inline double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

#define ENABLE_STD_ATTENTION 0
#define ENABLE_FLASH_ATTENTION 0

int gpuAssert(cudaError_t code) {
  if(code != cudaSuccess) {
    printf("GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
  }
  return 0;
}


#include "../include/attention.h"
#include "../include/flash_attention.h"

struct BenchmarkConfig {
    int seq_len = 128;
    int head_dim = 64;
    int num_runs = 10;
    bool verify = true; // verify by default
    bool verbose = false;
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --seq_len N        Sequence length (default: 128)\n";
    std::cout << "  --head_dim N       Head dimension (default: 64)\n";
    std::cout << "  --num_runs N       Number of benchmark runs (default: 10)\n";
    std::cout << "  --verify           Verify correctness between implementations\n";
    std::cout << "  --verbose          Print detailed timing information\n";
    std::cout << "  --help             Show this help message\n";
}


bool parse_args(int argc, char** argv, BenchmarkConfig& config) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage(argv[0]);
            return false;
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
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return false;
        }
    }

    // Basic validation
    if (config.seq_len <= 0 || config.head_dim <= 0 || config.num_runs <= 0) {
        std::cerr << "Error: All dimensions and num_runs must be positive" << std::endl;
        return false;
    }

    return true;
}

torch::Tensor create_random_tensor(int seq_len, int head_dim) {
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA)
        .requires_grad(false);

    return torch::randn({seq_len, head_dim}, options);
}

double benchmark_implementation(
    double (*duration_impl)(torch::Tensor, torch::Tensor, torch::Tensor),
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int num_runs, const std::string& name, bool verbose
) {
    // Warmup
    double warmup = duration_impl(Q, K, V);
    torch::cuda::synchronize();

    double total_time = 0.0;

    for (int run = 0; run < num_runs; run++) {
        double duration = duration_impl(Q, K, V);
        total_time += duration;
    }

    double avg_time = total_time / num_runs;

    if (verbose) {
        std::cout << name << " total time: " << std::fixed << std::setprecision(3)
                 << total_time << " ms, avg: " << avg_time << " ms" << std::endl;
    }

    return avg_time;
}

torch::Tensor torch_reference_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // Reference PyTorch attention implementation
    // TODO: Add scaling factor 1/sqrt(head_dim)
    auto scores = torch::matmul(Q, K.t());
    auto probs = torch::softmax(scores, -1);
    auto output = torch::matmul(probs, V);
    return output;
}

double torch_reference_attention_duration(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    double start, end;

    cudaDeviceSynchronize();
    start = getTimeStamp();

    auto scores = torch::matmul(Q, K.t());
    auto probs = torch::softmax(scores, -1);
    auto output = torch::matmul(probs, V);

    cudaDeviceSynchronize();
    end = getTimeStamp();

    return (end - start) * 1000.0; // convert to milliseconds
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
    std::cout << "  Sequence Length: " << config.seq_len << std::endl;
    std::cout << "  Head Dimension: " << config.head_dim << std::endl;
    std::cout << "  Number of Runs: " << config.num_runs << std::endl;
    std::cout << "  Verification: " << (config.verify ? "enabled" : "disabled") << std::endl;
    std::cout << std::endl;

    // Create input tensors
    std::cout << "Creating input tensors..." << std::endl;
    auto Q = create_random_tensor(config.seq_len, config.head_dim);
    auto K = create_random_tensor(config.seq_len, config.head_dim);
    auto V = create_random_tensor(config.seq_len, config.head_dim);

    std::cout << "Input tensor shape: [" << config.seq_len << ", " << config.head_dim << "]" << std::endl;
    std::cout << std::endl;

    // Benchmark PyTorch reference
    std::cout << "Benchmarking PyTorch reference..." << std::endl;
    double avg_time_torch = benchmark_implementation(
        torch_reference_attention_duration, Q, K, V, config.num_runs, "PyTorch", config.verbose
    );

    std::cout << std::endl;

    #if ENABLE_STD_ATTENTION
    // Benchmark standard attention
    std::cout << "Benchmarking standard attention..." << std::endl;
    double avg_time_attention = benchmark_implementation(
        attention::forward_duration, Q, K, V, config.num_runs, "Attention", config.verbose
    );

    std::cout << std::endl;
    #endif

    #if ENABLE_FLASH_ATTENTION
    // Benchmark flash attention (regular)
    std::cout << "Benchmarking flash attention (regular)..." << std::endl;
    double avg_time_flash = benchmark_implementation(
        flash_attention::forward_duration, Q, K, V, config.num_runs, "Flash", config.verbose
    );

    std::cout << std::endl;
    #endif
    std::cout << std::endl;

    // Print results
    std::cout << "Results:" << std::endl;
    std::cout << "--------" << std::endl;
    std::cout << "PyTorch Reference:       " << std::fixed << std::setprecision(3)
              << avg_time_torch << " ms (avg)" << std::endl;
    #if ENABLE_STD_ATTENTION
    std::cout << "Standard Attention:      " << std::fixed << std::setprecision(3)
              << avg_time_attention << " ms (avg)" << std::endl;
    #endif
    #if ENABLE_FLASH_ATTENTION
    std::cout << "Flash Attention:         " << std::fixed << std::setprecision(3)
              << avg_time_flash << " ms (avg)" << std::endl;
    #endif
    std::cout << "Flash Attention (Tiled): " << std::fixed << std::setprecision(3)
              << avg_time_flash_tiled << " ms (avg)" << std::endl;

    #if ENABLE_STD_ATTENTION
    if (avg_time_torch > 0 && avg_time_attention > 0) {
        double speedup_std = avg_time_torch / avg_time_attention;
        std::cout << "Speedup (Std/PyTorch): " << std::fixed << std::setprecision(2)
                  << speedup_std << "x" << std::endl;
    }
    #endif

    #if ENABLE_FLASH_ATTENTION
    if (avg_time_torch > 0 && avg_time_flash > 0) {
        double speedup = avg_time_torch / avg_time_flash;
        std::cout << "Speedup (Flash/PyTorch): " << std::fixed << std::setprecision(2)
                  << speedup << "x" << std::endl;
    }
    #endif

    if (avg_time_torch > 0 && avg_time_flash_tiled > 0) {
        double speedup_tiled = avg_time_torch / avg_time_flash_tiled;
        std::cout << "Speedup (Flash-Tiled/PyTorch): " << std::fixed << std::setprecision(2)
                  << speedup_tiled << "x" << std::endl;
    }

    std::cout << std::endl;

    // Verification if requested
    if (config.verify) {
        std::cout << "Running verification..." << std::endl;

        auto output_torch = torch_reference_attention(Q, K, V);
        #if ENABLE_STD_ATTENTION
        auto output_attention = attention::forward(Q, K, V);
        #endif
        #if ENABLE_FLASH_ATTENTION
        auto output_flash = flash_attention::forward(Q, K, V);
        #endif
        auto output_flash_tiled = flash_attention::forward_tiled(Q, K, V);

        #if ENABLE_STD_ATTENTION
        verify_correctness(output_attention, output_torch, "Standard Attention", "PyTorch Reference");
        #endif
        #if ENABLE_FLASH_ATTENTION
        verify_correctness(output_flash, output_torch, "Flash Attention", "PyTorch Reference");
        #if ENABLE_STD_ATTENTION
        verify_correctness(output_flash, output_attention, "Flash Attention", "Standard Attention");
        #endif
        #endif
        verify_correctness(output_flash_tiled, output_torch, "Flash Attention (Tiled)", "PyTorch Reference");
        #if ENABLE_STD_ATTENTION
        verify_correctness(output_flash_tiled, output_attention, "Flash Attention (Tiled)", "Standard Attention");
        #endif
        std::cout << std::endl;
    }

    std::cout << "Benchmark completed successfully!" << std::endl;

    return 0;
}
