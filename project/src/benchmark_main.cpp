#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstring>
#include <torch/torch.h>
#include <cuda_runtime.h>


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
    bool profile_kernels = false;
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --seq_len N        Sequence length (default: 128)\n";
    std::cout << "  --head_dim N       Head dimension (default: 64)\n";
    std::cout << "  --num_runs N       Number of benchmark runs (default: 10)\n";
    std::cout << "  --verify           Verify correctness between implementations\n";
    std::cout << "  --verbose          Print detailed timing information\n";
    std::cout << "  --profile          Enable kernel-level profiling\n";
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
        } else if (arg == "--profile") {
            config.profile_kernels = true;
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

// Forward declaration for profiling version
cudaError_t compute_with_profiling(float* Q, float* K, float* V, uint32_t N, uint32_t d, float* O);

double benchmark_implementation(
    torch::Tensor (*impl)(torch::Tensor, torch::Tensor, torch::Tensor),
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int num_runs, const std::string& name, bool verbose, bool profile_kernels = false
) {
    // Warmup
    auto warmup_result = impl(Q, K, V);
    torch::cuda::synchronize();

    std::vector<double> times;
    times.reserve(num_runs);

    for (int run = 0; run < num_runs; run++) {
        if (profile_kernels && run == 0 && name == "Attention") {
            // Special profiling run for standard attention
            std::cout << "\n  Kernel profiling for " << name << ":" << std::endl;

            auto Q_cont = Q.contiguous();
            auto K_cont = K.contiguous();
            auto V_cont = V.contiguous();
            auto O = torch::zeros_like(Q);

            auto start = std::chrono::high_resolution_clock::now();

            cudaError_t err = compute_with_profiling(
                Q_cont.data_ptr<float>(),
                K_cont.data_ptr<float>(),
                V_cont.data_ptr<float>(),
                Q.size(0), Q.size(1),
                O.data_ptr<float>()
            );

            torch::cuda::synchronize();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::milli>(end - start).count();
            times.push_back(duration);

            if (err != cudaSuccess) {
                std::cerr << "CUDA Error in profiling: " << cudaGetErrorString(err) << std::endl;
            }
        } else {
            auto start = std::chrono::high_resolution_clock::now();

            auto result = impl(Q, K, V);
            torch::cuda::synchronize();

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::milli>(end - start).count();
            times.push_back(duration);
        }

        if (verbose) {
            std::cout << name << " run " << std::setw(2) << run
                     << ": " << std::fixed << std::setprecision(3)
                     << times.back() << " ms" << std::endl;
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
    auto scores = torch::matmul(Q, K.t());
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
    std::cout << "  Sequence Length: " << config.seq_len << std::endl;
    std::cout << "  Head Dimension: " << config.head_dim << std::endl;
    std::cout << "  Number of Runs: " << config.num_runs << std::endl;
    std::cout << "  Verification: " << (config.verify ? "enabled" : "disabled") << std::endl;
    std::cout << "  Kernel Profiling: " << (config.profile_kernels ? "enabled" : "disabled") << std::endl;
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
        torch_reference_attention, Q, K, V, config.num_runs, "PyTorch", config.verbose, false
    );

    std::cout << std::endl;

    // Benchmark standard attention
    std::cout << "Benchmarking standard attention..." << std::endl;
    double avg_time_attention = benchmark_implementation(
        attention::forward, Q, K, V, config.num_runs, "Attention", config.verbose, config.profile_kernels
    );

    std::cout << std::endl;

    // Benchmark flash attention
    std::cout << "Benchmarking flash attention..." << std::endl;
    double avg_time_flash = benchmark_implementation(
        flash_attention::forward, Q, K, V, config.num_runs, "Flash", config.verbose, false
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
