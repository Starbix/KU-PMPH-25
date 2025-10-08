#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <chrono>
#include <vector>
#include "attention.cuh"
#include "flash_attention.cuh"

namespace py = pybind11;

// Benchmarking function for standard attention
std::pair<double, py::array_t<float>> benchmark_attention(
    py::array_t<float> q,
    py::array_t<float> k,
    py::array_t<float> v,
    int num_runs = 10
) {
    // Access the data from numpy arrays
    py::buffer_info q_info = q.request();
    py::buffer_info k_info = k.request();
    py::buffer_info v_info = v.request();

    // Create output array with the same shape as expected for attention output
    auto q_shape = q.shape();
    auto v_shape = v.shape();
    
    // Assume output has shape [batch_size, seq_len, head_dim]
    std::vector<ssize_t> output_shape = {q_shape[0], q_shape[1], v_shape[2]};
    py::array_t<float> output = py::array_t<float>(output_shape);
    
    // Benchmark
    double total_time = 0.0;
    
    // Placeholder for actual CUDA kernel call
    // In the real implementation, we would call the CUDA kernel here
    for (int i = 0; i < num_runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Placeholder for CUDA attention computation
        // attention::compute(...);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        total_time += duration.count();
    }
    
    return {total_time / num_runs, output};
}

// Benchmarking function for flash attention
std::pair<double, py::array_t<float>> benchmark_flash_attention(
    py::array_t<float> q,
    py::array_t<float> k,
    py::array_t<float> v,
    int num_runs = 10
) {
    // Access the data from numpy arrays
    py::buffer_info q_info = q.request();
    py::buffer_info k_info = k.request();
    py::buffer_info v_info = v.request();

    // Create output array with the same shape as expected for attention output
    auto q_shape = q.shape();
    auto v_shape = v.shape();
    
    // Assume output has shape [batch_size, seq_len, head_dim]
    std::vector<ssize_t> output_shape = {q_shape[0], q_shape[1], v_shape[2]};
    py::array_t<float> output = py::array_t<float>(output_shape);
    
    // Benchmark
    double total_time = 0.0;
    
    // Placeholder for actual CUDA kernel call
    // In the real implementation, we would call the CUDA kernel here
    for (int i = 0; i < num_runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Placeholder for CUDA flash attention computation
        // flash_attention::compute(...);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        total_time += duration.count();
    }
    
    return {total_time / num_runs, output};
}

// Compare the results of both implementations
py::dict compare_implementations(
    py::array_t<float> q,
    py::array_t<float> k,
    py::array_t<float> v,
    int num_runs = 10
) {
    auto [std_time, std_output] = benchmark_attention(q, k, v, num_runs);
    auto [flash_time, flash_output] = benchmark_flash_attention(q, k, v, num_runs);
    
    // Calculate error between the two implementations
    py::buffer_info std_info = std_output.request();
    py::buffer_info flash_info = flash_output.request();
    
    float* std_ptr = static_cast<float*>(std_info.ptr);
    float* flash_ptr = static_cast<float*>(flash_info.ptr);
    
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    size_t total_elements = std_info.size;
    
    for (size_t i = 0; i < total_elements; i++) {
        double abs_err = std::abs(std_ptr[i] - flash_ptr[i]);
        max_abs_error = std::max(max_abs_error, abs_err);
        mean_abs_error += abs_err;
    }
    
    mean_abs_error /= total_elements;
    
    // Return results as a dictionary
    py::dict results;
    results["standard_time_ms"] = std_time;
    results["flash_time_ms"] = flash_time;
    results["speedup"] = std_time / flash_time;
    results["max_abs_error"] = max_abs_error;
    results["mean_abs_error"] = mean_abs_error;
    
    return results;
}

// Module definition
PYBIND11_MODULE(attention_benchmark, m) {
    m.doc() = "Benchmark module for comparing standard attention and flash attention";
    
    m.def("benchmark_attention", &benchmark_attention, 
          "Benchmark standard attention",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("num_runs") = 10);
          
    m.def("benchmark_flash_attention", &benchmark_flash_attention, 
          "Benchmark flash attention",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("num_runs") = 10);
          
    m.def("compare_implementations", &compare_implementations,
          "Compare standard attention and flash attention",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("num_runs") = 10);
}