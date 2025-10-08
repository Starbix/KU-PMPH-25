cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(attention_benchmarks LANGUAGES CXX CUDA VERSION 1.0.0)

# needed for Zed
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA flags and architectures
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Create library for standard attention
add_library(attention STATIC
    src/attention/attention.cu
)
set_target_properties(attention PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(attention PRIVATE CUDA::cudart)

# Create library for flash attention
add_library(flash_attention STATIC
    src/flash_attention/flash_attention.cu
)
set_target_properties(flash_attention PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(flash_attention PRIVATE CUDA::cudart)

# Find Python Interpreter for running scripts
find_package(Python COMPONENTS Interpreter REQUIRED)

# Create benchmark executable
add_executable(benchmark
    src/benchmark_main.cpp
)
target_link_libraries(benchmark PRIVATE
    attention
    flash_attention
    CUDA::cudart
)

# No Python module needed - we now use the C++ executable directly

# Installation rules
install(TARGETS attention flash_attention benchmark
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h*"
)

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  C++ compiler:      ${CMAKE_CXX_COMPILER}")
message(STATUS "  CUDA compiler:     ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Python executable: ${Python_EXECUTABLE}")
message(STATUS "")
