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

# Find Python and pybind11 for Python bindings
find_package(Python COMPONENTS Interpreter Development REQUIRED)
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v3.0.1
)
FetchContent_MakeAvailable(pybind11)

# Create Python module
pybind11_add_module(attention_benchmark
    python/attention_benchmark.cpp
)
target_link_libraries(attention_benchmark PRIVATE
    attention
    flash_attention
    CUDA::cudart
)

# Installation rules
install(TARGETS attention flash_attention attention_benchmark
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
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
message(STATUS "  Python version:    ${Python_VERSION}")
message(STATUS "")
