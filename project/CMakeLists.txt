cmake_minimum_required(VERSION 4.0 FATAL_ERROR)
project(attention_benchmarks LANGUAGES CXX CUDA VERSION 1.0.0)

# Export compile commands for editor integrations
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------
# Compiler standards & CUDA settings
# ---------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default to Ampere (compute capability 8.0) for NVIDIA Ampere GPUs
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "CUDA architectures to build for" FORCE)
endif()

# Helpful default CUDA flags
if (NOT DEFINED CMAKE_CUDA_FLAGS)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()

# ---------------------
# Find required dependencies
# ---------------------
find_package(CUDAToolkit REQUIRED)

# Find Python first to get torch path
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Get torch cmake path from Python
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path, end='')"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    RESULT_VARIABLE TORCH_CMAKE_RESULT
)

if(NOT TORCH_CMAKE_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not find torch cmake path. Make sure PyTorch is installed: pip install torch")
endif()

# Add torch cmake path to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PATH})

find_package(Torch REQUIRED)

# Get torch version from Python since TORCH_VERSION might be empty
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.__version__, end='')"
    OUTPUT_VARIABLE TORCH_VERSION_FROM_PYTHON
    RESULT_VARIABLE TORCH_VERSION_RESULT
)

if(TORCH_VERSION_RESULT EQUAL 0)
    set(TORCH_VERSION_DISPLAY ${TORCH_VERSION_FROM_PYTHON})
else()
    set(TORCH_VERSION_DISPLAY ${TORCH_VERSION})
endif()

message(STATUS "Found Torch: ${TORCH_VERSION_DISPLAY}")
message(STATUS "Found Python: ${Python_VERSION}")

# ---------------------
# Include directories
# ---------------------
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

# ---------------------
# Core attention library
# ---------------------
set(ATTENTION_SOURCES
    src/attention/attention.cpp
    src/attention/attention.cu
)

add_library(attention_core STATIC ${ATTENTION_SOURCES})

set_target_properties(attention_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(attention_core
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${TORCH_INCLUDE_DIRS}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/attention
)

target_link_libraries(attention_core PUBLIC ${TORCH_LIBRARIES} CUDA::cudart)

# ---------------------
# Core flash attention library
# ---------------------
set(FLASH_SOURCES
    src/flash_attention/flash_attention.cpp
    src/flash_attention/flash_attention_kernel.cu
)

add_library(flash_attention_core STATIC ${FLASH_SOURCES})

set_target_properties(flash_attention_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(flash_attention_core
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${TORCH_INCLUDE_DIRS}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/flash_attention
)

target_link_libraries(flash_attention_core PUBLIC ${TORCH_LIBRARIES} CUDA::cudart)

# ---------------------
# Benchmark executable
# ---------------------
add_executable(benchmark src/benchmark_main.cpp)

target_include_directories(benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(benchmark PRIVATE
    attention_core
    flash_attention_core
    ${TORCH_LIBRARIES}
    CUDA::cudart
)

# Propagate TORCH_CXX_FLAGS if present
if(DEFINED TORCH_CXX_FLAGS)
    target_compile_options(benchmark PRIVATE ${TORCH_CXX_FLAGS})
    target_compile_options(attention_core PRIVATE ${TORCH_CXX_FLAGS})
    target_compile_options(flash_attention_core PRIVATE ${TORCH_CXX_FLAGS})
endif()

# ---------------------
# Configuration summary
# ---------------------
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  C++ compiler:       ${CMAKE_CXX_COMPILER}")
message(STATUS "  CUDA compiler:      ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Torch version:      ${TORCH_VERSION_DISPLAY}")
message(STATUS "  Python version:     ${Python_VERSION}")
message(STATUS "")
