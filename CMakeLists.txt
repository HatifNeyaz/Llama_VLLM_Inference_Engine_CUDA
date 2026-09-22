cmake_minimum_required(VERSION 3.18)

# Name our project and tell it we are using both C++ and CUDA
project(TinyVLLM_1660Ti LANGUAGES CXX CUDA)

# The author used C++17 features, so we force the compiler to use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ⚠️ YOUR 1660 Ti HARDWARE FIX:
# The GTX 1660 Ti is Turing architecture (sm_75). 
# This line prevents the compiler from targeting the author's RTX 5090!
set(CMAKE_CUDA_ARCHITECTURES "75")

# Find NVIDIA's cuBLAS matrix math library
find_package(CUDAToolkit REQUIRED)

# Tell CMake we are going to write our code in a file named main.cu
add_executable(tiny_vllm main.cu)

# Tell the compiler where to find json.hpp (checks both the main folder and the include folder)
target_include_directories(tiny_vllm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

# Link the cuBLAS math library to our program
target_link_libraries(tiny_vllm PRIVATE CUDA::cublas)