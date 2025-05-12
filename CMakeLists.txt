cmake_minimum_required(VERSION 3.20)
project(raytracing_optical_fiber LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add include directory for srcGPU
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/srcGPU)

# Collect all .cu files in srcGPU
file(GLOB CU_SOURCES "srcGPU/*.cu")

# Create the executable from all CUDA source files
add_executable(raytracing_optical_fiber
    ${CU_SOURCES}
)

# (Optional) Suppress deprecated GPU target warning
target_compile_options(raytracing_optical_fiber PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
