cmake_minimum_required(VERSION 3.30)
project(raytracing_optical_fiber)

set(CMAKE_CXX_STANDARD 20)

include_directories(.)

add_executable(raytracing_optical_fiber
        src/coordinate.cpp
        src/coordinate.hpp
        src/main.cpp
        src/ray.cpp
        src/ray.hpp)