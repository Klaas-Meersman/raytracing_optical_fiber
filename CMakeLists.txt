cmake_minimum_required(VERSION 3.30)
project(raytracing_optical_fiber)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


include_directories(.)


file(GLOB SOURCES "src/*.cpp")
add_executable(raytracing_optical_fiber
        src/coordinate.cpp
        src/coordinate.hpp
        src/main.cpp
        src/ray.cpp
        src/ray.hpp)