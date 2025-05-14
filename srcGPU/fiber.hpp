#pragma once

#include "coordinate.hpp"

class Fiber
{
private:
    double_t width, length; // Remove const for CUDA compatibility

public:
    // Inline constructor
    __host__ __device__ Fiber(double_t width = 1, double_t length = 1)
        : width(width), length(length) {}

    // Inline copy constructor
    __host__ __device__ Fiber(const Fiber& other)
        : width(other.width), length(other.length) {}

    // Inline assignment operator
    __host__ __device__ Fiber& operator=(const Fiber& other) {
        if (this != &other) {
            width = other.width;
            length = other.length;
        }
        return *this;
    }

    // Inline destructor
    __host__ __device__ ~Fiber() {}

    __host__ __device__ inline double_t getTopY() const { return width / 2; }
    __host__ __device__ inline double_t getBottomY() const { return -width / 2; }
    __host__ __device__ inline double_t getWidth() const { return width; }
    __host__ __device__ inline double_t getLength() const { return length; }
};