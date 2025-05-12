#ifndef FIBER_HPP
#define FIBER_HPP

#include "coordinate.hpp"

class Fiber
{
private:
    double_t width, length; // Remove const for CUDA compatibility

public:
    // Mark all special member functions as __host__ __device__
    __host__ __device__ Fiber(double_t width = 1, double_t length = 1);
    __host__ __device__ Fiber(const Fiber& other);
    __host__ __device__ Fiber& operator=(const Fiber& other);
    __host__ __device__ ~Fiber();

    __host__ __device__ inline double_t getTopY() const { return width / 2; }
    __host__ __device__ inline double_t getBottomY() const { return -width / 2; }
    __host__ __device__ inline double_t getWidth() const { return width; }
    __host__ __device__ inline double_t getLength() const { return length; }
};

#endif // FIBER_HPP
