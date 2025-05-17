#pragma once

#include "coordinate.hpp"

class Fiber
{
private:
    double_t width, height, length;

public:
    __host__ __device__ Fiber(double_t width = 1, double_t height = 1, double_t length = 1)
        : width(width), height(height), length(length) {}

    __host__ __device__ Fiber(const Fiber& other)
        : width(other.width), height(other.height), length(other.length) {}

    __host__ __device__ Fiber& operator=(const Fiber& other) {
        if (this != &other) {
            width = other.width;
            height = other.height;
            length = other.length;
        }
        return *this;
    }

    __host__ __device__ ~Fiber() {}

    __host__ __device__ inline double_t getTopY() const { return width / 2; }
    __host__ __device__ inline double_t getBottomY() const { return -width / 2; }
    __host__ __device__ inline double_t getTopZ() const { return height / 2; }
    __host__ __device__ inline double_t getBottomZ() const { return -height / 2; }
    __host__ __device__ inline double_t getWidth() const { return width; }
    __host__ __device__ inline double_t getHeight() const { return height; }
    __host__ __device__ inline double_t getLength() const { return length; }
};