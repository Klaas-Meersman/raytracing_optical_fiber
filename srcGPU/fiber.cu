#include "fiber.hpp"

#ifndef __CUDA_ARCH__
#include <stdexcept>
#include <string>
#endif

__host__ __device__
Fiber::Fiber(double_t width, double_t length)
    : width(width), length(length) {
#ifndef __CUDA_ARCH__
    if (length <= 0 || width <= 0) {
        throw std::invalid_argument("Width/length must be positive. It is not: " +
                                    std::to_string(width) + " " + std::to_string(length));
    }
#endif
}

__host__ __device__
Fiber::Fiber(const Fiber& other)
    : width(other.width), length(other.length) {}

__host__ __device__
Fiber& Fiber::operator=(const Fiber& other) {
    if (this != &other) {
        width = other.width;
        length = other.length;
    }
    return *this;
}

__host__ __device__
Fiber::~Fiber() {}
