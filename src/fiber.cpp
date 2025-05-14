// fiber.cpp
#include "fiber.hpp"
#include <stdexcept>
#include <string>
#include "ray.hpp"

Fiber::Fiber(double_t width, double_t height, double_t length)
    : width(width), height(height), length(length) {
    if (length <= 0 || width <= 0 || height <= 0) {
        throw std::invalid_argument("All dimensions must be positive.");
    }
}

Fiber::Fiber(const Fiber& other)
    : width(other.width), height(other.height), length(other.length) {}

Fiber& Fiber::operator=(const Fiber& other) {
    return *this; // Const members can't be reassigned
}

Fiber::~Fiber() = default;
