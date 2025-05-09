#include "fiber.hpp"
#include "ray.hpp"



Fiber::Fiber(double_t width, double_t length)
    :width(width),length(length){
    if (length <= 0 || width <= 0) {
        throw std::invalid_argument("Width/length must be positive. It is not: " + std::to_string(width) + " " + std::to_string(length));
    }
}

Fiber::Fiber(const Fiber& other)
    : width(other.width), length(other.length) {
}

Fiber& Fiber::operator=(const Fiber& other) {
    if (this != &other) {
        // Since width and length are const, they cannot be reassigned.
        // However, this operator is provided for completeness.
    }
    return *this;
}

Fiber::~Fiber(){
}
