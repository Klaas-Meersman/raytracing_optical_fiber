#include "fiber.hpp"
#include "ray.hpp"



Fiber::Fiber(double_t width, double_t length)
    :width(width),length(length){
    if (length <= 0 || width <= 0) {
        throw std::invalid_argument("Width/length must be positive");
    }
}

Fiber::~Fiber(){
}

int Fiber::maxSingleRayInFiber(double_t startX){
    return length - startX  + width;
}