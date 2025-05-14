#include "ray.hpp"
#include "fiber.hpp"
#include <cmath>
#include <numbers>

// Default constructor
__host__ __device__
Ray::Ray()
    : start(0, 0), end(0, 0), angleOfDeparture(0), fiber(nullptr), endHitFiber(false) {
}







// CUDA-compatible propagateRay (in-place, returns void)


#ifndef __CUDA_ARCH__
// Host-only methods (using STL or exceptions) can go here
#endif
