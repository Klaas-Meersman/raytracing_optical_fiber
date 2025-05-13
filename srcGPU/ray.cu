#include "ray.hpp"
#include "fiber.hpp"
#include <cmath>
#include <numbers>

// Default constructor
__host__ __device__
Ray::Ray()
    : start(0, 0), end(0, 0), angleOfDeparture(0), fiber(nullptr), direction(Direction::UP), endHitFiber(false) {
}

// Constructor with Fiber pointer
__host__ __device__
Ray::Ray(Coordinate start, double_t angleOfDeparture, const Fiber* fiber)
    : start(start), angleOfDeparture(angleOfDeparture), fiber(fiber), direction(Direction::UP), endHitFiber(false) {
    //if (!fiber) return;
    if (angleOfDeparture > 0 && angleOfDeparture < M_PI / 2) {
        direction = Direction::UP;
        this->end.y = fiber->getTopY();
        this->end.x = this->start.x + (fiber->getTopY() - this->start.y) / std::tan(this->angleOfDeparture);
    } else if (angleOfDeparture > 3 * M_PI / 4 && angleOfDeparture < 2 * M_PI) {
        direction = Direction::DOWN;
        this->end.y = fiber->getBottomY();
        this->end.x = this->start.x + (fiber->getBottomY() - this->start.y) / std::tan(this->angleOfDeparture); 
    } else {
        // On device, don't throw: just mark as invalid

    }
    if(this->end.x > fiber->getLength()){
        this->endHitFiber = true;
        this->end.x = fiber->getLength();
        this->end.y = std::tan(this->angleOfDeparture) * (fiber->getLength() - this->start.x) + this->start.y;
    }
}





// CUDA-compatible propagateRay (in-place, returns void)


#ifndef __CUDA_ARCH__
// Host-only methods (using STL or exceptions) can go here
#endif
