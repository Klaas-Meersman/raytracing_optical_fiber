#pragma once

#include "coordinate.hpp"
#include "fiber.hpp"

class Ray
{
public:
    // Mark constructors and destructor as __host__ __device__
    __host__ __device__ Ray();
    __host__ __device__ Ray(Coordinate start, double_t angleOfDeparture, const Fiber* fiber);
    __host__ __device__ Ray(const Ray& other);
    __host__ __device__ Ray& operator=(const Ray& other);
    __host__ __device__ ~Ray();

    enum class Direction {
        UP,
        DOWN
    };

private:
    Coordinate start;
    Coordinate end;
    // For device code, avoid references. Use a pointer or value.
    // const Fiber& fiber;
    const Fiber* fiber;
    double_t angleOfDeparture;
    Direction direction;
    bool endHitFiber = false;

public:
    // Remove std::vector from device code, or provide host-only alternatives.
    // __host__ std::vector<Coordinate> generateStraightPath(double dx);
    // __host__ Ray generateBounceRay(const Fiber& fiber);

    //__host__ __device__ void propagateRay();

    __host__ __device__ inline Coordinate getStart() const { return start; }
    __host__ __device__ inline Coordinate getEnd() const { return end; }
    __host__ __device__ inline double_t getAngleOfDeparture() const { return angleOfDeparture; }
    __host__ __device__ inline Direction getDirection() const { return direction; }
    __host__ __device__ inline bool getEndHitFiber() const { return endHitFiber; }
    __host__ __device__ inline void setEndHitFiber(bool endHitFiber) { this->endHitFiber = endHitFiber; }
    // CUDA-compatible propagateRay (in-place, returns void)
    __host__ __device__ inline void propagateRay() {
        if (!fiber) return;
        this->start.x = this->end.x;
        this->start.y = this->end.y;
        this->angleOfDeparture = std::numbers::pi*2 - this->angleOfDeparture;
        if (angleOfDeparture > 0 && angleOfDeparture < std::numbers::pi / 2) {
            this->direction = Direction::UP;
            this->end.y = fiber->getTopY();
            this->end.x = this->start.x + (fiber->getTopY() - this->start.y) / std::tan(this->angleOfDeparture);
        } else if (angleOfDeparture > 3 * std::numbers::pi / 4 && angleOfDeparture < 2 * std::numbers::pi) {
            this->direction = Direction::DOWN;
            this->end.y = fiber->getBottomY();
            this->end.x = this->start.x + (fiber->getBottomY() - this->start.y) / std::tan(this->angleOfDeparture); 
        } else {
            // On device, don't throw: just mark as invalid
            this->end.x = this->start.x;
            this->end.y = this->start.y;
            this->endHitFiber = true;
            return;
        }
        if(this->end.x > fiber->getLength()){
            this->endHitFiber = true;
            this->end.x = fiber->getLength();
            this->end.y = std::tan(this->angleOfDeparture) * (fiber->getLength() - this->start.x) + this->start.y;
        }
    }

    // Remove stream operator for device code
    // friend std::ostream& operator<<(std::ostream& os, const Ray& r);
};
