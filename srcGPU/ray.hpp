#pragma once

#include "coordinate.hpp"
#include "fiber.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class Ray
{
public:
    // Mark constructors and destructor as __host__ __device__
    __host__ __device__ Ray();
    // Constructor with Fiber pointer
    __host__ __device__ Ray(Coordinate start, double_t angleOfDeparture, const Fiber* fiber)
        : start(start), angleOfDeparture(angleOfDeparture), fiber(fiber), endHitFiber(false) {
        if (angleOfDeparture > 0 && angleOfDeparture < M_PI / 2) {
            this->end.y = fiber->getTopY();
            this->end.x = this->start.x + (fiber->getTopY() - this->start.y) / std::tan(this->angleOfDeparture);
        } else if (angleOfDeparture > 3 * M_PI / 4 && angleOfDeparture < 2 * M_PI) {
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
    __host__ __device__ Ray(const Ray& other): 
            start(other.start), end(other.end), fiber(other.fiber),
            angleOfDeparture(other.angleOfDeparture),
            endHitFiber(other.endHitFiber) {}
    __host__ __device__ Ray& operator=(const Ray& other) {
        if (this != &other) {
            start = other.start;
            end = other.end;
            fiber = other.fiber;
            angleOfDeparture = other.angleOfDeparture;
            endHitFiber = other.endHitFiber;
        }
        return *this;
    }
    __host__ __device__ ~Ray() {}


private:
    Coordinate start;
    Coordinate end;
    const Fiber* fiber;
    double_t angleOfDeparture;
    bool endHitFiber = false;

public:
    __host__ __device__ inline Coordinate getStart() const { return start; }
    __host__ __device__ inline Coordinate getEnd() const { return end; }
    __host__ __device__ inline double_t getAngleOfDeparture() const { return angleOfDeparture; }
    __host__ __device__ inline bool getEndHitFiber() const { return endHitFiber; }
    __host__ __device__ inline void setEndHitFiber(bool endHitFiber) { this->endHitFiber = endHitFiber; }
    // CUDA-compatible propagateRay (in-place, returns void)
    __host__ __device__ inline Ray propagateRay() {
        this->start.x = this->end.x;
        this->start.y = this->end.y;
        this->angleOfDeparture = M_PI*2 - this->angleOfDeparture;

        if (angleOfDeparture > 0 && angleOfDeparture < M_PI / 2) {
            this->end.y = fiber->getTopY();
            this->end.x = this->start.x + (fiber->getTopY() - this->start.y) / std::tan(this->angleOfDeparture);
        } else if (angleOfDeparture > 3 * M_PI / 4 && angleOfDeparture < 2 * M_PI) {
            this->end.y = fiber->getBottomY();
            this->end.x = this->start.x + (fiber->getBottomY() - this->start.y) / std::tan(this->angleOfDeparture); 
        } else {
            printf("DO I PASS HERE? I shouldn't\n");
        }

        if(this->end.x > fiber->getLength()){
            this->endHitFiber = true;
            this->end.x = fiber->getLength();
            this->end.y = std::tan(this->angleOfDeparture) * (fiber->getLength() - this->start.x) + this->start.y;
        }
        return *this;
    }
};
