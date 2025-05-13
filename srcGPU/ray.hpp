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
    __host__ __device__ Ray(Coordinate start, double_t angleOfDeparture, const Fiber* fiber);
    __host__ __device__ Ray(const Ray& other): 
            start(other.start), end(other.end), fiber(other.fiber),
            angleOfDeparture(other.angleOfDeparture), direction(other.direction),
            endHitFiber(other.endHitFiber) {}
    __host__ __device__ Ray& operator=(const Ray& other) {
        if (this != &other) {
            start = other.start;
            end = other.end;
            fiber = other.fiber;
            angleOfDeparture = other.angleOfDeparture;
            direction = other.direction;
            endHitFiber = other.endHitFiber;
        }
        return *this;
    }
    __host__ __device__ ~Ray() {}

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
    __host__ __device__ inline Ray propagateRay() {
        printf("Propagating ray...\n");
        printf("Start: (%f, %f)\n", this->start.x, this->start.y);
        printf("End: (%f, %f)\n", this->end.x, this->end.y);
        printf("Angle of departure: %f\n", this->angleOfDeparture);
        printf("Do I still get here?\n");

        this->start.x = this->end.x;
        this->start.y = this->end.y;
        this->angleOfDeparture = M_PI*2 - this->angleOfDeparture;

        printf("Start: (%f, %f)\n", this->start.x, this->start.y);
        
        printf("Angle of departure: %f\n", this->angleOfDeparture);

        if (angleOfDeparture > 0 && angleOfDeparture < M_PI / 2) {
            printf("Smaller than 90 degrees\n");
            this->direction = Direction::UP;
            this->end.y = fiber->getTopY();
            this->end.x = this->start.x + (fiber->getTopY() - this->start.y) / std::tan(this->angleOfDeparture);
        } else if (angleOfDeparture > 3 * M_PI / 4 && angleOfDeparture < 2 * M_PI) {
            printf("Greater than 270 degrees\n");
            this->direction = Direction::DOWN;
            this->end.y = fiber->getBottomY();
            this->end.x = this->start.x + (fiber->getBottomY() - this->start.y) / std::tan(this->angleOfDeparture); 
        } else {
            printf("DO I PASS HERE?\n");
        }
        if(this->end.x > fiber->getLength()){
            this->endHitFiber = true;
            this->end.x = fiber->getLength();
            this->end.y = std::tan(this->angleOfDeparture) * (fiber->getLength() - this->start.x) + this->start.y;
        }

        printf("End: (%f, %f)\n", this->end.x, this->end.y);
        return *this;
    }

    // Remove stream operator for device code
    // friend std::ostream& operator<<(std::ostream& os, const Ray& r);
};
