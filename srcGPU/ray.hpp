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
    __host__ __device__ Ray()
        : start(0, 0, 0), end(0, 0, 0), azimuth(0), elevation(0), fiber(nullptr), endHitFiber(false) {}

    // Constructor with Fiber pointer
    __host__ __device__ Ray(Coordinate start, double_t azimuth, double_t elevation, const Fiber* fiber)
        : start(start), azimuth(azimuth), elevation(elevation), fiber(fiber), endHitFiber(false) {

        // Compute direction vector
        double vx = std::cos(elevation) * std::cos(azimuth);
        double vy = std::sin(elevation);
        double vz = std::cos(elevation) * std::sin(azimuth);

        // "Distance" to each boundary of the fiber
        // Distance is the length to the fiber boundary along the ray direction
        double dx = (fiber->getLength() - start.x) / vx;
        double dy = (vy > 0) ? (fiber->getTopY() - start.y) / vy :
                    (vy < 0) ? (fiber->getBottomY() - start.y) / vy : fiber->getLength() * 2; // just a bigger number
        double dz = (vz > 0) ? (fiber->getTopZ() - start.z) / vz :
                    (vz < 0) ? (fiber->getBottomZ() - start.z) / vz : fiber->getLength() * 2; // just a bigger number

        // Take the smallest positive distance for the collision
        double d = fmin(fmin(dx, dy), dz);

        // Calculate endpoint
        end.x = start.x + vx * d;
        end.y = start.y + vy * d;
        end.z = start.z + vz * d;

        // If we hit the end of the fiber
        if (dx <= dy && dx <= dz) {
            endHitFiber = true;
        }
    }

    __host__ __device__ Ray(const Ray& other): 
            start(other.start), end(other.end), azimuth(other.azimuth), elevation(other.elevation),
            fiber(other.fiber), endHitFiber(other.endHitFiber) {}


    __host__ __device__ Ray& operator=(const Ray& other) {
        if (this != &other) {
            start = other.start;
            end = other.end;
            azimuth = other.azimuth;
            elevation = other.elevation;
            fiber = other.fiber; 
            endHitFiber = other.endHitFiber;
        }
        return *this;
    }

    __host__ __device__ ~Ray() {}


private:
    Coordinate start;
    Coordinate end;
    const Fiber* fiber;
    double_t azimuth;
    double_t elevation;
    bool endHitFiber = false;

public:
    __host__ __device__ inline Coordinate getStart() const { return start; }
    __host__ __device__ inline Coordinate getEnd() const { return end; }
    __host__ __device__ inline double_t getAzimuth() const { return azimuth; }
    __host__ __device__ inline double_t getElevation() const { return elevation; }
    __host__ __device__ inline bool getEndHitFiber() const { return endHitFiber; }
    __host__ __device__ inline void setEndHitFiber(bool endHitFiber) { this->endHitFiber = endHitFiber; }

    __host__ __device__ inline void propagateRay() {
        // Update start point to current end point
        start = end;

        // Recalculate direction vector
        double vx = std::cos(elevation) * std::cos(azimuth);
        double vy = std::sin(elevation);
        double vz = std::cos(elevation) * std::sin(azimuth);

        // "Distance" to boundaries: distance to each boundary along the direction vector
        double dx = (fiber->getLength() - start.x) / vx;
        double dy = (vy > 0) ? (fiber->getTopY() - start.y) / vy :
                    (vy < 0) ? (fiber->getBottomY() - start.y) / vy : fiber->getLength() * 2; // just a bigger number
        double dz = (vz > 0) ? (fiber->getTopZ() - start.z) / vz :
                    (vz < 0) ? (fiber->getBottomZ() - start.z) / vz : fiber->getLength() * 2; // just a bigger number

        double d = fmin(fmin(dx, dy), dz);

        // Endpoint at next collision
        end.x = start.x + vx * d;
        end.y = start.y + vy * d;
        end.z = start.z + vz * d;

        // Reflection: update angles if needed
        if (d == dy) {
            elevation = -elevation;
        }
        if (d == dz) {
            azimuth = 2 * M_PI - azimuth;
        }

        // Check if we reached the fiber end
        if (d == dx) {
            endHitFiber = true;
        }
    }
};
