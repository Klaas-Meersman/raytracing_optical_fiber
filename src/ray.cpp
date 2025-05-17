#include "ray.hpp"
#include "fiber.hpp"
#include <cmath>
#include <numbers>
#include <string>
#include <algorithm>

Ray::Ray()
    : start(0, 0), end(0, 0), azimuth(0), elevation(0), fiber(Fiber(1, 1,1)), endHitFiber(false) {
}

Ray::Ray(const Ray& other)
    : start(other.start), end(other.end), azimuth(other.azimuth), elevation(other.elevation),
      fiber(other.fiber), endHitFiber(other.endHitFiber) {
}

Ray& Ray::operator=(const Ray& other) {
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


Ray::~Ray(){
}

Ray::Ray(Coordinate start, double_t azimuth, double_t elevation, const Fiber &fiber)
    : start(start), azimuth(azimuth), elevation(elevation), fiber(fiber), endHitFiber(false) {
    
    double vx = std::cos(elevation) * std::cos(azimuth);
    double vy = std::sin(elevation);
    double vz = std::cos(elevation) * std::sin(azimuth);

    double tx = (fiber.getLength() - start.x) / vx;
    double ty = (vy > 0) ? (fiber.getTopY() - start.y) / vy :
               (vy < 0) ? (fiber.getBottomY() - start.y) / vy : std::numeric_limits<double>::infinity();
    double tz = (vz > 0) ? (fiber.getTopZ() - start.z) / vz :
               (vz < 0) ? (fiber.getBottomZ() - start.z) / vz : std::numeric_limits<double>::infinity();


    double t = std::min({tx, ty, tz});

    end.x = start.x + vx * t;
    end.y = start.y + vy * t;
    end.z = start.z + vz * t;

    if (tx <= ty && tx <= tz) {
        endHitFiber = true;
    }
}


void Ray::propagateRay() {
    start = end;

    double vx = std::cos(elevation) * std::cos(azimuth);
    double vy = std::sin(elevation);
    double vz = std::cos(elevation) * std::sin(azimuth);

    double tx = (fiber.getLength() - start.x) / vx;
    double ty = (vy > 0) ? (fiber.getTopY() - start.y) / vy :
               (vy < 0) ? (fiber.getBottomY() - start.y) / vy : std::numeric_limits<double>::infinity();
    double tz = (vz > 0) ? (fiber.getTopZ() - start.z) / vz :
               (vz < 0) ? (fiber.getBottomZ() - start.z) / vz : std::numeric_limits<double>::infinity();

    double t = std::min({tx, ty, tz});

    end.x = start.x + vx * t;
    end.y = start.y + vy * t;
    end.z = start.z + vz * t;

    if (t == ty) {
        elevation = -elevation;  
    }
    if (t == tz) {
        azimuth = 2 * 3.1415 - azimuth; 
    }

    if (t == tx) {
        endHitFiber = true;
    };
}