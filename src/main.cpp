#include <cstdio>
#include <iostream>
// include math library for pi
#include <cmath>
#include <numbers>

#include "coordinate.hpp"
#include "ray.hpp"

void traceRay(const Ray& initialRay, const Fiber& fiber) {
    Ray ray = initialRay;
    while (!ray.getEndHitFiber()) {
        std::vector<Coordinate> path = ray.generateStraightPath(0.2);
        for (const auto& coord : path) {
            std::cout << coord << std::endl;
        }
        ray = ray.generateBounceRay(fiber);
    }

    // Process and display the last ray's path
    std::vector<Coordinate> finalPath = ray.generateStraightPath(0.2);
    for (const auto& coord : finalPath) {
        std::cout << coord << std::endl;
    }
}

int main(){
    double length_fiber = 1002;
    double width_fiber = 5;
    Fiber fiber = Fiber(width_fiber, length_fiber);
    printf("fiber_length,%f\n", fiber.getLength());
    printf("fiber_top_y,%f\nfiber_bottom_y,%f\n", fiber.getTopY(), fiber.getBottomY());

    printf("x,y\n");
    
    Coordinate startCo = Coordinate(0, 0);
    double_t angleDegrees = 30;
    double_t angleRadians = angleDegrees / 180 * std::numbers::pi;

    Ray initialRay = Ray(startCo, angleRadians, fiber);
    traceRay(initialRay, fiber);
    return 0;
}
