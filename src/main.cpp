#include <cstdio>
#include <iostream>
// include math library for pi
#include <cmath>
#include <numbers>
#include "coordinate.hpp"
#include "ray.hpp"

/* //To trace one ray
 void traceRay(const Ray& initialRay, const Fiber& fiber) {
    Ray ray = initialRay;
    while (!ray.getEndHitFiber()) {
        std::vector<Coordinate> path = ray.generateStraightPath(0.2);
        for (const auto& coord : path) {
            std::cout << coord << std::endl;
        }
        //ray = ray.generateBounceRay(fiber);
        ray.propagateRay();
    }
    // Process and display the last ray's path
    std::vector<Coordinate> finalPath = ray.generateStraightPath(0.2);
    for (const auto& coord : finalPath) {
        std::cout << coord << std::endl;
    }
 }

int main(){
    double length_fiber = 100;
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
} */





// To trace multiple rays only to the end of the fiber
// we generate a 1000 rays and print their coordinates
void traceRays(const Fiber &fiber, int numRays)
{
    std::vector<Ray> rays;
    double_t angleRadians = 0;

    //lambertian distribution
    for (int i = 0; i < numRays; ++i)
    {
        double u = static_cast<double>(rand()) / RAND_MAX; // Uniform in [0,1]
        double theta = std::asin(u); // θ in [0, π/2] radians
    
        // Randomly choose between [0, 90] and [270, 360]
        if (rand() % 2 == 0)
        {
            // [0, 90] degrees
            angleRadians = theta;
        }
        else
        {
            // [270, 360] degrees
            angleRadians = 3 * std::numbers::pi / 2 + theta;
        }
    
        Coordinate startCo = Coordinate(0, 0);
        Ray ray = Ray(startCo, angleRadians, fiber);
        rays.push_back(ray);
    }

    // for all rays, propagate until they hit the fiber, then print their end coordinates
    for (const auto &ray : rays)
    {
        Ray currentRay = ray;
        while (!currentRay.getEndHitFiber())
        {
            currentRay.propagateRay();
        }
        if(currentRay.getEndHitFiber()){
            std::cout << currentRay.getEnd().x << ", " << currentRay.getEnd().y << std::endl;
        }
        //std::cout << currentRay.getStart().x << ", " << currentRay.getStart().y << ")" << std::endl;
    }
}

int main()
{
    double length_fiber = 100;
    double width_fiber = 5;
    Fiber fiber = Fiber(width_fiber, length_fiber);
    printf("fiber_length,%f\n", fiber.getLength());
    printf("fiber_top_y,%f\nfiber_bottom_y,%f\n", fiber.getTopY(), fiber.getBottomY());

    printf("x,y\n");

    int numRays = 1000000;
    traceRays(fiber, numRays);

    return 0;
}