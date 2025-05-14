#include <cstdio>
#include <iostream>
// include math library for pi
#include <cmath>
#include <numbers>
#include "coordinate.hpp"
#include "ray.hpp"

// //* To trace one ray
// void traceRay(const Ray& initialRay, const Fiber& fiber) {
//     Ray ray = initialRay;
//     while (!ray.getEndHitFiber()) {
//         std::vector<Coordinate> path = ray.generateStraightPath(0.2);
//         for (const auto& coord : path) {
//             std::cout << coord << std::endl;
//         }
//         //ray = ray.generateBounceRay(fiber);
//         ray.propagateRay();
//     }
//     // Process and display the last ray's path
//     std::vector<Coordinate> finalPath = ray.generateStraightPath(0.2);
//     for (const auto& coord : finalPath) {
//         std::cout << coord << std::endl;
//     }
// }

// int main(){
//     double length_fiber = 100;
//     double width_fiber = 5;
//     Fiber fiber = Fiber(width_fiber, length_fiber);
//     printf("fiber_length,%f\n", fiber.getLength());
//     printf("fiber_top_y,%f\nfiber_bottom_y,%f\n", fiber.getTopY(), fiber.getBottomY());

//     printf("x,y\n");

//     Coordinate startCo = Coordinate(0, 0);
//     double_t angleDegrees = 30;
//     double_t angleRadians = angleDegrees / 180 * std::numbers::pi;

//     Ray initialRay = Ray(startCo, angleRadians, fiber);
//     traceRay(initialRay, fiber);
//     return 0;
// }
// //*

// To trace multiple rays only to the end of the fiber
// we generate a 1000 rays and print their coordinates
void traceSingleRay(const Fiber &fiber)
{
    Coordinate startCo = Coordinate(0, 0);

    // Gebruik een vaste hoek voor betere visualisatie
    double_t angleDegrees = 30;
    double_t angleRadians = angleDegrees / 180 * 3.1415;

    Ray ray = Ray(startCo, angleRadians, fiber);

    while (!ray.getEndHitFiber())
    {
        std::cout << ray.getStart().x << "," << ray.getStart().y << std::endl;
        ray = ray.propagateRay();
    }

    // Print eindpunt
    std::cout << ray.getEnd().x << "," << ray.getEnd().y << std::endl;
}

// Debugging output toevoegen aan traceSingleRay
void traceDoubleRay(const Fiber &fiber)
{
    Coordinate startCo(0, 0);

    // Hoek in graden
    double angleDegrees = 30;
    double angleRad1 = angleDegrees / 180.0 * 3.1415;   // positieve hoek
    double angleRad2 = -angleDegrees / 180.0 * 3.1415;  // negatieve hoek

    Ray ray1(startCo, angleRad1, fiber);
    Ray ray2(startCo, angleRad2, fiber);

    // Eerst ray1 volgen
    while (!ray1.getEndHitFiber())
    {
        std::cout << "Ray1 Start: " << ray1.getStart().x << "," << ray1.getStart().y << std::endl;
        ray1 = ray1.propagateRay();
    }
    std::cout << "Ray1 End: " << ray1.getEnd().x << "," << ray1.getEnd().y << std::endl;

    // Dan ray2 volgen
    while (!ray2.getEndHitFiber())
    {
        std::cout << "Ray2 Start: " << ray2.getStart().x << "," << ray2.getStart().y << std::endl;
        ray2 = ray2.propagateRay();
    }
    std::cout << "Ray2 End: " << ray2.getEnd().x << "," << ray2.getEnd().y << std::endl;
}


int main(){
    double length_fiber = 100;
    double width_fiber = 5;
    Fiber fiber(width_fiber, length_fiber);

    // Metadata
    printf("fiber_length,%f\n", fiber.getLength());
    printf("fiber_top_y,%f\nfiber_bottom_y,%f\n", fiber.getTopY(), fiber.getBottomY());

    // CSV-header
    printf("x,y\n");

    // Dubbele straal volgen
    //traceDoubleRay(fiber);
    traceSingleRay(fiber);


    return 0;
}
