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
    Coordinate startCo = Coordinate(0, 0, 0);

    // Gebruik een vaste hoek voor betere visualisatie
    double_t angleDegrees = 30;
    double_t azimuth = angleDegrees / 180.0 * 3.1415;   // Hoek in de XZ-vlak
    double_t elevation = 20.0 / 180.0 * 3.1415;         // Hoek in de YZ-vlak

    Ray ray(startCo, azimuth, elevation, fiber);

    // Gebruik de juiste output voor 3D coördinaten
    while (!ray.getEndHitFiber())
    {
        std::cout << ray.getStart().x << "," << ray.getStart().y << "," << ray.getStart().z << std::endl;
        ray = ray.propagateRay();
    }
    // Print het eindpunt
    std::cout << ray.getEnd().x << "," << ray.getEnd().y << "," << ray.getEnd().z << std::endl;
}


int main(){
    double length_fiber = 100;
    double width_fiber = 5;
    double height_fiber = 5;  // Nieuw toegevoegd voor de Z-dimensie

    Fiber fiber(width_fiber, height_fiber, length_fiber);

    // Metadata
    printf("fiber_length,%f\n", fiber.getLength());
    printf("fiber_top_y,%f\nfiber_bottom_y,%f\n", fiber.getTopY(), fiber.getBottomY());
    printf("fiber_left_z,%f\nfiber_right_z,%f\n", fiber.getTopZ(), fiber.getBottomZ());

    // CSV-header
    printf("x,y,z\n");

    // Traceer de straal in 3D
    traceSingleRay(fiber);

    return 0;
}