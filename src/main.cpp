#include <cstdio>
#include <iostream>
#include <cmath>
#include <numbers>
#include "coordinate.hpp"
#include "ray.hpp"
#include <random>
#include <chrono>

// Trace a single ray, outputting only the endpoint (no ID)
void traceSingleRayNoID(const Fiber &fiber, double azimuth, double elevation) {
    Coordinate startCo(0, 0, 0);
    Ray ray(startCo, azimuth, elevation, fiber);

    while (!ray.getEndHitFiber()) {
        ray = ray.propagateRay();
    }
    // Output the end point: x,y,z (no id)
    std::cout << ray.getEnd().x << "," << ray.getEnd().y << "," << ray.getEnd().z << std::endl;
}



// Lambertian distribution: only rays from 270° to 90° azimuth, max at 0°
void traceLed(const Fiber &fiber, int numRays, double maxAzimuthDeg, double maxElevationDeg) {
    if (numRays < 20) numRays = 20;

    // Limit azimuth to [-90°, +90°], i.e., [-π/2, +π/2] radians
    double minAzimuthRad = -M_PI / 2;
    double maxAzimuthRad = M_PI / 2;
    double maxElevationRad = maxElevationDeg * M_PI / 180.0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> uniform(0.0, 1.0);

    int rayCount = 0;
    while (rayCount < numRays) {
        // Sample phi uniformly in [-π/2, +π/2]
        double phi = minAzimuthRad + uniform(gen) * (maxAzimuthRad - minAzimuthRad);

        // Cosine-weighted elevation (Lambertian)
        double cosThetaMin = std::cos(maxElevationRad);
        double cosTheta = uniform(gen) * (1.0 - cosThetaMin) + cosThetaMin;
        double theta = std::acos(cosTheta);

        traceSingleRayNoID(fiber, phi, theta);
        ++rayCount;
    }
}


int main(){
    double length_fiber = 100;
    double width_fiber = 5;
    double height_fiber = 5;

    Fiber fiber(width_fiber, height_fiber, length_fiber);

    printf("fiber_length,%f\n", fiber.getLength());
    printf("fiber_top_y,%f\nfiber_bottom_y,%f\n", fiber.getTopY(), fiber.getBottomY());
    printf("fiber_top_z,%f\nfiber_bottom_z,%f\n", fiber.getTopZ(), fiber.getBottomZ());
    printf("x,y,z\n");

    int aantalRays = 1000000;
    double maxAzimuth = 70;   // in degrees
    double maxElevation = 70; // in degrees
    //traceMultipleRaysRandom(fiber, aantalRays, maxAzimuth, maxElevation);
    //traceMultipleRaysSegmented(fiber, aantalRays, maxAzimuth, maxElevation);

    auto start = std::chrono::steady_clock::now();
    traceLed(fiber, aantalRays, maxAzimuth, maxElevation);
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << elapsed << " ms\n";

    return 0;
}
