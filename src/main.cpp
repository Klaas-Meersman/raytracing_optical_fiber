#include <cstdio>
#include <iostream>
#include <cmath>
#include <numbers>
#include "coordinate.hpp"
#include "ray.hpp"
#include <random>
#include <chrono>

// Trace a single ray, returning only the endpoint
Coordinate traceSingleRay(const Fiber &fiber, double azimuth, double elevation) {
    Coordinate startCo(0, 0, 0);
    Ray ray(startCo, azimuth, elevation, fiber);

    while (!ray.getEndHitFiber()) {
        ray.propagateRay();
    }
    return ray.getEnd();
}

// Store endpoints in a pre-allocated array
void traceLed(const Fiber &fiber, int numRays, double maxAngleDeg, Coordinate* endpoints) {
    double minAzimuthDeg1 = 360.0 - maxAngleDeg;
    double maxAzimuthDeg1 = 360.0;
    double minAzimuthDeg2 = 0.0;
    double maxAzimuthDeg2 = maxAngleDeg;
    double maxElevationDeg = maxAngleDeg;
    double maxElevationRad = maxElevationDeg * M_PI / 180.0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> uniform(0.0, 1.0);

    for (int rayCount = 0; rayCount < numRays; ++rayCount) {
        double u = uniform(gen);
        double phiDeg;
        if (u < 0.5) {
            phiDeg = minAzimuthDeg1 + (maxAzimuthDeg1 - minAzimuthDeg1) * (u / 0.5);
        } else {
            phiDeg = minAzimuthDeg2 + (maxAzimuthDeg2 - minAzimuthDeg2) * ((u - 0.5) / 0.5);
        }
        double phi = phiDeg * M_PI / 180.0;

        double cosThetaMin = std::cos(maxElevationRad);
        double cosTheta = uniform(gen) * (1.0 - cosThetaMin) + cosThetaMin;
        double theta = std::acos(cosTheta);

        if (uniform(gen) < 0.5) {
            theta = -theta;
        }

        endpoints[rayCount] = traceSingleRay(fiber, phi, theta);
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

    int numberOfRays = 1000000;
    double maxAngle = 85; // Max angle in degrees

    // Pre-allocate array
    Coordinate* endpoints = new Coordinate[numberOfRays];

    auto start = std::chrono::steady_clock::now();

    traceLed(fiber, numberOfRays, maxAngle, endpoints);

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // Print all endpoints after timing
    for (int i = 0; i < numberOfRays; ++i) {
        printf("%f,%f,%f\n", endpoints[i].x, endpoints[i].y, endpoints[i].z);
    }

    std::cout << elapsed << " ms\n";



    delete[] endpoints;
    return 0;
}