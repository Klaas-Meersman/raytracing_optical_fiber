#include <cstdio>
#include <iostream>
// include math library for pi
#include <cmath>
#include <numbers>
#include "coordinate.hpp"
#include "ray.hpp"
#include <random>
#include <chrono>

void traceSingleRayWithID(const Fiber &fiber, int rayID, double azimuth, double elevation) {
    Coordinate startCo(0, 0, 0);
    Ray ray(startCo, azimuth, elevation, fiber);

    while (!ray.getEndHitFiber()) {
        std::cout << rayID << "," << ray.getStart().x << "," << ray.getStart().y << "," << ray.getStart().z << std::endl;
        ray = ray.propagateRay();
    }
    // Output the end point
    std::cout << rayID << "," << ray.getEnd().x << "," << ray.getEnd().y << "," << ray.getEnd().z << std::endl;
}


void traceMultipleRaysRandom(const Fiber &fiber, int numRays, double maxAzimuthDeg, double maxElevationDeg) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<> azimuthDist(-maxAzimuthDeg, maxAzimuthDeg);
    std::uniform_real_distribution<> elevationDist(-maxElevationDeg, maxElevationDeg);

    for (int i = 0; i < numRays; ++i) {
        double azimuth = azimuthDist(gen) / 180.0 * 3.1415;
        double elevation = elevationDist(gen) / 180.0 * 3.1415;
        traceSingleRayWithID(fiber, i, azimuth, elevation);
    }
}

void traceMultipleRaysSegmented(const Fiber &fiber, int totalRays, double maxAzimuthDeg, double maxElevationDeg) {
    // Benadering: probeer een rooster van sqrt(N) x sqrt(N)
    int elevationSteps = static_cast<int>(std::sqrt(totalRays));
    int azimuthSteps = (totalRays + elevationSteps - 1) / elevationSteps;  // afronden naar boven

    int rayID = 0;

    double azimuthMin = -maxAzimuthDeg;
    double elevationMin = -maxElevationDeg;
    double azimuthStep = (2 * maxAzimuthDeg) / std::max(1, azimuthSteps - 1);
    double elevationStep = (2 * maxElevationDeg) / std::max(1, elevationSteps - 1);

    for (int i = 0; i < azimuthSteps; ++i) {
        double azimuthDeg = azimuthMin + i * azimuthStep;
        for (int j = 0; j < elevationSteps; ++j) {
            if (rayID >= totalRays) break;  // stop zodra het juiste aantal is bereikt

            double elevationDeg = elevationMin + j * elevationStep;

            double azimuthRad = azimuthDeg * 3.1415 / 180.0;
            double elevationRad = elevationDeg * 3.1415 / 180.0;

            traceSingleRayWithID(fiber, rayID++, azimuthRad, elevationRad);
        }
    }
}


//lambertian distribution
void traceLed(const Fiber &fiber, int numRays, double maxAzimuthDeg, double maxElevationDeg) {
   if (numRays < 20) numRays = 20;

    // Convert degrees to radians
    double maxAzimuthRad = maxAzimuthDeg * M_PI / 180.0;
    double maxElevationRad = maxElevationDeg * M_PI / 180.0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> uniform(0.0, 1.0);

    int rayID = 0;
    while (rayID < numRays) {
        // Sample azimuth uniformly
        double phi = (uniform(gen) * 2.0 - 1.0) * maxAzimuthRad; // [-maxAzimuthRad, +maxAzimuthRad]

        // Sample elevation with cosine-weighted distribution
        // cos(theta) uniformly in [cos(maxElevationRad), 1]
        double cosThetaMin = std::cos(maxElevationRad);
        double cosTheta = uniform(gen) * (1.0 - cosThetaMin) + cosThetaMin;
        double theta = std::acos(cosTheta);

        // Optionally, you can filter out rays outside the maxAzimuth/maxElevation if you want a strict cone

        traceSingleRayWithID(fiber, rayID++, phi, theta);
    }
}


int main(){
    double length_fiber = 100;
    double width_fiber = 5;
    double height_fiber = 5;  

    Fiber fiber(width_fiber, height_fiber, length_fiber);

    // Metadata
    printf("fiber_length,%f\n", fiber.getLength());
    printf("fiber_top_y,%f\nfiber_bottom_y,%f\n", fiber.getTopY(), fiber.getBottomY());
    printf("fiber_left_z,%f\nfiber_right_z,%f\n", fiber.getTopZ(), fiber.getBottomZ());

    // CSV-header
    std::cout << "id,x,y,z\n";

    int aantalRays = 100000;
    double maxAzimuth = 60;   // in degrees
    double maxElevation = 60; // in degrees
    //traceMultipleRaysRandom(fiber, aantalRays, maxAzimuth, maxElevation);

    //traceMultipleRaysSegmented(fiber,aantalRays , maxAzimuth, maxElevation);
    

    auto start = std::chrono::steady_clock::now();

    traceLed(fiber,aantalRays,maxAzimuth,maxElevation);

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time: " << elapsed << " ms\n";

    return 0;
}