#include <cstdio>
#include <iostream>
// include math library for pi
#include <cmath>
#include <numbers>
#include "coordinate.hpp"
#include "ray.hpp"
#include <random>

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


// Lambertian distribution: only rays from 270° to 90° azimuth, max at 0°
void traceLed(const Fiber &fiber, int numRays, double maxAngleDeg) {
    if (numRays < 20) numRays = 20;


     // Max elevation angle
    // Azimuth: [360-maxAngleDeg, 360°) U [0°, maxAngleDeg)
    double minAzimuthDeg1 = 360.0 - maxAngleDeg;
    double maxAzimuthDeg1 = 360.0;
    double minAzimuthDeg2 = 0.0;
    double maxAzimuthDeg2 = maxAngleDeg;
    double maxElevationDeg = maxAngleDeg;
    double maxElevationRad = maxElevationDeg * M_PI / 180.0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> uniform(0.0, 1.0);

    int rayCount = 0;
    int rayID = 0;
    while (rayCount < numRays) {
        // Sample azimuth in [270°, 360°) U [0°, 90°)
        double u = uniform(gen);
        double phiDeg;
        if (u < 0.5) {
            phiDeg = minAzimuthDeg1 + (maxAzimuthDeg1 - minAzimuthDeg1) * (u / 0.5);
        } else {
            phiDeg = minAzimuthDeg2 + (maxAzimuthDeg2 - minAzimuthDeg2) * ((u - 0.5) / 0.5);
        }
        double phi = phiDeg * M_PI / 180.0;

        // Cosine-weighted elevation (Lambertian)
        double cosThetaMin = std::cos(maxElevationRad);
        double cosTheta = uniform(gen) * (1.0 - cosThetaMin) + cosThetaMin;
        double theta = std::acos(cosTheta);

        // Randomly flip to negative y hemisphere
        if (uniform(gen) < 0.5) {
            theta = -theta;
        }

        traceSingleRayWithID(fiber, rayID++, phi, theta);
        ++rayCount;
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

    int aantalRays = 500;
    double maxAngleDeg = 30;

    //traceMultipleRaysRandom(fiber, aantalRays, maxAzimuth, maxElevation);

    //traceMultipleRaysSegmented(fiber,aantalRays , maxAzimuth, maxElevation);
    traceLed(fiber,aantalRays,maxAngleDeg);

    return 0;
}