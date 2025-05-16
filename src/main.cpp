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
void traceLed(const Fiber &fiber, int numRays, double maxAzimuthDeg, double maxElevationDeg) {
    if (numRays < 20) numRays = 20;

    int elevationSteps = static_cast<int>(std::sqrt(numRays));
    int azimuthSteps = (numRays + elevationSteps - 1) / elevationSteps;

    int rayID = 0;

    double gamma = 3; // exponent voor verdeling — hoe hoger, hoe dichter bij 0

    auto generateNonlinearSteps = [&](int steps, double maxDeg) -> std::vector<double> {
        std::vector<double> angles;
        for (int i = 0; i < steps; ++i) {
            double t = static_cast<double>(i) / (steps - 1);   // 0 .. 1
            double x = 2 * t - 1;                              // -1 .. 1
            double curved = x * std::pow(std::abs(x), gamma); // meer punten rond 0
            double angle = curved * maxDeg;                   // -maxDeg .. maxDeg
            angles.push_back(angle * 3.1415 / 180.0);           // graden naar radialen
        }
        return angles;
    };

    std::vector<double> azimuthAngles = generateNonlinearSteps(azimuthSteps, maxAzimuthDeg);
    std::vector<double> elevationAngles = generateNonlinearSteps(elevationSteps, maxElevationDeg);

    for (double elevation : elevationAngles) {
        for (double azimuth : azimuthAngles) {
            if (rayID >= numRays) break;
            traceSingleRayWithID(fiber, rayID++, azimuth, elevation);
        }
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

    int aantalRays = 1000;
    double maxAzimuth = 60;   // in degrees
    double maxElevation = 60; // in degrees
    //traceMultipleRaysRandom(fiber, aantalRays, maxAzimuth, maxElevation);

    //traceMultipleRaysSegmented(fiber,aantalRays , maxAzimuth, maxElevation);
    traceLed(fiber,aantalRays,maxAzimuth,maxElevation);

    return 0;
}