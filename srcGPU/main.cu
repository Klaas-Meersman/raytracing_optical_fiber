#include <cstdio>
#include <iostream>
#include <cmath>
#include "coordinate.hpp"
#include "ray.hpp"
#include "fiber.hpp"
#include <chrono>
#include <curand_kernel.h>



__global__ void traceRayGPU(const Fiber* fiber, Ray *rays, int numRays)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRays) {
        while (!rays[idx].getEndHitFiber()) {
            rays[idx] = rays[idx].propagateRay();

        }
    }
}


__global__ void initCurandStates(curandState *states, int numRays, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRays) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void initRays(Ray* rays, int numRays, const Fiber* fiber, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;

    curandState localState = states[idx];

    double maxAngleDeg = 70.0;
    double minAzimuthDeg1 = 360.0 - maxAngleDeg;
    double maxAzimuthDeg1 = 360.0;
    double minAzimuthDeg2 = 0.0;
    double maxAzimuthDeg2 = maxAngleDeg;
    double maxElevationDeg = maxAngleDeg;
    double maxElevationRad = maxElevationDeg * M_PI / 180.0;

    // Use cuRAND for random numbers
    double u = curand_uniform_double(&localState);
    double phiDeg;
    if (u < 0.5) {
        phiDeg = minAzimuthDeg1 + (maxAzimuthDeg1 - minAzimuthDeg1) * (u / 0.5);
    } else {
        phiDeg = minAzimuthDeg2 + (maxAzimuthDeg2 - minAzimuthDeg2) * ((u - 0.5) / 0.5);
    }
    double phi = phiDeg * M_PI / 180.0;

    double v = curand_uniform_double(&localState);
    double cosThetaMin = cos(maxElevationRad);
    double cosTheta = v * (1.0 - cosThetaMin) + cosThetaMin;
    double theta = acos(cosTheta);

    double w = curand_uniform_double(&localState);
    if (w < 0.5) {
        theta = -theta;
    }

    rays[idx] = Ray(Coordinate(0, 0, 0), phi, theta, fiber);

    // Save state back
    states[idx] = localState;
}


void runTraceRayGPU(Fiber* fiber,int numRays,bool printDensity)
{
    int blockSize = 256;
    int numBlocks = (numRays + blockSize - 1) / blockSize;
    Ray* ray_array = new Ray[numRays];
    Ray* GPU_rays;

    Fiber* GPU_fiber;
    cudaMalloc((void**)&GPU_fiber, sizeof(Fiber));
    cudaMemcpy(GPU_fiber, fiber, sizeof(Fiber), cudaMemcpyHostToDevice);

    // Allocate and initialize cuRAND states
    curandState* d_states;
    cudaMalloc(&d_states, numRays * sizeof(curandState));
    unsigned long seed = static_cast<unsigned long>(time(NULL));
    initCurandStates<<<numBlocks, blockSize>>>(d_states, numRays, seed);
    cudaDeviceSynchronize();

    cudaMalloc((void**)&GPU_rays, numRays * sizeof(Ray));
    initRays<<<numBlocks, blockSize>>>(GPU_rays, numRays, GPU_fiber, d_states);
    cudaDeviceSynchronize();

    traceRayGPU<<<numBlocks, blockSize>>>(GPU_fiber, GPU_rays, numRays);
    cudaDeviceSynchronize();

    cudaMemcpy(ray_array, GPU_rays, numRays * sizeof(Ray), cudaMemcpyDeviceToHost);

    cudaFree(GPU_rays);
    cudaFree(GPU_fiber);
    cudaFree(d_states); // <--- free cuRAND states

    if(printDensity){
        for (int i = 0; i < numRays; ++i) {
            std::cout << ray_array[i].getEnd().x << ", " << ray_array[i].getEnd().y << ", " << ray_array[i].getEnd().z  <<  std::endl;
        }
    }
    delete[] ray_array;
}


int main(){
    //Density simulation
    bool printDensity = true;
    double length_fiber = 100;
    double width_fiber = 5;
    double height_fiber = 5;
    Fiber fiber(width_fiber, height_fiber, length_fiber);
    printf("fiber_length,%f\n", fiber.getLength());
    printf("fiber_top_y,%f\nfiber_bottom_y,%f\n", fiber.getTopY(), fiber.getBottomY());
    printf("fiber_top_z,%f\nfiber_bottom_z,%f\n", fiber.getTopZ(), fiber.getBottomZ());
    printf("x,y,z\n");

    int numRays = 1000000;

    auto start = std::chrono::steady_clock::now();

    runTraceRayGPU(&fiber, numRays,printDensity);

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << elapsed << "ms\n";

}