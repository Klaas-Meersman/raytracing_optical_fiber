#include <cstdio>
#include <iostream>
#include <cmath>
#include <numbers>
#include "coordinate.hpp"
#include "ray.hpp"
#include "fiber.hpp"

// Only include <vector> and use std::vector in host code
#ifndef __CUDA_ARCH__
#include <vector>
#endif

// CPU function to trace rays (host only)
#ifndef __CUDA_ARCH__
void traceRaysCPU(const Fiber &fiber, int numRays)
{
    std::vector<Ray> rays;
    double_t angleRadians = 0;

    // lambertian distribution
    for (int i = 0; i < numRays; ++i)
    {
        double u = static_cast<double>(rand()) / RAND_MAX;
        double theta = std::asin(u);

        if (rand() % 2 == 0)
            angleRadians = theta;
        else
            angleRadians = 3 * std::numbers::pi / 2 + theta;

        Coordinate startCo = Coordinate(0, 0);
        Ray ray = Ray(startCo, angleRadians, &fiber);
        rays.push_back(ray);
    }

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
    }
}
#endif

// GPU kernel and GPU wrapper function remain as before, but use Fiber*
__global__ void traceRayGPU(const Fiber* fiber, Ray *rays, int numRays)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRays) {
        while (!rays[idx].getEndHitFiber())
        {
            rays[idx].propagateRay();
        }
    }
}

void runTraceRayGPU(const Fiber* fiber, Ray *rays, int numRays)
{
    int blockSize = 256;
    int numBlocks = (numRays + blockSize - 1) / blockSize;

    // Allocate GPU memory for Fiber
    Fiber* GPU_fiber;
    cudaMalloc((void**)&GPU_fiber, sizeof(Fiber));
    cudaMemcpy(GPU_fiber, fiber, sizeof(Fiber), cudaMemcpyHostToDevice);

    Ray *GPU_rays;
    cudaMalloc((void **)&GPU_rays, numRays * sizeof(Ray));
    cudaMemcpy(GPU_rays, rays, numRays * sizeof(Ray), cudaMemcpyHostToDevice);

    traceRayGPU<<<numBlocks, blockSize>>>(GPU_fiber, GPU_rays, numRays);
    cudaDeviceSynchronize();
    cudaMemcpy(rays, GPU_rays, numRays * sizeof(Ray), cudaMemcpyDeviceToHost);

    cudaFree(GPU_rays);
    cudaFree(GPU_fiber);

    // print the rays calculated on the GPU
    for (int i = 0; i < numRays; ++i)
    {
        if (rays[i].getEndHitFiber())
        {
            std::cout << rays[i].getEnd().x << ", " << rays[i].getEnd().y << std::endl;
        }
    }
}

int main()
{
    double length_fiber = 100;
    double width_fiber = 5;
    Fiber fiber(width_fiber, length_fiber);
    printf("fiber_length,%f\n", fiber.getLength());
    printf("fiber_top_y,%f\nfiber_bottom_y,%f\n", fiber.getTopY(), fiber.getBottomY());
    printf("x,y\n");

    int numRays = 10000000;

    // Only use std::vector on the host
    #ifndef __CUDA_ARCH__
    std::vector<Ray> rays;
    double_t angleRadians = 0;
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
            angleRadians = 3 * M_PI / 2 + theta;
        }

        Coordinate startCo = Coordinate(0, 0);
        Ray ray = Ray(startCo, angleRadians, &fiber);
        rays.push_back(ray);
    }
    // Pass the data pointer to the GPU function
    runTraceRayGPU(&fiber, rays.data(), numRays);
    #endif

    return 0;
}
