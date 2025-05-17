#include <cstdio>
#include <iostream>
#include <cmath>
#include "coordinate.hpp"
#include "ray.hpp"
#include "fiber.hpp"
#include <chrono>



__global__ void traceRayGPU(const Fiber* fiber, Ray *rays, int numRays)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRays) {
        while (!rays[idx].getEndHitFiber()) {
            rays[idx] = rays[idx].propagateRay();

        }
    }
}


__device__ double rand_uniform(int idx, int salt = 0) {
    // Simple LCG for per-thread reproducible pseudo-random numbers
    unsigned int seed = 123456789u + idx * 65497u + salt * 9973u;
    seed = (1103515245u * seed + 12345u) & 0x7fffffff;
    return seed / double(0x7fffffff);
}

__global__ void initRays(Ray* rays, int numRays, const Fiber* fiber) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;

    double maxAngleDeg = 70.0; // Only change this value!
    double minAzimuthDeg1 = 360.0 - maxAngleDeg;
    double maxAzimuthDeg1 = 360.0;
    double minAzimuthDeg2 = 0.0;
    double maxAzimuthDeg2 = maxAngleDeg;
    double maxElevationDeg = maxAngleDeg;
    double maxElevationRad = maxElevationDeg * M_PI / 180.0;

    // Per-thread pseudo-random numbers
    double u = rand_uniform(idx, 0);
    double phiDeg;
    if (u < 0.5) {
        phiDeg = minAzimuthDeg1 + (maxAzimuthDeg1 - minAzimuthDeg1) * (u / 0.5);
    } else {
        phiDeg = minAzimuthDeg2 + (maxAzimuthDeg2 - minAzimuthDeg2) * ((u - 0.5) / 0.5);
    }
    double phi = phiDeg * M_PI / 180.0;

    // Cosine-weighted elevation (Lambertian)
    double v = rand_uniform(idx, 1);
    double cosThetaMin = cos(maxElevationRad);
    double cosTheta = v * (1.0 - cosThetaMin) + cosThetaMin;
    double theta = acos(cosTheta);

    // Randomly flip to negative y hemisphere
    double w = rand_uniform(idx, 2);
    if (w < 0.5) {
        theta = -theta;
    }

    rays[idx] = Ray(Coordinate(0, 0, 0), phi, theta, fiber);
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

    

    cudaMalloc((void**)&GPU_rays, numRays * sizeof(Ray));
    initRays<<<numBlocks, blockSize>>>(GPU_rays, numRays, GPU_fiber);
    cudaDeviceSynchronize();
    
    // Allocate GPU memory for Fiber

    traceRayGPU<<<numBlocks, blockSize>>>(GPU_fiber, GPU_rays, numRays);
    cudaDeviceSynchronize();

    cudaMemcpy(ray_array, GPU_rays, numRays * sizeof(Ray), cudaMemcpyDeviceToHost);

    cudaFree(GPU_rays);
    cudaFree(GPU_fiber);
    
    if(printDensity){
    // print the rays density at end of fiber calculated on the GPU
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