#include <cstdio>
#include <iostream>
#include <cmath>
#include "coordinate.hpp"
#include "ray.hpp"
#include "fiber.hpp"
#include <chrono>
#include <curand_kernel.h>



__global__ void traceRayGPU(const Fiber* fiber, Ray *rays, int numRays){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRays) {
        while (!rays[idx].getEndHitFiber()) {
            rays[idx].propagateRay();
        }
    }
}

//to be able to use random numbers in the kernel
__global__ void initCurandStates(curandState *states, int numRays, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRays) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

/* __global__ void initRays(Ray* rays, int numRays, const Fiber* fiber, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;

    curandState localState = states[idx];

    double maxAngleDeg = 85.0;
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


void runTraceRayGPU(Fiber* fiber,int numRays)
{
    int blockSize = 128;
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
    
    for (int i = 0; i < numRays; ++i) {
        //std::cout << ray_array[i].getEnd().x << ", " << ray_array[i].getEnd().y << ", " << ray_array[i].getEnd().z  <<  std::endl;
        //printf version
        printf("%f, %f, %f\n", ray_array[i].getEnd().x, ray_array[i].getEnd().y, ray_array[i].getEnd().z);
    }
    delete[] ray_array;
} */

__global__ void initRaysBinned(Ray* rays, int numRays, const Fiber* fiber, curandState* states, double binMin, double binMax, double maxAngleDeg) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;

    curandState localState = states[idx];

    // Azimuth: [360-maxAngleDeg, 360°) U [0°, maxAngleDeg)
    double minAzimuthDeg1 = 360.0 - maxAngleDeg;
    double maxAzimuthDeg1 = 360.0;
    double minAzimuthDeg2 = 0.0;
    double maxAzimuthDeg2 = maxAngleDeg;

    double u = curand_uniform_double(&localState);
    double phiDeg;
    if (u < 0.5) {
        phiDeg = minAzimuthDeg1 + (maxAzimuthDeg1 - minAzimuthDeg1) * (u / 0.5);
    } else {
        phiDeg = minAzimuthDeg2 + (maxAzimuthDeg2 - minAzimuthDeg2) * ((u - 0.5) / 0.5);
    }
    double phi = phiDeg * M_PI / 180.0;

    // Elevation: sample uniformly within this bin's range
    double v = curand_uniform_double(&localState);

    // Lambertian: cosine-weighted elevation within bin
    double cosThetaMin = cos(binMax * M_PI / 180.0);
    double cosThetaMax = cos(binMin * M_PI / 180.0);
    double w = curand_uniform_double(&localState);
    double cosTheta = w * (cosThetaMax - cosThetaMin) + cosThetaMin;
    double theta = acos(cosTheta);

    // Randomly flip to negative y hemisphere
    double flip = curand_uniform_double(&localState);
    if (flip < 0.5) {
        theta = -theta;
    }

    rays[idx] = Ray(Coordinate(0, 0, 0), phi, theta, fiber);

    states[idx] = localState;
}


Ray* runTraceRayGPU(Fiber* fiber, int numRays){
    int blockSize = 256;
    int numBins = 16; // Number of angle bins (tune as needed)
    double maxAngleDeg = 85.0;

    // Allocate device fiber ONCE
    Fiber* GPU_fiber;
    cudaMalloc((void**)&GPU_fiber, sizeof(Fiber));
    cudaMemcpy(GPU_fiber, fiber, sizeof(Fiber), cudaMemcpyHostToDevice);

    // Prepare host-side arrays for results
    Ray* ray_array = new Ray[numRays];
    int raysPerBin = numRays / numBins;
    int rayOffset = 0;

    for (int bin = 0; bin < numBins; ++bin) {
        int raysInThisBin = (bin == numBins - 1) ? (numRays - rayOffset) : raysPerBin;

        // Allocate device memory for rays and cuRAND states for this bin
        Ray* GPU_rays;
        cudaMalloc(&GPU_rays, raysInThisBin * sizeof(Ray));
        curandState* d_states;
        cudaMalloc(&d_states, raysInThisBin * sizeof(curandState));

        // Initialize cuRAND states
        unsigned long seed = static_cast<unsigned long>(time(NULL)) + bin;
        int numBlocks = (raysInThisBin + blockSize - 1) / blockSize;
        initCurandStates<<<numBlocks, blockSize>>>(d_states, raysInThisBin, seed);

        // Calculate elevation angle range for this bin
        double binMin = -maxAngleDeg + (2.0 * maxAngleDeg) * bin / numBins;
        double binMax = -maxAngleDeg + (2.0 * maxAngleDeg) * (bin + 1) / numBins;

        // Kernel to initialize rays in this bin with elevation angles in [binMin, binMax]
        initRaysBinned<<<numBlocks, blockSize>>>(GPU_rays, raysInThisBin, GPU_fiber, d_states, binMin, binMax, maxAngleDeg);

        // Trace rays in this bin
        traceRayGPU<<<numBlocks, blockSize>>>(GPU_fiber, GPU_rays, raysInThisBin);

        // Copy results back to correct offset in host array
        cudaMemcpy(ray_array + rayOffset, GPU_rays, raysInThisBin * sizeof(Ray), cudaMemcpyDeviceToHost);

        // Free device memory for this bin
        cudaFree(GPU_rays);
        cudaFree(d_states);

        rayOffset += raysInThisBin;
    }

    cudaFree(GPU_fiber);

    return ray_array;
}

int main(){
    //Density simulation
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

    Ray* ray_array = runTraceRayGPU(&fiber, numRays);

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    for (int i = 0; i < numRays; ++i) {
        printf("%f, %f, %f\n", ray_array[i].getEnd().x, ray_array[i].getEnd().y, ray_array[i].getEnd().z);
    }
    printf("%lld ms\n", elapsed);

    delete[] ray_array;
}