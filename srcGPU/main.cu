#include <cstdio>
#include <iostream>
#include <cmath>
#include "coordinate.hpp"
#include "ray.hpp"
#include "fiber.hpp"
#include <chrono>



__global__ void traceRayGPU(const Fiber* fiber, Ray *rays, int numRays)
{
    const int maxbounces = 1000000;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRays) {
        int bounce = 0;
        while (!rays[idx].getEndHitFiber() && bounce < maxbounces) {
            rays[idx] = rays[idx].propagateRay();
            bounce++;
            /* if(maxbounces == bounce){
                printf("Hit max bounces\n");
            } */
        }
    }
}

/* __global__ void initRays(Ray* rays, int numRays, const Fiber* fiber) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRays) {
        double u = static_cast<double>(idx) / numRays; // or pass in randoms
        double theta = asin(u);
        double angleRadians = (idx % 2 == 0) ? theta : 3 * M_PI / 2 + theta;
        rays[idx] = Ray(Coordinate(0, 0, 0), angleRadians, angleRadians, fiber);
        //printf("Angle: %f\n", angleRadians);
    }
} */

__device__ double nonlinear_angle(int i, int steps, double maxDeg, double gamma) {
    if (steps <= 1) return 0.0;
    double t = (double)i / (steps - 1);      // 0 .. 1
    double x = 2.0 * t - 1.0;                // -1 .. 1
    double curved = x * pow(fabs(x), gamma); // nonlinear distribution
    double angle = curved * maxDeg;          // -maxDeg .. maxDeg
    return angle * (M_PI / 180.0);           // degrees to radians
}

__global__ void initRays(
    Ray* rays, int numRays, const Fiber* fiber
) {
    double maxAzimuthDeg = 30;
    double maxElevationDeg = 30;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;

    // Calculate grid dimensions as in your CPU code
    int elevationSteps = (int)sqrt((double)numRays);
    int azimuthSteps = (numRays + elevationSteps - 1) / elevationSteps;
    double gamma = 3.0;

    // Map idx to 2D grid
    int elevationIdx = idx / azimuthSteps;
    int azimuthIdx   = idx % azimuthSteps;

    // Guard against overrun (last row may be incomplete)
    if (elevationIdx >= elevationSteps) return;

    double azimuth   = nonlinear_angle(azimuthIdx, azimuthSteps,   maxAzimuthDeg,   gamma);
    double elevation = nonlinear_angle(elevationIdx, elevationSteps, maxElevationDeg, gamma);

    rays[idx] = Ray(Coordinate(0, 0, 0), azimuth, elevation, fiber);
}


void runTraceRayGPU(Fiber* fiber,int numRays)
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
    
    // print the rays calculated on the GPU
    /* for (int i = 0; i < numRays; ++i) {
        std::cout << ray_array[i].getEnd().x << ", " << ray_array[i].getEnd().y << ", " << ray_array[i].getEnd().z  <<  std::endl;
    } */
    delete[] ray_array;
}


/* __global__  void debugRayGPU(Fiber* fiber, Ray* input_ray, Ray* output_rays, int maxBounces)
{
    int idx = 0;

    while(!input_ray->getEndHitFiber() && idx< maxBounces)
    {
        //printf("Passing in main while not hit end ray propagtion\n");
        input_ray->propagateRay();
        output_rays[idx] = *input_ray;
        idx++;
    }
}

__global__ void initRay(Ray* d_ray, Coordinate start, double_t angle, const Fiber* fiber) {
    *d_ray = Ray(start, angle, fiber);
}

void runDebugTraceRayGPU(Fiber* fiber){
    double_t angleDegrees = 30;
    double_t angleRadians = angleDegrees * (M_PI / 180.0);

    // Allocate the fiber on the GPU
    Fiber* GPU_fiber;
    cudaMalloc((void**)&GPU_fiber, sizeof(Fiber));
    cudaMemcpy(GPU_fiber, fiber, sizeof(Fiber), cudaMemcpyHostToDevice);

    Coordinate startCo = Coordinate(0, 0);
    // IMPORTANT: Use GPU_fiber as the pointer!
    Ray* d_ray;
    cudaMalloc((void**)&d_ray, sizeof(Ray));

    int maxBounces = 100;

    Ray* output_rays;
    cudaMalloc((void**)&output_rays, maxBounces * sizeof(Ray));

    initRay<<<1, 1>>>(d_ray, startCo, angleRadians, GPU_fiber);
    cudaDeviceSynchronize();

    debugRayGPU<<<1, 1>>>(GPU_fiber, d_ray, output_rays, maxBounces);
    cudaDeviceSynchronize();

    // Copy results back
    Ray* rays = new Ray[maxBounces];
    cudaMemcpy(rays, output_rays, maxBounces * sizeof(Ray), cudaMemcpyDeviceToHost);

    std::cout << rays[0].getStart().x << ", " << rays[0].getStart().y << std::endl;

    for (int i = 0; i < maxBounces; ++i) {
        // Stop printing if the ray has hit the fiber end
        if (rays[i].getEndHitFiber() || (i > 0 && rays[i].getStart().x == rays[i].getEnd().x && rays[i].getStart().y == rays[i].getEnd().y)) {
            std::cout << rays[i].getEnd().x << ", " << rays[i].getEnd().y << std::endl;
            break;
        }
        std::cout << rays[i].getEnd().x << ", " << rays[i].getEnd().y << std::endl;
    }

    //

    cudaFree(output_rays);
    cudaFree(GPU_fiber);
    cudaFree(d_ray);
    delete[] rays;
}

int main()
{
    double length_fiber = 100;
    double width_fiber = 5;
    Fiber fiber(width_fiber, length_fiber);
    printf("fiber_length,%f\n", fiber.getLength());
    printf("fiber_top_y,%f\nfiber_bottom_y,%f\n", fiber.getTopY(), fiber.getBottomY());
    printf("x,y\n");
    runDebugTraceRayGPU(&fiber);

    return 0;
} */


int main(){
    //Density simulation

    double length_fiber = 100;
    double width_fiber = 5;
    double height_fiber = 5;
    Fiber fiber(width_fiber, height_fiber, length_fiber);
    printf("fiber_length,%f\n", fiber.getLength());
    printf("fiber_top_y,%f\nfiber_bottom_y,%f\n", fiber.getTopY(), fiber.getBottomY());
    printf("x,y\n");

    int numRays = 100000000;

    auto start = std::chrono::steady_clock::now();

    runTraceRayGPU(&fiber, numRays);

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time: " << elapsed << " ms\n";

}
