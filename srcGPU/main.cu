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


// GPU kernel and GPU wrapper function remain as before, but use Fiber*
__global__ void traceRayGPU(const Fiber* fiber, Ray *rays, int numRays)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRays) {
        while (!rays[idx].getEndHitFiber())
        {
            rays[idx].propagateRay();
            printf("Ray %d: Start (%f, %f), End (%f, %f)\n", idx, rays[idx].getStart().x, rays[idx].getStart().y, rays[idx].getEnd().x, rays[idx].getEnd().y);
        }
    }
}




void runTraceRayGPU(const Fiber* fiber,int numRays)
{
    // Only use std::vector on the host
    #ifndef __CUDA_ARCH__
    std::vector<Ray> rays_vector;
    double_t angleRadians = 0;

    //lambertian distrubtion
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
        Ray ray = Ray(startCo, angleRadians, fiber);
        rays_vector.push_back(ray);
    }
    Ray* rays = rays_vector.data();
    
   

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
    #endif
}


__global__  void debugRayGPU(Fiber* fiber, Ray* input_ray, Ray* output_rays, int maxBounces)
{
    int idx = 0;

    while(!input_ray->getEndHitFiber() && idx< maxBounces)
    {
        printf("Passing in main while not hit end ray propagtion\n");
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

/*     // Print all intermediate rays
    for (int i = 0; i < maxBounces; ++i) {
        // Stop printing if the ray has hit the fiber end
        if (rays[i].getEndHitFiber() || (i > 0 && rays[i].getStart().x == rays[i].getEnd().x && rays[i].getStart().y == rays[i].getEnd().y)) {
            std::cout << "Bounce " << i << ": (" << rays[i].getStart().x << ", " << rays[i].getStart().y << ") -> ("
                      << rays[i].getEnd().x << ", " << rays[i].getEnd().y << ") [endHitFiber=" << rays[i].getEndHitFiber() << "]\n";
            break;
        }
        std::cout << "Bounce " << i << ": (" << rays[i].getStart().x << ", " << rays[i].getStart().y << ") -> ("
                  << rays[i].getEnd().x << ", " << rays[i].getEnd().y << ") [endHitFiber=" << rays[i].getEndHitFiber() << "]\n";
    } */

    cudaFree(output_rays);
    cudaFree(GPU_fiber);
    delete[] rays;
}

int main()
{
    //Density simulation
 /*    double length_fiber = 100;
    double width_fiber = 5;
    Fiber fiber(width_fiber, length_fiber);
    printf("fiber_length,%f\n", fiber.getLength());
    printf("fiber_top_y,%f\nfiber_bottom_y,%f\n", fiber.getTopY(), fiber.getBottomY());
    printf("x,y\n");

    int numRays = 1000000;

    runTraceRayGPU(&fiber, numRays); */


    double length_fiber = 100;
    double width_fiber = 5;
    Fiber fiber(width_fiber, length_fiber);
    printf("fiber_length,%f\n", fiber.getLength());
    printf("fiber_top_y,%f\nfiber_bottom_y,%f\n", fiber.getTopY(), fiber.getBottomY());
    printf("x,y\n");
    runDebugTraceRayGPU(&fiber);

    return 0;
}
