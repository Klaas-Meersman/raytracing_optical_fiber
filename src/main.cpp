#include <cstdio>
#include <iostream>
// include math library for pi
#include <cmath>
#include <numbers>

#include "coordinate.hpp"
#include "ray.hpp"

int main()
{

    double length_fiber = 90;
    double width_fiber = 5;
    Fiber fiber = Fiber(width_fiber, length_fiber);
    printf("fiber_length,%f\n", fiber.getLength());
    printf("fiber_top_y,%f\nfiber_bottom_y,%f\n", fiber.getTopY(), fiber.getBottomY());

    printf("x,y\n");
    
    Coordinate startCo = Coordinate(0, 0);
    double_t angleDegrees = 30;
    double_t angleRadians = angleDegrees / 180 * std::numbers::pi;

    Ray r = Ray(startCo, angleRadians, fiber);
    //std::cout << "Start ray: " << r << std::endl;
    std::vector<Coordinate> rayCo = r.generateStraightPath(0.2);
    for (int i = 0; i < rayCo.size(); i++){
        std::cout << rayCo[i] << std::endl;
    }

    Ray bouncedRay = r;
    while (!bouncedRay.getEndHitFiber()){
        bouncedRay = bouncedRay.generateBounceRay(fiber);
        //std::cout << "Bounced ray: " << bouncedRay << std::endl;
        std::vector<Coordinate> bouncedRayCo = bouncedRay.generateStraightPath(0.2);
        for (int i = 0; i < bouncedRayCo.size(); i++){
            std::cout << bouncedRayCo[i] << std::endl;
        }
    }
    return 0;
}
