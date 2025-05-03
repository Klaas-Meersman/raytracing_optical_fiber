#include <cstdio>
#include <iostream>
//include math library for pi
#include <cmath>
#include <numbers>

#include "coordinate.hpp"
#include "ray.hpp"


int main(){

    double length_fiber = 15;
    double width_fiber = 5;
    Fiber fiber = Fiber(width_fiber,length_fiber);

    Coordinate startCo = Coordinate(0,0);
    double_t angleDegrees = 40;
    double_t angleRadians = angleDegrees / 180 * std::numbers::pi;

    printf("First ray\n");
    Ray r = Ray(startCo,angleRadians,3);
    //std::cout << "Ray: " << r << std::endl;
    std::vector<Coordinate> rayCo = r.generateStraightPath(fiber, 0.2);
    for (int i = 0; i < rayCo.size(); i++) {
        std::cout << rayCo[i] << std::endl;
    }



    printf("Bounce 1\n");
    Ray bouncedRay1= r.generateBounceRay(fiber);
    //std::cout << "Bounced Ray 1: " << bouncedRay1 << std::endl;
    std::vector<Coordinate> bouncedRayCo1 = bouncedRay1.generateStraightPath(fiber, 0.2);
    for (int i = 0; i < bouncedRayCo1.size(); i++) {
        std::cout << bouncedRayCo1[i] << std::endl;
    }

    printf("Bounce 2\n");
    Ray bouncedRay2= bouncedRay1.generateBounceRay(fiber);
    //std::cout << "Bounced Ray 2: " << bouncedRay2 << std::endl;
    std::vector<Coordinate> bouncedRayCo2 = bouncedRay2.generateStraightPath(fiber, 0.2);
    for (int i = 0; i < bouncedRayCo2.size(); i++) {
        std::cout << bouncedRayCo2[i] << std::endl;
    }

    printf("Bounce 3\n");
    Ray bouncedRay3= bouncedRay2.generateBounceRay(fiber);
    //std::cout << "Bounced Ray 3: " << bouncedRay3 << std::endl;
    std::vector<Coordinate> bouncedRayCo3 = bouncedRay3.generateStraightPath(fiber, 0.2);
    for (int i = 0; i < bouncedRayCo3.size(); i++) {
        std::cout << bouncedRayCo3[i] << std::endl;
    }

    Ray bouncedRay4= bouncedRay3.generateBounceRay(fiber);
    if(&bouncedRay4 == &bouncedRay3){
        printf("Ray ends here\n");
    }
    std::vector<Coordinate> bouncedRayCo4 = bouncedRay4.generateStraightPath(fiber, 0.2);
    for (int i = 0; i < bouncedRayCo4.size(); i++) {
        std::cout << bouncedRayCo4[i] << std::endl;
    } 
   /*  printf("Bounce 4\n");
    Ray bouncedRay4= bouncedRay3.generateBounceRay(fiber);
    //std::cout << "Bounced Ray 4: " << bouncedRay4 << std::endl;
    std::vector<Coordinate> bouncedRayCo4 = bouncedRay4.generateStraightPath(fiber, 0.2);
    for (int i = 0; i < bouncedRayCo4.size(); i++) {
        std::cout << bouncedRayCo4[i] << std::endl;
    } */

 

    return 0;
}
