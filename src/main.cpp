#include <cstdio>
#include <iostream>
//include math library for pi
#include <cmath>
#include <numbers>

#include "coordinate.hpp"
#include "ray.hpp"


int main(){

    double length_fiber = 10;
    double width_fiber = 1;


    Coordinate startCo = Coordinate(0,0);
    double angleDegrees = 50;
    double angleRadians = angleDegrees / 180 * std::numbers::pi;
    Ray r = Ray(startCo,angleRadians,length_fiber + width_fiber);
    std::vector<Coordinate> rayCo = r.generateStraightPath(0.4);
    for (int i = 0; i < rayCo.size(); i++) {
        std::cout << rayCo[i] << std::endl;
    }

    r.generateBounceRay(r);


    return 0;
}
