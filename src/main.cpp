#include <cstdio>
#include <iostream>

#include "coordinate.hpp"
#include "ray.hpp"


int main(){

    double length_fiber = 10;
    double width_fiber = 1;


    Coordinate startCo = Coordinate(0,0);

    Ray r = Ray(startCo,50,length_fiber + width_fiber);
    std::vector<Coordinate> rayCo = r.generateStraightPath(0.4);
    for (int i = 0; i < rayCo.size(); i++) {
        std::cout << rayCo[i] << std::endl;
    }


    return 0;
}
