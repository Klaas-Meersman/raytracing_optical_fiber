#include "ray.hpp"
#include <cmath>





//constructor def
Ray::Ray(Coordinate start, double angleOfDeparture, double maxLength)
    :start(start),angleOfDeparture(angleOfDeparture/180 * std::numbers::pi),maxLength(maxLength){
}


Ray::~Ray(){
}


std::vector<Coordinate> Ray::generateStraightPath(double dx){
    std::vector<Coordinate> path;
    Coordinate current = start;
    double_t dy = std::tan(angleOfDeparture) * dx;
    for (double i = 0; i < maxLength ;i+=dx) {
        current.x += dx;
        current.y += dy;
        path.push_back(current);
    }
    return path;
}


