#include "ray.hpp"
#include "fiber.hpp"
#include <cmath>
#include <numbers>





//constructor def
Ray::Ray(Coordinate start, double_t angleOfDeparture, int maxLength)
    : start(start), angleOfDeparture(angleOfDeparture), maxLength(maxLength) {
    if (angleOfDeparture > 0 && angleOfDeparture < std::numbers::pi / 2) {
        direction = Direction::UP;
    } else if (angleOfDeparture > 3 * std::numbers::pi / 4 && angleOfDeparture < 2 * std::numbers::pi) {
        direction = Direction::DOWN;
    } else if (angleOfDeparture > std::numbers::pi / 2 && angleOfDeparture < 3 * std::numbers::pi / 4) {
        throw std::invalid_argument("This is an unexpected direction as all rays should go from left to right. Angle of departure is between pi/2 and 3pi/4: " + std::to_string(angleOfDeparture));
    } else {
        throw std::invalid_argument("This is an unexpected direction as all rays should go from left to right. Angle of departure is: " + std::to_string(angleOfDeparture/std::numbers::pi));
    }
}



Ray::~Ray(){
}

//we can't really use this. would be to many calculations that are not necesary
std::vector<Coordinate> Ray::generateStraightPath(Fiber fiber, double dx){
    std::vector<Coordinate> path;
    Coordinate current = start;
    double_t dy = std::tan(angleOfDeparture) * dx;

    double i = 0;
    while (current.y <= fiber.getTopY() && current.y >= fiber.getBottomY()) {
        current.x += dx;
        current.y += dy;
        path.push_back(current);
        i+=dx;
    }

    return path;
}


//generate start coordinate with a = b/(tan@)
Ray Ray::generateBounceRay(Fiber fiber){
    //printf("generating bounce ray\n");
    Coordinate start;
    double_t startX,startY,angleOfDeparture;

    if(this->direction == Ray::Direction::UP){
        startY = fiber.getTopY();
        startX = this->start.x + (fiber.getTopY() - this->start.y)/std::tan(this->angleOfDeparture); //so 
    }else if(this->direction == Ray::Direction::DOWN){
        startY = fiber.getBottomY();
        startX = this->start.x + (fiber.getBottomY() - this->start.y)/std::tan(this->angleOfDeparture); //so
    }
    start = Coordinate(startX,startY);
    angleOfDeparture =  std::numbers::pi*2 - this->angleOfDeparture;
    return Ray(start, angleOfDeparture, fiber.maxSingleRayInFiber(startX));
}







