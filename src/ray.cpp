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
    path.push_back(current);
    double i = 0;
    while (true) {
        current.x += dx;
        current.y += dy;
        if (current.y > fiber.getTopY() || current.y < fiber.getBottomY()) {
            break;
        }

        //add the end point to the path if it is the last ray
        if(this->endHitFiber){
            if(current.x > fiber.getLength()){
                path.push_back(this->end);
                break;
            }
        }
        path.push_back(current);
    }
    return path;
}


Ray Ray::generateBounceRay(Fiber fiber){
    Coordinate start;
    double_t startX,startY,angleOfDeparture;

    if(this->direction == Ray::Direction::UP){
        this->end.y = startY = fiber.getTopY();
        this->end.x = startX = this->start.x + (fiber.getTopY() - this->start.y)/std::tan(this->angleOfDeparture); //so 
    }else if(this->direction == Ray::Direction::DOWN){
        this->end.y = startY = fiber.getBottomY();
        this->end.x = startX = this->start.x + (fiber.getBottomY() - this->start.y)/std::tan(this->angleOfDeparture); //so
    }
    //check if it is the last ray
    if(startX > fiber.getLength()){
        this->endHitFiber = true;
        this->end.x = fiber.getLength();
        this->end.y = std::tan(this->angleOfDeparture) * (fiber.getLength() - this->start.x) + this->start.y;
        return *this; //incase its the last ray, we return the ray again instead of a new one
    }
    start = Coordinate(startX,startY);
    angleOfDeparture =  std::numbers::pi*2 - this->angleOfDeparture;
    Ray bouncedRay = Ray(start, angleOfDeparture, fiber.maxSingleRayInFiber(startX));
    return bouncedRay;
}







