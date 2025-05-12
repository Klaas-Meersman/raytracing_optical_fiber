#include "ray.hpp"
#include "fiber.hpp"
#include <cmath>
#include <numbers>
#include <string>


Ray::Ray()
    : start(0, 0), end(0, 0), angleOfDeparture(0), fiber(Fiber(1, 1)), direction(Direction::UP), endHitFiber(false) {
}

//constructor def
Ray::Ray(Coordinate start, double_t angleOfDeparture,const Fiber &fiber)
    : start(start), angleOfDeparture(angleOfDeparture),fiber(fiber) {
        //this->propagateRay();
    if (angleOfDeparture > 0 && angleOfDeparture < 3.1415 / 2) {
        direction = Direction::UP;
        this->end.y = fiber.getTopY();
        this->end.x = this->start.x + (fiber.getTopY() - this->start.y)/std::tan(this->angleOfDeparture);
    } else if (angleOfDeparture > 3 * 3.1415 / 4 && angleOfDeparture < 2 * 3.1415) {
        direction = Direction::DOWN;
        this->end.y = fiber.getBottomY();
        this->end.x = this->start.x + (fiber.getBottomY() - this->start.y)/std::tan(this->angleOfDeparture); 
    } else if (angleOfDeparture > 3.1415 / 2 && angleOfDeparture < 3 * 3.1415 / 4) {
        throw std::invalid_argument("This is an unexpected direction as all rays should go from left to right. Angle of departure is between pi/2 and 3pi/4: " + std::to_string(angleOfDeparture));
    } else {
        throw std::invalid_argument("This is an unexpected direction as all rays should go from left to right. Angle of departure is: " + std::to_string(angleOfDeparture/3.1415));
    }
    if(this->end.x > fiber.getLength()){
        this->endHitFiber = true;
        this->end.x = fiber.getLength();
        this->end.y = std::tan(this->angleOfDeparture) * (fiber.getLength() - this->start.x) + this->start.y;
    }
}

//copy constructor
Ray::Ray(const Ray& other)
    : start(other.start), end(other.end), angleOfDeparture(other.angleOfDeparture),
     fiber(other.fiber),direction(other.direction), endHitFiber(other.endHitFiber) {
}

Ray& Ray::operator=(const Ray& other) {
    if (this != &other) {
        start = other.start;
        end = other.end;
        angleOfDeparture = other.angleOfDeparture;
        direction = other.direction;
        endHitFiber = other.endHitFiber;
        // No need to assign fiber, as it is a reference
    }
    return *this;
}

Ray::~Ray(){
}

//we can't really use this. would be to many calculations that are not necesary, go to show if it works
std::vector<Coordinate> Ray::generateStraightPath(double dx){
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


Ray Ray::generateBounceRay(const Fiber& fiber){
    Coordinate start = Coordinate(this->end.x,this->end.y);
    double_t angleOfDeparture =  3.1415*2 - this->angleOfDeparture;
    Ray bouncedRay = Ray(start, angleOfDeparture, fiber);
    return bouncedRay;
}

Ray Ray::propagateRay(){
    //printf("Start: %f,%f\n", this->start.x, this->start.y);
    //printf("End: %f,%f\n", this->end.x, this->end.y);
    this->start.x = this->end.x;
    this->start.y = this->end.y;
    this->angleOfDeparture = 3.1415*2 - this->angleOfDeparture;
    if (angleOfDeparture > 0 && angleOfDeparture < 3.1415 / 2) { //or just check direction and switch direction
        this->direction = Direction::UP;
        this->end.y = fiber.getTopY();
        this->end.x = this->start.x + (fiber.getTopY() - this->start.y)/std::tan(this->angleOfDeparture);
    } else if (angleOfDeparture > 3 * 3.1415 / 4 && angleOfDeparture < 2 * 3.1415) {
        this->direction = Direction::DOWN;
        this->end.y = fiber.getBottomY();
        this->end.x = this->start.x + (fiber.getBottomY() - this->start.y)/std::tan(this->angleOfDeparture); 
    } else if (angleOfDeparture > 3.1415 / 2 && angleOfDeparture < 3 * 3.1415 / 4) {
        throw std::invalid_argument("This is an unexpected direction as all rays should go from left to right. Angle of departure is between pi/2 and 3pi/4: " + std::to_string(angleOfDeparture));
    } else {
        throw std::invalid_argument("This is an unexpected direction as all rays should go from left to right. Angle of departure is: " + std::to_string(angleOfDeparture/3.1415));
    }
    if(this->end.x > fiber.getLength()){
        this->endHitFiber = true;
        this->end.x = fiber.getLength();
        this->end.y = std::tan(this->angleOfDeparture) * (fiber.getLength() - this->start.x) + this->start.y;
    }
    
    return *this;
}