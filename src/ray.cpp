#include "ray.hpp"
#include <cmath>
#include <numbers>





//constructor def
Ray::Ray(Coordinate start, double_t angleOfDeparture, double maxLength)
    :start(start),angleOfDeparture(angleOfDeparture),maxLength(maxLength){
    if (0< angleOfDeparture < std::numbers::pi/2) {
        direction = Direction::UP;
    } else if (3*std::numbers::pi/4 < angleOfDeparture < std::numbers::pi) {
        direction = Direction::DOWN;
    } else if (std::numbers::pi/2 < angleOfDeparture < 3*std::numbers::pi/4) {
        throw std::invalid_argument("This is an unexpected direction as all rays should go from left to right. Angle of departure is between pi/2 and 3pi/4: " + std::to_string(angleOfDeparture));
    } else{
        throw std::invalid_argument("This is an unexpected direction as all rays should go from left to right. Angle of departure is: " + std::to_string(angleOfDeparture));
    }
}



Ray::~Ray(){
}

//we can't really use this. would be to many calculations that are not necesary
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


//generate start coordinate with a = b/(tan@)
Ray Ray::generateBounceRay(Ray incomingRay, Fiber fiber){
    printf("generating bounce ray\n");
    Coordinate start;
    double_t startx,starty,angleOfDeparture;
    if(incomingRay.direction == Ray::Direction::UP){
        starty = fiber.getTopY();;
        startx = incomingRay.start.x + (fiber.getTopY() - incomingRay.start.y)/std::tan(incomingRay.angleOfDeparture); //so 
        start = Coordinate(startx,starty);
        angleOfDeparture = 2 * std::number::pi*2 - incomingRay.angleOfDeparture;
        return Ray(start, angleOfDeparture, 100); //100 has to be replaced
    }else if(incomingRay.direction == Ray::Direction::DOWN){
        starty = bottomYFiber;
        //to be done, ez
    }
}







