#ifndef FIBER_HPP
#define FIBER_HPP

#include <vector>
#include "coordinate.hpp"


class Fiber
{
private:
    const double_t width, length;
    

public:
    //constructor
    Fiber(double_t width, double_t length);
    //destructor
     ~Fiber();
    //this would typically be no longer than width + length of the fiber it is in
    int maxSingleRayInFiber(double_t startX);

    //getters for top y
    inline double_t getTopY(){
        return width/2;
    }
    inline double_t getBottomY(){
        return -width/2;
    }
    inline double_t getWidth() const { return width; }
    inline double_t getLength() const { return length; }

};

#endif