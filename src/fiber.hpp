#include <vector>
#include "coordinate.hpp"


class Fiber
{
private:
    const double_t topY, bottomY, width, length;

public:
    //constructor
    Fiber(double_t topY, double_t bottomY, double_t width, double_t length);
    //destructor
     ~Fiber();
    //this would typically be no longer than width + length of the fiber it is in
    inline int Fiber::maxSingleRayInFiber(){
        return width+length;
    }

    //getters for top y
    inline double_t Fiber::getTopY(){
        return topY;
    }
    inline double_t Fiber::getBottomY(){
        return bottomY;
    }
};