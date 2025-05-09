#ifndef FIBER_HPP
#define FIBER_HPP

#include <vector>
#include "coordinate.hpp"


class Fiber
{
private:
    const double_t width, length;
public:
    Fiber(double_t width, double_t length);
    Fiber(const Fiber& other);
    Fiber& operator=(const Fiber& other);
    ~Fiber();

    inline double_t getTopY() const{
        return width/2;
    }
    inline double_t getBottomY()const {
        return -width/2;
    }
    inline double_t getWidth() const { return width; }
    inline double_t getLength() const { return length; }

};

#endif