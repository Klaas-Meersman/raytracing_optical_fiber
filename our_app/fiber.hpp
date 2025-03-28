#ifndef FIBER_HPP
#define FIBER_HPP
#include "coordinate.hpp"

#include <iostream>

class Fiber {
private:
    int length;
    int height;
    int index;
    Co start;
public:
    Fiber(int length, int height, int index, Co start);
};

#endif
