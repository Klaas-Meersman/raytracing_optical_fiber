#ifndef FIBER_HPP
#define FIBER_HPP

#include "coordinate.hpp"
#include "ray.hpp"

class Fiber {
public:
    Co topLeft, bottomRight;

    // Constructor
    Fiber(Co tl, Co br);

    // Controleer of een Ray met de Fiber botst
    bool checkCollision(const Ray& ray);
};

#endif // FIBER_HPP

