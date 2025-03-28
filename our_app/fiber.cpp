#include "fiber.hpp"

// Constructor
Fiber::Fiber(Co tl, Co br) : topLeft(tl), bottomRight(br) {}

// Controleer of de Ray binnen de Fiber valt
bool Fiber::checkCollision(const Ray& ray) {    
    return (ray.eind.x >= topLeft.x && ray.eind.x <= bottomRight.x &&
            ray.eind.y >= topLeft.y && ray.eind.y <= bottomRight.y);
}
