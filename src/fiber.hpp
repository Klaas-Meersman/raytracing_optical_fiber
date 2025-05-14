// fiber.hpp
#ifndef FIBER_HPP
#define FIBER_HPP

#include <vector>
#include "coordinate.hpp"

class Fiber {
private:
    const double_t width;
    const double_t height;
    const double_t length;

public:
    Fiber(double_t width, double_t height, double_t length);
    Fiber(const Fiber& other);
    Fiber& operator=(const Fiber& other);
    ~Fiber();

    [[nodiscard]] double_t getLength() const { return length; }
    [[nodiscard]] double_t getTopY() const { return height / 2.0; }
    [[nodiscard]] double_t getBottomY() const { return -height / 2.0; }
    [[nodiscard]] double_t getTopZ() const { return width / 2.0; }
    [[nodiscard]] double_t getBottomZ() const { return -width / 2.0; }
};

#endif // FIBER_HPP
