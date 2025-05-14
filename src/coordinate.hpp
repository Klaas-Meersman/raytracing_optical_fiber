// coordinate.hpp
#ifndef COORDINATE_HPP
#define COORDINATE_HPP

#include <cmath>
#include <ostream>

struct Coordinate {
    double_t x = 0;
    double_t y = 0;
    double_t z = 0;

    constexpr Coordinate() = default;
    constexpr Coordinate(double_t x, double_t y, double_t z = 0) noexcept : x(x), y(y), z(z) {}

    constexpr Coordinate operator+(const Coordinate& other) const noexcept {
        return {x + other.x, y + other.y, z + other.z};
    }

    constexpr Coordinate operator-(const Coordinate& other) const noexcept {
        return {x - other.x, y - other.y, z - other.z};
    }

    [[nodiscard]] double distance(const Coordinate& other) const noexcept {
        double dx = x - other.x;
        double dy = y - other.y;
        double dz = z - other.z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
};

inline std::ostream& operator<<(std::ostream& os, const Coordinate& c) {
    os << c.x << "," << c.y << "," << c.z;
    return os;
}

#endif // COORDINATE_HPP
