#ifndef COORDINATE_HPP
#define COORDINATE_HPP

#include <cmath> // For sqrt in the distance function
#include <ostream>

struct Coordinate {
    // Public members for simple data
    double_t x = 0;
    double_t y = 0;

    // Constructors
    constexpr Coordinate() = default;
    constexpr Coordinate(double_t x, double_t y) noexcept : x(x), y(y) {}

    // Optional: Useful operators defined inline
    constexpr Coordinate operator+(const Coordinate& other) const noexcept {
        return {x + other.x, y + other.y};
    }

    constexpr Coordinate operator-(const Coordinate& other) const noexcept {
        return {x - other.x, y - other.y};
    }

    // Optional: Method declared here but defined in cpp file
    // (only if it's complex and would clutter the header)
    [[nodiscard]] double distance(const Coordinate& other) const noexcept;
};

// Small inline functions can be defined directly in the header
inline double Coordinate::distance(const Coordinate& other) const noexcept {
    const double_t dx = x - other.x;
    const double_t dy = y - other.y;
    return std::sqrt(dx*dx + dy*dy);
}

//tostring
inline std::ostream& operator<<(std::ostream& os, const Coordinate& c) {
    os << c.x << "," << c.y;
    return os;
}

#endif // COORDINATE_HPP