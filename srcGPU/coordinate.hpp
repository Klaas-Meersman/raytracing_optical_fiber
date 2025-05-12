#ifndef COORDINATE_HPP
#define COORDINATE_HPP

#include <cmath> // For sqrt in the distance function

struct Coordinate {
    double_t x = 0;
    double_t y = 0;

    // Constructors
    __host__ __device__ constexpr Coordinate() = default;
    __host__ __device__ constexpr Coordinate(double_t x, double_t y) noexcept : x(x), y(y) {}

    // Operators
    __host__ __device__ constexpr Coordinate operator+(const Coordinate& other) const noexcept {
        return {x + other.x, y + other.y};
    }

    __host__ __device__ constexpr Coordinate operator-(const Coordinate& other) const noexcept {
        return {x - other.x, y - other.y};
    }

    // Distance function
    __host__ __device__ double distance(const Coordinate& other) const noexcept {
        const double_t dx = x - other.x;
        const double_t dy = y - other.y;
        return sqrt(dx*dx + dy*dy); // sqrt is available on device
    }
};

// Host-only: ostream operator
#ifndef __CUDA_ARCH__
#include <ostream>
inline std::ostream& operator<<(std::ostream& os, const Coordinate& c) {
    os << c.x << "," << c.y;
    return os;
}
#endif

#endif // COORDINATE_HPP
