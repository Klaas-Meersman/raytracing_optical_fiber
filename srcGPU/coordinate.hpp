#ifndef COORDINATE_HPP
#define COORDINATE_HPP

#include <cmath>

struct Coordinate {
    double_t x = 0;
    double_t y = 0;
    double_t z = 0;

    constexpr Coordinate() = default;
    __host__ __device__ constexpr Coordinate(double_t x, double_t y, double_t z) noexcept : x(x), y(y), z(z) {}

    __host__ __device__ constexpr Coordinate operator+(const Coordinate& other) const noexcept {
        return {x + other.x, y + other.y, z + other.z};
    }

    __host__ __device__ constexpr Coordinate operator-(const Coordinate& other) const noexcept {
        return {x - other.x, y - other.y, z - other.z};
    }

    __host__ __device__ double distance(const Coordinate& other) const noexcept {
        const double_t dx = x - other.x;
        const double_t dy = y - other.y;
        const double_t dz = z - other.z;
        return sqrt(dx*dx + dy*dy + dz*dz); // sqrt is available on device
    }
};

#ifndef __CUDA_ARCH__
#include <ostream>
inline std::ostream& operator<<(std::ostream& os, const Coordinate& c) {
    os << c.x << "," << c.y << "," << c.z;
    return os;
}
#endif

#endif // COORDINATE_HPP
