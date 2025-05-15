#include <vector>
#include "coordinate.hpp"
#include "fiber.hpp"


class Ray
{
public:
    Ray();
    Ray(Coordinate start, double_t azimuth,double_t elevation, const Fiber& fiber);
    Ray(const Ray& other);
    Ray& operator=(const Ray& other);
    ~Ray();

private:
    Coordinate start;
    Coordinate end;
    Fiber fiber; 
    double_t azimuth;
    double_t elevation;
    bool endHitFiber = false;

public:
    std::vector<Coordinate> generateStraightPath(double dx);
    Ray generateBounceRay(const Fiber& fiber);
    Ray propagateRay();
    inline Coordinate getStart() const { return start; }
    inline Coordinate getEnd() const { return end; }
    inline double_t getAzimuth() const { return azimuth; }
    inline double_t getElevation() const { return elevation; }
    inline bool getEndHitFiber() const { return endHitFiber; }
    inline void setEndHitFiber(bool endHitFiber) { this->endHitFiber = endHitFiber; }

    friend std::ostream& operator<<(std::ostream& os, const Ray& r) {
        os << "Ray(start: " << r.start << ", elevation: " 
        << r.elevation/3.1415*180 << ", azimuth: " 
        << r.azimuth/3.1415*180 << ", direction1: ";
        return os;
    }
};