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

    enum class Direction1 {
        UP,
        DOWN,
    }; 
    enum class Direction2 {
        IN,
        OUT 
    };// toevoeging van in en out 

private:
    Coordinate start;
    Coordinate end;
    Fiber fiber; 
    double_t azimuth;
    double_t elevation;
    Direction1 direction1;
    Direction2 direction2;
    bool endHitFiber = false;

public:
    std::vector<Coordinate> generateStraightPath(double dx);
    Ray generateBounceRay(const Fiber& fiber);
    Ray propagateRay();
    inline Coordinate getStart() const { return start; }
    inline Coordinate getEnd() const { return end; }
    inline double_t getAzimuth() const { return azimuth; }
    inline double_t getElevation() const { return elevation; }
    inline Direction1 getDirection1() const { return direction1; }
    inline Direction2 getDirection2() const { return direction2; }
    inline bool getEndHitFiber() const { return endHitFiber; }
    inline void setEndHitFiber(bool endHitFiber) { this->endHitFiber = endHitFiber; }

    friend std::ostream& operator<<(std::ostream& os, const Ray& r) {
        os << "Ray(start: " << r.start << ", elevation: " 
        << r.elevation/3.1415*180 << ", azimuth: " 
        << r.azimuth/3.1415*180 << ", direction1: " 
        << (r.direction1 == Direction1::UP ? "UP" : "DOWN") << ")";
        return os;
    }
};