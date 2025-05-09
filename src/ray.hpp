#include <vector>
#include "coordinate.hpp"
#include "fiber.hpp"


class Ray
{
public:
    Ray();
    Ray(Coordinate start, double_t angleOfDeparture, const Fiber& fiber);
    Ray(const Ray& other);
    Ray& operator=(const Ray& other);
    ~Ray();

    enum class Direction {
        UP,
        DOWN
    };

private:
    Coordinate start;
    Coordinate end;
    const Fiber& fiber;
    double_t angleOfDeparture;
    Direction direction;
    bool endHitFiber = false;

public:
    std::vector<Coordinate> generateStraightPath(double dx);
    Ray generateBounceRay(const Fiber& fiber);
    inline Coordinate getStart() const { return start; }
    inline double_t getAngleOfDeparture() const { return angleOfDeparture; }
    inline Direction getDirection() const { return direction; }
    inline bool getEndHitFiber() const { return endHitFiber; }
    inline void setEndHitFiber(bool endHitFiber) { this->endHitFiber = endHitFiber; }

    friend std::ostream& operator<<(std::ostream& os, const Ray& r) {
        os << "Ray(start: " << r.start << ", angleOfDeparture: " 
        << r.angleOfDeparture/std::numbers::pi*180 << ", direction: " 
        << (r.direction == Direction::UP ? "UP" : "DOWN") << ")";
        return os;
    }
};