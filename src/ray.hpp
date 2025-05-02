#include <vector>
#include "coordinate.hpp"
#include "fiber.hpp"


class Ray
{
private:
    const int maxLength;

public:
    //constructor
  Ray(Coordinate start, double_t angleOfDeparture, int maxLength);
  //destructor
  ~Ray();


  enum class Direction {
    UP,
    DOWN
  };

private:
    Coordinate start;

    //we assume the angle between the starting point 
    //and the endpoint of an arrow point to the right
    //simply: as it would be in a goniometric circle
    //angleOfDeparture is in radians
    //angleOfDeparture is the angle between the ray and the x-axis
    const double_t angleOfDeparture;
    Direction direction;

    //this would typically be no longer than width + length of the fiber it is in
public:
    std::vector<Coordinate> generateStraightPath(Fiber fiber, double dx);
    Ray generateBounceRay(Fiber fiber);
    //getters for start
    inline Coordinate getStart() const { return start; }
    inline double_t getAngleOfDeparture() const { return angleOfDeparture; }
    inline Direction getDirection() const { return direction; }
    //tostring
    friend std::ostream& operator<<(std::ostream& os, const Ray& r) {
        os << "Ray(start: " << r.start << ", angleOfDeparture: " << r.angleOfDeparture/std::numbers::pi*180 << ", direction: " << (r.direction == Direction::UP ? "UP" : "DOWN") << ")";
        return os;
    }
};