#include <vector>
#include "coordinate.hpp"
#include "fiber.hpp"


class Ray
{
private:
    const int maxLength;

public:
    //constructor
  Ray(Coordinate start, double_t angleOfDeparture, double maxLength);
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
    std::vector<Coordinate> generateStraightPath(double dx);
    Ray generateBounceRay(Ray incomingRay, Fiber fiber);
};