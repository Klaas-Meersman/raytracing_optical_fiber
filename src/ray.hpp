#include <vector>
#include "coordinate.hpp"


class Ray
{
private:
    const int maxLength;

public:
    //constructor
  Ray(Coordinate start, double angleOfDeparture, double maxLength);
  //destructor
  ~Ray();

private:
    Coordinate start;

    //we assume the angle between the starting point 
    //and the endpoint of an arrow point to the right
    //simply: as it would be in a goniometric circle
    const double_t angleOfDeparture;

    //this would typically be no longer than width + length of the fiber it is in
public:
    std::vector<Coordinate> generateStraightPath(double dx);
};