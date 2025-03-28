#include <vector>
#include "coordinate.hpp"


class Ray
{

public:
    //constructor
  Ray(Coordinate start, double angleOfDeparture, int maxLength);
  //destructor
  ~Ray();

private:
    Coordinate start;

    //we assume the angle between the starting point 
    //and the endpoint of an arrow point to the right
    //simply: as it would be in a goniometric circle
    const double angleOfDeparture;

    //this would typically be no longer than width + length of the fiber it is in
    const int maxLength; 

    std::vector<int> generateStraightPath(int stepHeight);

};