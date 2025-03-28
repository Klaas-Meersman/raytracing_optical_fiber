#include <iostream>
#include <vector>
#include "coordinate.hpp"


class Ray
{


private:
    Co start;

    //we assume the angle between the starting point 
    //and the endpoint of an arrow point to the right
    //simply: as it would be in a goniometric circle
    int angleOfDeparture;



    std::vector<int> generateStraightPath(int step);




};