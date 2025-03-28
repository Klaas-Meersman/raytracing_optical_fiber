#include "coordinate.hpp"



//constructor def
Coordinate::Coordinate(int x, int y):x(x),y(y){
    //this->x = x;
    //this->y = y;
}

//copy constructor
Coordinate::Coordinate(const Coordinate& other){}

Coordinate::~Coordinate(){
}



Coordinate& Coordinate::getCo() {
    return *this;
}

