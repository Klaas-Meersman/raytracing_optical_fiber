#ifndef COORDINATE_HPP
#define COORDINATE_HPP



class Coordinate{
    
private:
    int x;
    int y;


public:
    //constructor
    Coordinate(int x, int y);

    //copy constructor
    Coordinate(const Coordinate& other);

    //destructor
    ~Coordinate();

    Coordinate& getCo();


};
#endif // COORDINATE_HPP
