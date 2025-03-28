# 0 "/home/klaas/github/raytracing_optical_fiber/src/coordinate.cpp"
# 1 "/home/klaas/github/raytracing_optical_fiber/cmake-build-debug//"
# 0 "<built-in>"
# 0 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 0 "<command-line>" 2
# 1 "/home/klaas/github/raytracing_optical_fiber/src/coordinate.cpp"
# 1 "/home/klaas/github/raytracing_optical_fiber/src/coordinate.hpp" 1





class Coordinate{

private:
    int x;
    int y;


public:

    Coordinate(int x, int y);


    Coordinate(const Coordinate& other);


    ~Coordinate();

    Coordinate& getCo();


};
# 2 "/home/klaas/github/raytracing_optical_fiber/src/coordinate.cpp" 2




Coordinate::Coordinate(int x, int y):x(x),y(y){


}


Coordinate::Coordinate(const Coordinate& other){}

Coordinate::~Coordinate(){
}



Coordinate& Coordinate::getCo() {
    return *this;
}
