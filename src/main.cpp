#include <cstdio>

#include "coordinate.hpp"
#include "ray.hpp"


int main(){

    int a = 2;
    int b = 3;

    Coordinate *co = new Coordinate(a,b);

    Ray *r = new Ray(*co, 0, 10);
    delete co;
    delete r;
    printf("hello world\n");


    return 0;
}
