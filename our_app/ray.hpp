#ifndef RAY_HPP
#define RAY_HPP

#include "coordinate.hpp"
#include <vector>

class Ray {
public:
    Co start;
    Co eind;
    int degree;

    // Constructor
    Ray(Co s, Co e, int d);

    // Methode om te botsen en een nieuwe Ray te genereren
    Ray botsen(const Co& new_end, int new_degree);

    // Methode om de Ray in een bestand te schrijven
    void saveToFile(const std::string& filename);
};

#endif // RAY_HPP
