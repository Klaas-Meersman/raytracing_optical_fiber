#include "ray.hpp"
#include <fstream>
#include <iostream>

// Constructor
Ray::Ray(Co s, Co e, int d) : start(s), eind(e), degree(d) {}

// Methode om een nieuwe Ray te genereren na een botsing
Ray Ray::botsen(const Co& new_end, int new_degree) {
    return Ray(this->eind, new_end, new_degree);
}

// Methode om de Ray in een bestand te schrijven
void Ray::saveToFile(const std::string& filename) {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        file << "Ray: Start (" << start.x << ", " << start.y << ") "
             << "-> Eind (" << eind.x << ", " << eind.y << ") "
             << "Degree: " << degree << "°\n";
        file.close();
    }
}
