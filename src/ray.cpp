#include "ray.hpp"
#include "fiber.hpp"
#include <cmath>
#include <numbers>
#include <string>


Ray::Ray()
    : start(0, 0), end(0, 0), azimuth(0), elevation(0), fiber(Fiber(1, 1,1)), endHitFiber(false) {
}

Ray::Ray(const Ray& other)
    : start(other.start), end(other.end), azimuth(other.azimuth), elevation(other.elevation),
      fiber(other.fiber), endHitFiber(other.endHitFiber) {
}

Ray& Ray::operator=(const Ray& other) {
    if (this != &other) {
        start = other.start;
        end = other.end;
        azimuth = other.azimuth;
        elevation = other.elevation;
        fiber = other.fiber; 
        endHitFiber = other.endHitFiber;
    }
    return *this;
}


Ray::~Ray(){
}

Ray::Ray(Coordinate start, double_t azimuth, double_t elevation, const Fiber &fiber)
    : start(start), azimuth(azimuth), elevation(elevation), fiber(fiber), endHitFiber(false) {
    
    // Richtingsvector afleiden
    double vx = std::cos(elevation) * std::cos(azimuth);
    double vy = std::sin(elevation);
    double vz = std::cos(elevation) * std::sin(azimuth);

    // "Tijd" tot elke grensvlak van de fiber
    //Tijd is de afstand tot de fiber volgens de richting van de ray
    double tx = (fiber.getLength() - start.x) / vx;
    double ty = (vy > 0) ? (fiber.getTopY() - start.y) / vy :
               (vy < 0) ? (fiber.getBottomY() - start.y) / vy : std::numeric_limits<double>::infinity();
    double tz = (vz > 0) ? (fiber.getTopZ() - start.z) / vz :
               (vz < 0) ? (fiber.getBottomZ() - start.z) / vz : std::numeric_limits<double>::infinity();

    // Neem kleinste positieve t-waarde voor botsing
    double t = std::min({tx, ty, tz});

    // Eindpunt berekenen
    end.x = start.x + vx * t;
    end.y = start.y + vy * t;
    end.z = start.z + vz * t;

    // Als we de zijkant van de fiber raken
    if (tx <= ty && tx <= tz) {
        endHitFiber = true;
    }
}


Ray Ray::propagateRay() {
    // Update startpunt naar huidig eindpunt
    start = end;

    // Richtingsvector opnieuw berekenen
    double vx = std::cos(elevation) * std::cos(azimuth);
    double vy = std::sin(elevation);
    double vz = std::cos(elevation) * std::sin(azimuth);

    // "Tijd" tot grenzen, hier opnieuw tijd is afstand tot de grenzen volgens de richtingsvector.
    double tx = (fiber.getLength() - start.x) / vx;
    double ty = (vy > 0) ? (fiber.getTopY() - start.y) / vy :
               (vy < 0) ? (fiber.getBottomY() - start.y) / vy : std::numeric_limits<double>::infinity();
    double tz = (vz > 0) ? (fiber.getTopZ() - start.z) / vz :
               (vz < 0) ? (fiber.getBottomZ() - start.z) / vz : std::numeric_limits<double>::infinity();


    double t = std::min({tx, ty, tz});

    // Eindpunt bij volgende botsing
    end.x = start.x + vx * t;
    end.y = start.y + vy * t;
    end.z = start.z + vz * t;

    // Reflectie: dusss hoeken updaten 
    if (t == ty) {
        elevation = -elevation;  
    }
    if (t == tz) {
        azimuth = 2 * 3.1415 - azimuth; 
    }

    //Kijken of we aan het einde zitten!
    if (t == tx) {
        endHitFiber = true;
    }

    return *this;
}