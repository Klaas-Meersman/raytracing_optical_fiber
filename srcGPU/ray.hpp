#pragma once

#include "coordinate.hpp"
#include "fiber.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class Ray
{
public:
    // Mark constructors and destructor as __host__ __device__
    __host__ __device__ Ray();

    // Constructor with Fiber pointer
    __host__ __device__ Ray(Coordinate start, double_t azimuth, double_t elevation, const Fiber* fiber)
        : start(start), azimuth(azimuth), elevation(elevation), fiber(fiber), endHitFiber(false) {

        // Richtingsvector afleiden
        double vx = std::cos(elevation) * std::cos(azimuth);
        double vy = std::sin(elevation);
        double vz = std::cos(elevation) * std::sin(azimuth);

        // "Tijd" tot elke grensvlak van de fiber
        //Tijd is de afstand tot de fiber volgens de richting van de ray
        double tx = (fiber->getLength() - start.x) / vx;
        double ty = (vy > 0) ? (fiber->getTopY() - start.y) / vy :
                (vy < 0) ? (fiber->getBottomY() - start.y) / vy : fiber->getLength()*2; //just a bigger number
        double tz = (vz > 0) ? (fiber->getTopZ() - start.z) / vz :
                (vz < 0) ? (fiber->getBottomZ() - start.z) / vz : fiber->getLength()*2; //just a bigger number

        // Neem kleinste positieve t-waarde voor botsing
        double t = fmin(fmin(tx, ty), tz);

        // Eindpunt berekenen
        end.x = start.x + vx * t;
        end.y = start.y + vy * t;
        end.z = start.z + vz * t;

        // Als we de zijkant van de fiber raken
        if (tx <= ty && tx <= tz) {
            endHitFiber = true;
        }
    }

    __host__ __device__ Ray(const Ray& other): 
            start(other.start), end(other.end), azimuth(other.azimuth), elevation(other.elevation),
            fiber(other.fiber), endHitFiber(other.endHitFiber) {}


    __host__ __device__ Ray& operator=(const Ray& other) {
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

    __host__ __device__ ~Ray() {}


private:
    Coordinate start;
    Coordinate end;
    const Fiber* fiber;
    double_t azimuth;
    double_t elevation;
    bool endHitFiber = false;

public:
    __host__ __device__ inline Coordinate getStart() const { return start; }
    __host__ __device__ inline Coordinate getEnd() const { return end; }
    __host__ __device__ inline double_t getAzimuth() const { return azimuth; }
    __host__ __device__ inline double_t getElevation() const { return elevation; }
    __host__ __device__ inline bool getEndHitFiber() const { return endHitFiber; }
    __host__ __device__ inline void setEndHitFiber(bool endHitFiber) { this->endHitFiber = endHitFiber; }

    __host__ __device__ inline Ray propagateRay() {
        // Update startpunt naar huidig eindpunt
        start = end;

        // Richtingsvector opnieuw berekenen
        double vx = std::cos(elevation) * std::cos(azimuth);
        double vy = std::sin(elevation);
        double vz = std::cos(elevation) * std::sin(azimuth);

        // "Tijd" tot grenzen, hier opnieuw tijd is afstand tot de grenzen volgens de richtingsvector.
        double tx = (fiber->getLength() - start.x) / vx;
        double ty = (vy > 0) ? (fiber->getTopY() - start.y) / vy :
                (vy < 0) ? (fiber->getBottomY() - start.y) / vy : fiber->getLength()*2; //just a bigger number
        double tz = (vz > 0) ? (fiber->getTopZ() - start.z) / vz :
                (vz < 0) ? (fiber->getBottomZ() - start.z) / vz : fiber->getLength()*2; //just a bigger number


        double t = fmin(fmin(tx, ty), tz);


        // Eindpunt bij volgende botsing
        end.x = start.x + vx * t;
        end.y = start.y + vy * t;
        end.z = start.z + vz * t;

        // Reflectie: dusss hoeken updaten 
        if (t == ty) {
            elevation = -elevation;  
        }
        if (t == tz) {
            azimuth = 2 * M_PI - azimuth; 
        }

        //Kijken of we aan het einde zitten!
        if (t == tx) {
            endHitFiber = true;
        }

        return *this;
        }
};
