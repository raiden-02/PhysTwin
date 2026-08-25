#pragma once

#include "phystwin/physics_model.hpp"
#include "phystwin/trajectory.hpp"

#include <stdexcept>
#include <string>

namespace phystwin {

class NotImplemented : public std::runtime_error {
public:
    explicit NotImplemented(const std::string& what) : std::runtime_error(what) {}
};

// Nonlinear fit of θ = (vx0, vy0, g, e). Implementation is Checkpoint 1.
class Fitter {
public:
    Reconstruction fit(const Trajectory& observed) const;
};

}  // namespace phystwin
