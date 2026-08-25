#pragma once

#include "phystwin/physics_model.hpp"
#include "phystwin/trajectory.hpp"

namespace phystwin {

// Minimize unweighted image-space position residuals for
// theta = (vx0, vy0, g, e). Ground is estimated as max observed y.
class Fitter {
public:
    Reconstruction fit(const Trajectory& observed) const;
};

}  // namespace phystwin
