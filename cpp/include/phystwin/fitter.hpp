#pragma once

#include "phystwin/physics_model.hpp"
#include "phystwin/trajectory.hpp"

#include <optional>

namespace phystwin {

struct FitOptions {
    // Ground is the object's center y at contact, in input pixels.
    // If omitted, the largest observed centroid y is used.
    std::optional<double> ground_y;
};

// Minimize unweighted image-space position residuals for
// theta = (vx0, vy0, g, e).
class Fitter {
public:
    Reconstruction fit(const Trajectory& observed,
                       const FitOptions& options = {}) const;
};

}  // namespace phystwin
