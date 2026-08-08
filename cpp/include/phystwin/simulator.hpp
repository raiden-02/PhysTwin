#pragma once

#include "phystwin/physics_model.hpp"
#include "phystwin/trajectory.hpp"

#include <vector>

namespace phystwin {

// Deterministic 2D point-mass integrator. Implementation is Checkpoint 1.
class Simulator {
public:
    Trajectory run(const Parameters& theta,
                   const Environment& env,
                   const std::vector<double>& times) const;
};

}  // namespace phystwin
