#pragma once

#include "phystwin/physics_model.hpp"
#include "phystwin/trajectory.hpp"

namespace phystwin {

// Generate frame-aligned, noise-free observations from known parameters.
Trajectory generate_synthetic(const Parameters& theta,
                              const Environment& env,
                              int frame_count);

}  // namespace phystwin
