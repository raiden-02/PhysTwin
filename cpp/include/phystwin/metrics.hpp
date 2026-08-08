#pragma once

#include "phystwin/trajectory.hpp"

namespace phystwin {

// Pair observations by index after the simulator has been sampled at observation times.
double rmse(const Trajectory& observed, const Trajectory& simulated);
double mae(const Trajectory& observed, const Trajectory& simulated);

}  // namespace phystwin
