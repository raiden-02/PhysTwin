#pragma once

#include "phystwin/trajectory.hpp"

namespace phystwin {

// Fitted dynamics. Gravity is a scale in px/s^2, not 9.81 m/s^2.
struct Parameters {
    double vx0 = 0.0;
    double vy0 = 0.0;
    double g = 0.0;
    double e = 0.0;
};

// Scene quantities taken from the video, not fitted as θ.
struct Environment {
    double x0 = 0.0;
    double y0 = 0.0;
    double y_ground = 0.0;
    double dt = 1.0 / 60.0;
};

struct Reconstruction {
    Parameters parameters;
    Environment environment;
    Trajectory simulated;
    double rmse = 0.0;
    double mae = 0.0;
    int n = 0;
    int iterations = 0;
    double fit_seconds = 0.0;
};

}  // namespace phystwin
