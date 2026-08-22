#pragma once

#include "phystwin/trajectory.hpp"

#include <string>
#include <vector>

namespace phystwin {

struct PendulumParameters {
    double omega0 = 0.0;
    double lambda = 0.0;
    double damping = 0.0;
};

struct PendulumEnvironment {
    double pivot_x = 0.0;
    double pivot_y = 0.0;
    double radius = 0.0;
    double theta0 = 0.0;
    double integration_step = 1.0 / 240.0;
};

struct PendulumReconstruction {
    PendulumParameters parameters;
    PendulumEnvironment environment;
    Trajectory simulated;
    double rmse = 0.0;
    double mae = 0.0;
    double rmse_x = 0.0;
    double rmse_y = 0.0;
    double normalized_rmse = 0.0;
    double robust_cost = 0.0;
    double radial_mad = 0.0;
    double angular_span = 0.0;
    double pivot_adjustment = 0.0;
    std::string quality;
    int n = 0;
    int search_generations = 0;
    int refinement_iterations = 0;
    double fit_seconds = 0.0;
};

class PendulumSimulator {
public:
    std::vector<double> run_angles(const PendulumParameters& parameters,
                                   const PendulumEnvironment& environment,
                                   const std::vector<double>& times) const;

    Trajectory run(const PendulumParameters& parameters,
                   const PendulumEnvironment& environment,
                   const std::vector<double>& times) const;
};

class PendulumFitter {
public:
    PendulumReconstruction fit(const Trajectory& observed) const;
};

}  // namespace phystwin
