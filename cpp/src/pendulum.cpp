#include "phystwin/pendulum.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace phystwin {
namespace {

void validate(const PendulumParameters& parameters,
              const PendulumEnvironment& environment,
              const std::vector<double>& times) {
    if (!std::isfinite(parameters.omega0) || !std::isfinite(parameters.lambda) ||
        !std::isfinite(parameters.damping)) {
        throw std::invalid_argument("pendulum parameters must be finite");
    }
    if (parameters.lambda <= 0.0) {
        throw std::invalid_argument("pendulum lambda must be > 0");
    }
    if (parameters.damping < 0.0) {
        throw std::invalid_argument("pendulum damping must be >= 0");
    }
    if (!std::isfinite(environment.pivot_x) ||
        !std::isfinite(environment.pivot_y) ||
        !std::isfinite(environment.radius) ||
        !std::isfinite(environment.theta0) ||
        !std::isfinite(environment.integration_step)) {
        throw std::invalid_argument("pendulum environment must be finite");
    }
    if (environment.radius <= 0.0) {
        throw std::invalid_argument("pendulum radius must be > 0");
    }
    if (environment.integration_step <= 0.0) {
        throw std::invalid_argument("pendulum integration step must be > 0");
    }
    double previous = 0.0;
    for (const double time : times) {
        if (!std::isfinite(time) || time < previous) {
            throw std::invalid_argument(
                "pendulum sample times must be finite and nondecreasing");
        }
        previous = time;
    }
}

void rk4_step(const PendulumParameters& parameters,
              const double step,
              double& theta,
              double& omega) {
    const auto acceleration = [&](const double angle, const double angular_velocity) {
        return -parameters.lambda * std::sin(angle) -
               parameters.damping * angular_velocity;
    };

    const double k1_theta = omega;
    const double k1_omega = acceleration(theta, omega);
    const double k2_theta = omega + 0.5 * step * k1_omega;
    const double k2_omega =
        acceleration(theta + 0.5 * step * k1_theta,
                     omega + 0.5 * step * k1_omega);
    const double k3_theta = omega + 0.5 * step * k2_omega;
    const double k3_omega =
        acceleration(theta + 0.5 * step * k2_theta,
                     omega + 0.5 * step * k2_omega);
    const double k4_theta = omega + step * k3_omega;
    const double k4_omega =
        acceleration(theta + step * k3_theta, omega + step * k3_omega);

    theta += step * (k1_theta + 2.0 * k2_theta + 2.0 * k3_theta + k4_theta) /
             6.0;
    omega += step * (k1_omega + 2.0 * k2_omega + 2.0 * k3_omega + k4_omega) /
             6.0;
}

}  // namespace

std::vector<double> PendulumSimulator::run_angles(
    const PendulumParameters& parameters,
    const PendulumEnvironment& environment,
    const std::vector<double>& times) const {
    validate(parameters, environment, times);

    std::vector<double> angles;
    angles.reserve(times.size());
    double theta = environment.theta0;
    double omega = parameters.omega0;
    double simulation_time = 0.0;

    for (const double target : times) {
        while (simulation_time + 1e-12 < target) {
            const double step =
                std::min(environment.integration_step, target - simulation_time);
            rk4_step(parameters, step, theta, omega);
            simulation_time += step;
        }
        angles.push_back(theta);
    }
    return angles;
}

Trajectory PendulumSimulator::run(const PendulumParameters& parameters,
                                  const PendulumEnvironment& environment,
                                  const std::vector<double>& times) const {
    const std::vector<double> angles = run_angles(parameters, environment, times);
    Trajectory result;
    result.model = DynamicsModel::pendulum;
    result.pivot = ReferencePoint{environment.pivot_x, environment.pivot_y};
    result.fps = 1.0 / (4.0 * environment.integration_step);
    result.observations.reserve(times.size());
    for (std::size_t i = 0; i < times.size(); ++i) {
        result.observations.push_back({
            .frame = static_cast<int>(std::llround(times[i] * result.fps)),
            .t = times[i],
            .x = environment.pivot_x + environment.radius * std::sin(angles[i]),
            .y = environment.pivot_y + environment.radius * std::cos(angles[i]),
        });
    }
    return result;
}

}  // namespace phystwin
