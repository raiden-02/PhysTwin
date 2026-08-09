#include "phystwin/simulator.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace phystwin {

Trajectory Simulator::run(const Parameters& theta,
                          const Environment& env,
                          const std::vector<double>& times) const {
    if (!(env.dt > 0.0) || !std::isfinite(env.dt)) {
        throw std::invalid_argument("simulation dt must be finite and > 0");
    }
    if (env.y0 > env.y_ground) {
        throw std::invalid_argument("initial y must not be below the ground line");
    }
    if (!std::isfinite(theta.vx0) || !std::isfinite(theta.vy0) ||
        !std::isfinite(theta.g) || !std::isfinite(theta.e)) {
        throw std::invalid_argument("simulation parameters must be finite");
    }
    if (theta.g < 0.0) {
        throw std::invalid_argument("gravity scale must be >= 0");
    }
    if (theta.e < 0.0 || theta.e > 1.0) {
        throw std::invalid_argument("restitution must be in [0, 1]");
    }

    double previous_time = 0.0;
    for (const double time : times) {
        if (!std::isfinite(time) || time < previous_time) {
            throw std::invalid_argument("sample times must be finite and nondecreasing");
        }
        previous_time = time;
    }

    Trajectory result;
    result.version = 1;
    result.fps = 1.0 / env.dt;
    result.observations.reserve(times.size());

    double x = env.x0;
    double y = env.y0;
    double vx = theta.vx0;
    double vy = theta.vy0;
    double simulation_time = 0.0;

    const auto integrate = [&](double step, double& px, double& py, double& pvy) {
        pvy += theta.g * step;
        px += vx * step;
        py += pvy * step;
        if (py >= env.y_ground && pvy > 0.0) {
            py = env.y_ground;
            pvy = -theta.e * pvy;
        }
    };

    for (std::size_t i = 0; i < times.size(); ++i) {
        const double target = times[i];

        // Use fixed env.dt steps. A final partial step only handles timestamps
        // that are not exact frame multiples because of decimal JSON rounding.
        while (simulation_time + env.dt <= target + 1e-12) {
            integrate(env.dt, x, y, vy);
            simulation_time += env.dt;
        }
        const double remainder = target - simulation_time;
        if (remainder > 1e-12) {
            integrate(remainder, x, y, vy);
            simulation_time = target;
        }

        result.observations.push_back(
            {.frame = static_cast<int>(std::llround(target / env.dt)),
             .t = target,
             .x = x,
             .y = std::min(y, env.y_ground)});
    }
    return result;
}

}  // namespace phystwin
