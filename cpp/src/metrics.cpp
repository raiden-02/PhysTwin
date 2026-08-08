#include "phystwin/metrics.hpp"

#include <cmath>
#include <stdexcept>

namespace phystwin {
namespace {

void require_aligned(const Trajectory& observed, const Trajectory& simulated) {
    if (observed.observations.size() != simulated.observations.size()) {
        throw std::runtime_error("observed and simulated trajectories have different lengths");
    }
}

}  // namespace

double rmse(const Trajectory& observed, const Trajectory& simulated) {
    require_aligned(observed, simulated);
    if (observed.observations.empty()) {
        throw std::runtime_error("cannot compute RMSE on an empty trajectory");
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < observed.observations.size(); ++i) {
        const double dx = observed.observations[i].x - simulated.observations[i].x;
        const double dy = observed.observations[i].y - simulated.observations[i].y;
        sum += dx * dx + dy * dy;
    }
    return std::sqrt(sum / static_cast<double>(observed.observations.size()));
}

double mae(const Trajectory& observed, const Trajectory& simulated) {
    require_aligned(observed, simulated);
    if (observed.observations.empty()) {
        throw std::runtime_error("cannot compute MAE on an empty trajectory");
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < observed.observations.size(); ++i) {
        const double dx = observed.observations[i].x - simulated.observations[i].x;
        const double dy = observed.observations[i].y - simulated.observations[i].y;
        sum += std::hypot(dx, dy);
    }
    return sum / static_cast<double>(observed.observations.size());
}

}  // namespace phystwin
