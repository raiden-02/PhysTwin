#include "phystwin/pendulum.hpp"

#include "phystwin/metrics.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace phystwin {
namespace {

constexpr double kPi = 3.14159265358979323846;
using Candidate = std::array<double, 3>;  // omega0, lambda, damping

struct Geometry {
    ReferencePoint pivot;
    double radius = 0.0;
    double radial_mad = 0.0;
    std::vector<double> angles;
};

struct Bounds {
    double omega_limit = 0.0;
    double lambda_min = 0.05;
    double lambda_max = 0.0;
    double damping_max = 0.0;
};

double median(std::vector<double> values) {
    if (values.empty()) {
        throw std::invalid_argument("cannot take median of an empty set");
    }
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    const double upper = values[middle];
    if (values.size() % 2 != 0) {
        return upper;
    }
    std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
    return 0.5 * (values[middle - 1] + upper);
}

double clamp_unit(const double value) {
    return std::clamp(value, 0.0, 1.0);
}

void validate_observed(const Trajectory& observed) {
    if (observed.model != DynamicsModel::pendulum) {
        throw std::invalid_argument("pendulum fitter requires model=pendulum");
    }
    if (!observed.pivot.has_value() || !std::isfinite(observed.pivot->x) ||
        !std::isfinite(observed.pivot->y)) {
        throw std::invalid_argument("pendulum fitting requires a finite pivot");
    }
    if (!(observed.fps > 0.0) || !std::isfinite(observed.fps)) {
        throw std::invalid_argument("trajectory fps must be finite and > 0");
    }
    if (observed.observations.size() < 12) {
        throw std::invalid_argument(
            "pendulum fitting requires at least 12 observations");
    }
    for (std::size_t i = 0; i < observed.observations.size(); ++i) {
        const auto& point = observed.observations[i];
        if (!std::isfinite(point.t) || !std::isfinite(point.x) ||
            !std::isfinite(point.y)) {
            throw std::invalid_argument(
                "observations must contain finite t, x, and y");
        }
        if (i > 0 && point.t <= observed.observations[i - 1].t) {
            throw std::invalid_argument(
                "observation times must be strictly increasing");
        }
    }
    const double duration =
        observed.observations.back().t - observed.observations.front().t;
    if (duration < 0.25) {
        throw std::invalid_argument(
            "pendulum trajectory must span at least 0.25 seconds");
    }
}

Geometry geometry_for(const Trajectory& observed, const ReferencePoint pivot) {
    std::vector<double> distances;
    distances.reserve(observed.observations.size());
    for (const auto& point : observed.observations) {
        distances.push_back(std::hypot(point.x - pivot.x, point.y - pivot.y));
    }
    const double radius = median(distances);
    std::vector<double> deviations;
    deviations.reserve(distances.size());
    for (const double distance : distances) {
        deviations.push_back(std::abs(distance - radius));
    }

    std::vector<double> angles;
    angles.reserve(observed.observations.size());
    for (const auto& point : observed.observations) {
        double angle = std::atan2(point.x - pivot.x, point.y - pivot.y);
        if (!angles.empty()) {
            while (angle - angles.back() > kPi) {
                angle -= 2.0 * kPi;
            }
            while (angle - angles.back() < -kPi) {
                angle += 2.0 * kPi;
            }
        }
        angles.push_back(angle);
    }
    return {
        .pivot = pivot,
        .radius = radius,
        .radial_mad = median(deviations),
        .angles = std::move(angles),
    };
}

Geometry refine_geometry(const Trajectory& observed) {
    const ReferencePoint clicked = *observed.pivot;
    Geometry best = geometry_for(observed, clicked);
    if (best.radius < 5.0) {
        throw std::invalid_argument(
            "pendulum radius is below 5 px; pivot and target are too close");
    }

    const double bound = std::clamp(0.08 * best.radius, 3.0, 30.0);
    const auto score = [&](const Geometry& geometry) {
        const double adjustment =
            std::hypot(geometry.pivot.x - clicked.x, geometry.pivot.y - clicked.y);
        return geometry.radial_mad + 0.08 * adjustment;
    };

    double best_score = score(best);
    double step = 0.5 * bound;
    while (step >= 0.125) {
        bool improved = false;
        const std::array<std::array<double, 2>, 8> directions{{
            {-step, 0.0},
            {step, 0.0},
            {0.0, -step},
            {0.0, step},
            {-step, -step},
            {-step, step},
            {step, -step},
            {step, step},
        }};
        for (const auto& direction : directions) {
            const double dx = direction[0];
            const double dy = direction[1];
            const ReferencePoint candidate{
                .x = best.pivot.x + dx,
                .y = best.pivot.y + dy,
            };
            if (std::abs(candidate.x - clicked.x) > bound ||
                std::abs(candidate.y - clicked.y) > bound) {
                continue;
            }
            Geometry trial = geometry_for(observed, candidate);
            const double trial_score = score(trial);
            if (trial_score + 1e-9 < best_score) {
                best = std::move(trial);
                best_score = trial_score;
                improved = true;
            }
        }
        if (!improved) {
            step *= 0.5;
        }
    }
    return best;
}

PendulumParameters decode(const Candidate& candidate, const Bounds& bounds) {
    return {
        .omega0 = (2.0 * candidate[0] - 1.0) * bounds.omega_limit,
        .lambda =
            bounds.lambda_min +
            candidate[1] * (bounds.lambda_max - bounds.lambda_min),
        .damping = candidate[2] * bounds.damping_max,
    };
}

Candidate encode(const PendulumParameters& parameters, const Bounds& bounds) {
    return {
        clamp_unit((parameters.omega0 / bounds.omega_limit + 1.0) * 0.5),
        clamp_unit((parameters.lambda - bounds.lambda_min) /
                   (bounds.lambda_max - bounds.lambda_min)),
        clamp_unit(parameters.damping / bounds.damping_max),
    };
}

double huber(const double residual, const double delta) {
    const double magnitude = std::abs(residual);
    if (magnitude <= delta) {
        return 0.5 * residual * residual;
    }
    return delta * (magnitude - 0.5 * delta);
}

std::optional<double> estimate_period(const std::vector<double>& times,
                                      const std::vector<double>& angles) {
    std::vector<double> upward_crossings;
    std::vector<double> downward_crossings;
    for (std::size_t i = 1; i < angles.size(); ++i) {
        const double previous = angles[i - 1];
        const double current = angles[i];
        if ((previous < 0.0 && current >= 0.0) ||
            (previous > 0.0 && current <= 0.0)) {
            const double fraction =
                std::abs(previous) / (std::abs(previous) + std::abs(current));
            const double crossing =
                times[i - 1] + fraction * (times[i] - times[i - 1]);
            if (previous < 0.0) {
                upward_crossings.push_back(crossing);
            } else {
                downward_crossings.push_back(crossing);
            }
        }
    }

    std::vector<double> periods;
    const auto append_periods = [&](const std::vector<double>& crossings) {
        for (std::size_t i = 1; i < crossings.size(); ++i) {
            periods.push_back(crossings[i] - crossings[i - 1]);
        }
    };
    append_periods(upward_crossings);
    append_periods(downward_crossings);
    if (periods.empty()) {
        return std::nullopt;
    }
    return median(std::move(periods));
}

std::string classify(const double normalized_rmse) {
    if (normalized_rmse <= 0.05) {
        return "good";
    }
    if (normalized_rmse <= 0.15) {
        return "fair";
    }
    return "poor";
}

}  // namespace

PendulumReconstruction PendulumFitter::fit(const Trajectory& observed) const {
    validate_observed(observed);
    const auto start = std::chrono::steady_clock::now();
    const Geometry geometry = refine_geometry(observed);
    if (geometry.radial_mad > std::max(3.0, 0.15 * geometry.radius)) {
        throw std::invalid_argument(
            "unusable pivot relationship: target distance from pivot varies too much");
    }

    const auto [min_angle, max_angle] =
        std::minmax_element(geometry.angles.begin(), geometry.angles.end());
    const double angular_span = *max_angle - *min_angle;
    if (angular_span < 0.12 || geometry.radius * angular_span < 8.0) {
        throw std::invalid_argument(
            "pendulum track is nearly stationary or has insufficient angular span");
    }

    const double t0 = observed.observations.front().t;
    std::vector<double> times;
    times.reserve(observed.observations.size());
    std::vector<double> angular_speeds;
    angular_speeds.reserve(observed.observations.size() - 1);
    for (std::size_t i = 0; i < observed.observations.size(); ++i) {
        times.push_back(observed.observations[i].t - t0);
        if (i > 0) {
            const double dt =
                observed.observations[i].t - observed.observations[i - 1].t;
            angular_speeds.push_back(
                (geometry.angles[i] - geometry.angles[i - 1]) / dt);
        }
    }

    std::vector<double> absolute_speeds;
    absolute_speeds.reserve(angular_speeds.size());
    for (const double speed : angular_speeds) {
        absolute_speeds.push_back(std::abs(speed));
    }
    const double speed_scale = median(absolute_speeds);
    if (speed_scale < 0.02) {
        throw std::invalid_argument(
            "pendulum track has no measurable angular motion");
    }
    const double duration = times.back();
    const std::optional<double> observed_period =
        estimate_period(times, geometry.angles);
    const double frequency_lambda =
        observed_period.has_value()
            ? std::pow(2.0 * kPi / *observed_period, 2.0)
            : 0.0;
    const Bounds bounds{
        .omega_limit = std::max(2.0, 6.0 * speed_scale),
        .lambda_min = 0.05,
        .lambda_max =
            std::min(400.0,
                     std::max(25.0,
                              std::max(4.0 * frequency_lambda,
                                       12.0 * speed_scale * speed_scale +
                                           16.0 / (duration * duration)))),
        .damping_max = std::min(20.0, std::max(2.0, 10.0 / duration)),
    };

    const std::size_t initial_count =
        std::min<std::size_t>(angular_speeds.size(), 10);
    std::vector<double> initial_speeds(angular_speeds.begin(),
                                       angular_speeds.begin() + initial_count);
    const double initial_omega = median(initial_speeds);
    const PendulumEnvironment environment{
        .pivot_x = geometry.pivot.x,
        .pivot_y = geometry.pivot.y,
        .radius = geometry.radius,
        .theta0 = geometry.angles.front(),
        .integration_step = std::min(1.0 / 240.0, 1.0 / observed.fps / 4.0),
    };
    const double huber_delta = std::max(2.0, 0.02 * geometry.radius);
    const PendulumSimulator simulator;
    const auto objective = [&](const Candidate& candidate) {
        const std::vector<double> simulated =
            simulator.run_angles(decode(candidate, bounds), environment, times);
        double cost = 0.0;
        for (std::size_t i = 0; i < simulated.size(); ++i) {
            const double tangential_error =
                geometry.radius * (geometry.angles[i] - simulated[i]);
            cost += huber(tangential_error, huber_delta);
        }
        return cost / static_cast<double>(simulated.size());
    };

    constexpr std::size_t population_size = 64;
    constexpr int generations = 220;
    constexpr double differential_weight = 0.75;
    constexpr double crossover_rate = 0.9;
    std::mt19937 generator(0x50454e44U);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_int_distribution<std::size_t> member(0, population_size - 1);

    std::vector<Candidate> population(population_size);
    std::vector<double> scores(population_size);
    population[0] =
        encode({initial_omega,
                std::clamp(frequency_lambda > 0.0 ? frequency_lambda : 9.81,
                           bounds.lambda_min,
                           bounds.lambda_max),
                0.1},
               bounds);
    population[1] =
        encode({initial_omega, std::clamp(4.0, bounds.lambda_min, bounds.lambda_max), 0.05},
               bounds);
    population[2] =
        encode({initial_omega, std::clamp(25.0, bounds.lambda_min, bounds.lambda_max), 0.25},
               bounds);
    for (std::size_t i = 3; i < population_size; ++i) {
        population[i] = {unit(generator), unit(generator), unit(generator)};
    }
    for (std::size_t i = 0; i < population_size; ++i) {
        scores[i] = objective(population[i]);
    }

    for (int generation = 0; generation < generations; ++generation) {
        for (std::size_t i = 0; i < population_size; ++i) {
            std::size_t a = member(generator);
            std::size_t b = member(generator);
            std::size_t c = member(generator);
            while (a == i) {
                a = member(generator);
            }
            while (b == i || b == a) {
                b = member(generator);
            }
            while (c == i || c == a || c == b) {
                c = member(generator);
            }
            Candidate trial = population[i];
            const std::size_t forced =
                static_cast<std::size_t>(unit(generator) * trial.size()) %
                trial.size();
            for (std::size_t dimension = 0; dimension < trial.size(); ++dimension) {
                if (unit(generator) <= crossover_rate || dimension == forced) {
                    trial[dimension] =
                        clamp_unit(population[a][dimension] +
                                   differential_weight *
                                       (population[b][dimension] -
                                        population[c][dimension]));
                }
            }
            const double score = objective(trial);
            if (score < scores[i]) {
                population[i] = trial;
                scores[i] = score;
            }
        }
    }

    const std::size_t best_index = static_cast<std::size_t>(
        std::min_element(scores.begin(), scores.end()) - scores.begin());
    Candidate best = population[best_index];
    double best_score = scores[best_index];
    double step = 0.04;
    int refinement_iterations = 0;
    while (step > 1e-7 && refinement_iterations < 192) {
        Candidate next = best;
        double next_score = best_score;
        for (std::size_t dimension = 0; dimension < best.size(); ++dimension) {
            for (const double direction : {-1.0, 1.0}) {
                Candidate trial = best;
                trial[dimension] =
                    clamp_unit(trial[dimension] + direction * step);
                const double score = objective(trial);
                if (score < next_score) {
                    next = trial;
                    next_score = score;
                }
            }
        }
        if (next_score < best_score) {
            best = next;
            best_score = next_score;
        } else {
            step *= 0.5;
        }
        ++refinement_iterations;
    }

    PendulumReconstruction reconstruction;
    reconstruction.parameters = decode(best, bounds);
    reconstruction.environment = environment;
    reconstruction.simulated =
        simulator.run(reconstruction.parameters, environment, times);
    reconstruction.simulated.fps = observed.fps;
    reconstruction.simulated.frame_width = observed.frame_width;
    reconstruction.simulated.frame_height = observed.frame_height;
    for (std::size_t i = 0; i < reconstruction.simulated.observations.size(); ++i) {
        reconstruction.simulated.observations[i].frame =
            observed.observations[i].frame;
        reconstruction.simulated.observations[i].t = observed.observations[i].t;
    }
    reconstruction.rmse = rmse(observed, reconstruction.simulated);
    reconstruction.mae = mae(observed, reconstruction.simulated);
    double sum_x = 0.0;
    double sum_y = 0.0;
    for (std::size_t i = 0; i < observed.observations.size(); ++i) {
        const double dx =
            observed.observations[i].x -
            reconstruction.simulated.observations[i].x;
        const double dy =
            observed.observations[i].y -
            reconstruction.simulated.observations[i].y;
        sum_x += dx * dx;
        sum_y += dy * dy;
    }
    const double count = static_cast<double>(observed.observations.size());
    reconstruction.rmse_x = std::sqrt(sum_x / count);
    reconstruction.rmse_y = std::sqrt(sum_y / count);
    reconstruction.normalized_rmse =
        reconstruction.rmse / geometry.radius;
    reconstruction.robust_cost = best_score;
    reconstruction.radial_mad = geometry.radial_mad;
    reconstruction.angular_span = angular_span;
    reconstruction.pivot_adjustment =
        std::hypot(geometry.pivot.x - observed.pivot->x,
                   geometry.pivot.y - observed.pivot->y);
    reconstruction.quality = classify(reconstruction.normalized_rmse);
    if (geometry.radial_mad > 0.08 * geometry.radius &&
        reconstruction.quality == "good") {
        reconstruction.quality = "fair";
    }
    reconstruction.n = static_cast<int>(observed.observations.size());
    reconstruction.search_generations = generations;
    reconstruction.refinement_iterations = refinement_iterations;
    reconstruction.fit_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    return reconstruction;
}

}  // namespace phystwin
