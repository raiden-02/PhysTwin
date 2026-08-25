#include "phystwin/fitter.hpp"

#include "phystwin/metrics.hpp"
#include "phystwin/simulator.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace phystwin {
namespace {

using Candidate = std::array<double, 3>;  // normalized vy0, g, e

struct Bounds {
    double velocity_limit = 0.0;
    double gravity_limit = 0.0;
};

double clamp_unit(double value) {
    return std::clamp(value, 0.0, 1.0);
}

Parameters decode(const Candidate& candidate, double vx0, const Bounds& bounds) {
    return {
        .vx0 = vx0,
        .vy0 = (2.0 * candidate[0] - 1.0) * bounds.velocity_limit,
        .g = candidate[1] * bounds.gravity_limit,
        .e = candidate[2],
    };
}

Candidate encode(double vy0, double g, double e, const Bounds& bounds) {
    return {
        clamp_unit((vy0 / bounds.velocity_limit + 1.0) * 0.5),
        clamp_unit(g / bounds.gravity_limit),
        clamp_unit(e),
    };
}

double estimate_vx(const Trajectory& observed, const std::vector<double>& times) {
    const double x0 = observed.observations.front().x;
    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t i = 0; i < times.size(); ++i) {
        numerator += times[i] * (observed.observations[i].x - x0);
        denominator += times[i] * times[i];
    }
    if (denominator <= 0.0) {
        throw std::invalid_argument("trajectory must span a positive duration");
    }
    return numerator / denominator;
}

std::pair<double, double> estimate_initial_vertical_motion(const Trajectory& observed,
                                                           double ground,
                                                           double dt) {
    const auto& points = observed.observations;
    double gravity_sum = 0.0;
    int gravity_count = 0;

    // For fixed frame spacing, semi-implicit Euler has
    // y[n] - 2*y[n-1] + y[n-2] = g*dt^2 before contact.
    for (std::size_t i = 2; i < points.size(); ++i) {
        const double dt0 = points[i - 1].t - points[i - 2].t;
        const double dt1 = points[i].t - points[i - 1].t;
        if (std::abs(dt0 - dt) > dt * 1e-3 || std::abs(dt1 - dt) > dt * 1e-3) {
            continue;
        }
        if (points[i - 2].y >= ground - 1e-6 ||
            points[i - 1].y >= ground - 1e-6 ||
            points[i].y >= ground - 1e-6) {
            continue;
        }
        const double estimate =
            (points[i].y - 2.0 * points[i - 1].y + points[i - 2].y) / (dt * dt);
        if (std::isfinite(estimate) && estimate >= 0.0) {
            gravity_sum += estimate;
            ++gravity_count;
        }
    }

    const double g = gravity_count > 0 ? gravity_sum / gravity_count : 0.0;
    const double first_dt = points[1].t - points[0].t;
    const double first_slope = (points[1].y - points[0].y) / first_dt;
    const double vy0 = first_slope - g * first_dt;
    return {vy0, g};
}

void validate_observations(const Trajectory& observed) {
    if (!(observed.fps > 0.0) || !std::isfinite(observed.fps)) {
        throw std::invalid_argument("trajectory fps must be finite and > 0");
    }
    if (observed.observations.size() < 8) {
        throw std::invalid_argument("fitting requires at least 8 observations");
    }
    for (std::size_t i = 0; i < observed.observations.size(); ++i) {
        const auto& point = observed.observations[i];
        if (!std::isfinite(point.t) || !std::isfinite(point.x) || !std::isfinite(point.y)) {
            throw std::invalid_argument("observations must contain finite t, x, and y");
        }
        if (i > 0 && point.t <= observed.observations[i - 1].t) {
            throw std::invalid_argument("observation times must be strictly increasing");
        }
    }
}

}  // namespace

Reconstruction Fitter::fit(const Trajectory& observed) const {
    validate_observations(observed);
    const auto start = std::chrono::steady_clock::now();

    const double t0 = observed.observations.front().t;
    std::vector<double> times;
    times.reserve(observed.observations.size());
    for (const auto& point : observed.observations) {
        times.push_back(point.t - t0);
    }

    Environment env;
    env.x0 = observed.observations.front().x;
    env.y0 = observed.observations.front().y;
    env.dt = 1.0 / observed.fps;
    env.y_ground = std::max_element(
                       observed.observations.begin(),
                       observed.observations.end(),
                       [](const Observation& left, const Observation& right) {
                           return left.y < right.y;
                       })
                       ->y;

    const double vx0 = estimate_vx(observed, times);
    const auto [estimated_vy0, estimated_g] =
        estimate_initial_vertical_motion(observed, env.y_ground, env.dt);

    double max_vertical_speed = 0.0;
    for (std::size_t i = 1; i < observed.observations.size(); ++i) {
        const double elapsed =
            observed.observations[i].t - observed.observations[i - 1].t;
        max_vertical_speed =
            std::max(max_vertical_speed,
                     std::abs(observed.observations[i].y -
                              observed.observations[i - 1].y) /
                         elapsed);
    }
    const double duration = times.back();
    const Bounds bounds = {
        .velocity_limit =
            std::max({100.0, 4.0 * max_vertical_speed, 2.0 * std::abs(estimated_vy0)}),
        .gravity_limit =
            std::max({100.0,
                      8.0 * max_vertical_speed / duration,
                      2.0 * std::abs(estimated_g)}),
    };

    const Simulator simulator;
    int objective_evaluations = 0;
    const auto objective = [&](const Candidate& candidate) {
        ++objective_evaluations;
        const Trajectory simulated = simulator.run(decode(candidate, vx0, bounds), env, times);
        double sum = 0.0;
        for (std::size_t i = 0; i < observed.observations.size(); ++i) {
            const double dy = observed.observations[i].y - simulated.observations[i].y;
            sum += dy * dy;
        }
        return sum / static_cast<double>(observed.observations.size());
    };

    // The hard collision clamp makes the residual piecewise smooth. A fixed-seed
    // bounded differential search gives a repeatable global initialization.
    constexpr std::size_t population_size = 48;
    constexpr int generations = 160;
    constexpr double differential_weight = 0.75;
    constexpr double crossover_rate = 0.9;

    std::mt19937 generator(0x50545931U);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_int_distribution<std::size_t> member(0, population_size - 1);

    std::vector<Candidate> population(population_size);
    std::vector<double> scores(population_size);
    population[0] = encode(estimated_vy0, estimated_g, 0.5, bounds);
    population[1] = encode(estimated_vy0, estimated_g, 0.75, bounds);
    population[2] = encode(estimated_vy0, estimated_g, 0.25, bounds);
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
            const std::size_t forced_dimension = member(generator) % trial.size();
            for (std::size_t dimension = 0; dimension < trial.size(); ++dimension) {
                if (unit(generator) <= crossover_rate || dimension == forced_dimension) {
                    trial[dimension] =
                        clamp_unit(population[a][dimension] +
                                   differential_weight *
                                       (population[b][dimension] - population[c][dimension]));
                }
            }

            const double trial_score = objective(trial);
            if (trial_score < scores[i]) {
                population[i] = trial;
                scores[i] = trial_score;
            }
        }
    }

    const auto best_position =
        std::min_element(scores.begin(), scores.end()) - scores.begin();
    Candidate best = population[static_cast<std::size_t>(best_position)];
    double best_score = scores[static_cast<std::size_t>(best_position)];

    // Deterministic coordinate refinement recovers sub-grid parameter values
    // once the global search has found the correct collision schedule.
    double step = 0.05;
    int refinement_iterations = 0;
    while (step > 1e-8 && refinement_iterations < 256) {
        Candidate next_best = best;
        double next_score = best_score;
        for (std::size_t dimension = 0; dimension < best.size(); ++dimension) {
            for (const double direction : {-1.0, 1.0}) {
                Candidate trial = best;
                trial[dimension] = clamp_unit(trial[dimension] + direction * step);
                const double score = objective(trial);
                if (score < next_score) {
                    next_best = trial;
                    next_score = score;
                }
            }
        }
        if (next_score < best_score) {
            best = next_best;
            best_score = next_score;
        } else {
            step *= 0.5;
        }
        ++refinement_iterations;
    }

    Reconstruction reconstruction;
    reconstruction.parameters = decode(best, vx0, bounds);
    reconstruction.environment = env;
    reconstruction.simulated =
        simulator.run(reconstruction.parameters, reconstruction.environment, times);
    reconstruction.simulated.frame_width = observed.frame_width;
    reconstruction.simulated.frame_height = observed.frame_height;
    for (std::size_t i = 0; i < reconstruction.simulated.observations.size(); ++i) {
        reconstruction.simulated.observations[i].frame = observed.observations[i].frame;
        reconstruction.simulated.observations[i].t = observed.observations[i].t;
    }
    reconstruction.rmse = rmse(observed, reconstruction.simulated);
    reconstruction.mae = mae(observed, reconstruction.simulated);
    reconstruction.n = static_cast<int>(observed.observations.size());
    reconstruction.iterations = generations + refinement_iterations;
    reconstruction.fit_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    (void)objective_evaluations;
    return reconstruction;
}

}  // namespace phystwin
