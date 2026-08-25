#include "phystwin/fitter.hpp"
#include "phystwin/metrics.hpp"
#include "phystwin/simulator.hpp"
#include "phystwin/synthetic.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(double actual, double expected, double tolerance, const std::string& name) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(name + " error exceeded tolerance: expected " +
                                 std::to_string(expected) + ", got " +
                                 std::to_string(actual) + ", tolerance " +
                                 std::to_string(tolerance));
    }
}

}  // namespace

int main() {
    try {
        {
            constexpr phystwin::Parameters step_parameters = {
                .vx0 = 2.0,
                .vy0 = -1.0,
                .g = 10.0,
                .e = 0.5,
            };
            constexpr phystwin::Environment step_environment = {
                .x0 = 0.0,
                .y0 = 0.0,
                .y_ground = 100.0,
                .dt = 0.1,
            };
            const phystwin::Trajectory stepped =
                phystwin::Simulator{}.run(step_parameters, step_environment, {0.0, 0.1, 0.2});
            require_near(stepped.observations[1].x, 0.2, 1e-12, "Euler x step 1");
            require_near(stepped.observations[1].y, 0.0, 1e-12, "Euler y step 1");
            require_near(stepped.observations[2].x, 0.4, 1e-12, "Euler x step 2");
            require_near(stepped.observations[2].y, 0.1, 1e-12, "Euler y step 2");

            constexpr phystwin::Environment collision_environment = {
                .x0 = 0.0,
                .y0 = 9.0,
                .y_ground = 10.0,
                .dt = 0.1,
            };
            const phystwin::Trajectory bounced = phystwin::Simulator{}.run(
                {.vx0 = 0.0, .vy0 = 5.0, .g = 0.0, .e = 0.5},
                collision_environment,
                {0.0, 0.1, 0.2, 0.3});
            require_near(bounced.observations[2].y, 10.0, 1e-12, "ground clamp");
            require_near(bounced.observations[3].y, 9.75, 1e-12, "restitution bounce");
        }

        constexpr phystwin::Parameters actual = {
            .vx0 = 180.0,
            .vy0 = -420.0,
            .g = 980.0,
            .e = 0.72,
        };
        constexpr phystwin::Environment environment = {
            .x0 = 150.0,
            .y0 = 120.0,
            .y_ground = 720.0,
            .dt = 1.0 / 60.0,
        };

        const phystwin::Trajectory observed =
            phystwin::generate_synthetic(actual, environment, 241);
        require(observed.observations.size() == 241, "synthetic frame count");
        require_near(observed.observations.front().x, environment.x0, 1e-12, "initial x");
        require_near(observed.observations.front().y, environment.y0, 1e-12, "initial y");
        int ground_contacts = 0;
        for (const auto& point : observed.observations) {
            if (std::abs(point.y - environment.y_ground) < 1e-12) {
                ++ground_contacts;
            }
        }
        require(ground_contacts >= 2, "synthetic case must contain at least two ground contacts");

        const phystwin::Reconstruction recovered = phystwin::Fitter{}.fit(observed);

        // Explicit recovery tolerances. These are tight enough that changing a
        // fitted parameter materially makes this test fail.
        constexpr double velocity_tolerance = 0.5;      // px/s
        constexpr double gravity_tolerance = 1.0;       // px/s^2
        constexpr double restitution_tolerance = 0.002; // dimensionless
        constexpr double rmse_tolerance = 0.1;          // px

        require_near(recovered.parameters.vx0, actual.vx0, velocity_tolerance, "vx0");
        require_near(recovered.parameters.vy0, actual.vy0, velocity_tolerance, "vy0");
        require_near(recovered.parameters.g, actual.g, gravity_tolerance, "g");
        require_near(
            recovered.parameters.e, actual.e, restitution_tolerance, "restitution");
        require(recovered.rmse < rmse_tolerance, "recovered RMSE exceeded 0.1 px");
        require(recovered.mae < rmse_tolerance, "recovered MAE exceeded 0.1 px");
        require(recovered.quality == "good", "exact synthetic fit must be good");
        require(recovered.ground_source == "max_observed_centroid_y",
                "default ground source");

        const phystwin::Reconstruction explicit_ground = phystwin::Fitter{}.fit(
            observed, {.ground_y = environment.y_ground});
        require_near(explicit_ground.environment.y_ground,
                     environment.y_ground,
                     1e-12,
                     "explicit ground");
        require(explicit_ground.ground_source == "explicit",
                "explicit ground source");
        require(explicit_ground.rmse < rmse_tolerance,
                "explicit-ground RMSE exceeded 0.1 px");

        const phystwin::Reconstruction invalid_ground = phystwin::Fitter{}.fit(
            observed, {.ground_y = environment.y_ground - 120.0});
        require(invalid_ground.quality == "poor",
                "ground crossed by observations must be a poor fit");
        require_near(invalid_ground.ground_violation,
                     120.0,
                     1e-9,
                     "ground violation");

        // Negative control: the same simulator with a perturbed fit must be
        // observably wrong. This catches a metric or assertion wired to the
        // wrong trajectory.
        phystwin::Parameters perturbed = actual;
        perturbed.vy0 += 40.0;
        perturbed.g *= 0.9;
        perturbed.e -= 0.1;
        std::vector<double> times;
        times.reserve(observed.observations.size());
        for (const auto& point : observed.observations) {
            times.push_back(point.t);
        }
        const phystwin::Trajectory wrong =
            phystwin::Simulator{}.run(perturbed, environment, times);
        const double perturbed_rmse = phystwin::rmse(observed, wrong);
        require(perturbed_rmse > 20.0, "perturbed negative-control RMSE was too small");

        std::cout << "synthetic parameter recovery\n"
                  << "parameter     actual       recovered      abs_error\n"
                  << "vx0           " << actual.vx0 << "          "
                  << recovered.parameters.vx0 << "          "
                  << std::abs(recovered.parameters.vx0 - actual.vx0) << "\n"
                  << "vy0           " << actual.vy0 << "         "
                  << recovered.parameters.vy0 << "         "
                  << std::abs(recovered.parameters.vy0 - actual.vy0) << "\n"
                  << "g             " << actual.g << "          "
                  << recovered.parameters.g << "          "
                  << std::abs(recovered.parameters.g - actual.g) << "\n"
                  << "e             " << actual.e << "         "
                  << recovered.parameters.e << "         "
                  << std::abs(recovered.parameters.e - actual.e) << "\n"
                  << "RMSE: " << recovered.rmse << " px\n"
                  << "MAE: " << recovered.mae << " px\n"
                  << "perturbed RMSE: " << perturbed_rmse << " px\n"
                  << "search_generations: " << recovered.search_generations << "\n"
                  << "refinement_iterations: " << recovered.refinement_iterations
                  << "\n"
                  << "fit time: " << recovered.fit_seconds << " s\n";
        require(recovered.search_generations == 160, "search generation budget");
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "synthetic_fit failed: " << ex.what() << "\n";
        return 1;
    }
}
