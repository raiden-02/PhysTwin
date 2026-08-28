#include "phystwin/pendulum.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(const double actual,
                  const double expected,
                  const double tolerance,
                  const std::string& name) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            name + " error exceeded tolerance: expected " +
            std::to_string(expected) + ", got " + std::to_string(actual) +
            ", tolerance " + std::to_string(tolerance));
    }
}

template <typename Function>
void require_throws(Function&& function, const std::string& expected_text) {
    try {
        function();
    } catch (const std::invalid_argument& error) {
        require(std::string(error.what()).find(expected_text) != std::string::npos,
                "wrong failure text: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected invalid_argument containing: " +
                             expected_text);
}

phystwin::Trajectory make_exact() {
    constexpr phystwin::PendulumParameters parameters{
        .omega0 = -0.35,
        .lambda = 7.2,
        .damping = 0.22,
    };
    constexpr phystwin::PendulumEnvironment environment{
        .pivot_x = 400.0,
        .pivot_y = 170.0,
        .radius = 220.0,
        .theta0 = 0.9,
        .integration_step = 1.0 / 240.0,
    };
    std::vector<double> times;
    for (int frame = 0; frame <= 240; ++frame) {
        times.push_back(static_cast<double>(frame) / 60.0);
    }
    phystwin::Trajectory trajectory =
        phystwin::PendulumSimulator{}.run(parameters, environment, times);
    trajectory.fps = 60.0;
    trajectory.frame_width = 800;
    trajectory.frame_height = 600;
    for (std::size_t i = 0; i < trajectory.observations.size(); ++i) {
        trajectory.observations[i].frame = static_cast<int>(i);
    }
    return trajectory;
}

}  // namespace

int main() {
    try {
        const phystwin::Trajectory exact = make_exact();
        const phystwin::PendulumReconstruction recovered =
            phystwin::PendulumFitter{}.fit(exact);

        require_near(recovered.parameters.omega0, -0.35, 0.03, "omega0");
        require_near(recovered.parameters.lambda, 7.2, 0.08, "lambda");
        require_near(recovered.parameters.damping, 0.22, 0.025, "damping");
        require_near(recovered.environment.radius, 220.0, 1e-8, "radius");
        require(recovered.rmse < 0.15, "noise-free RMSE exceeded 0.15 px");
        require(recovered.quality == "good", "noise-free fit must be good");
        require(recovered.environment.anchor_mode == phystwin::AnchorMode::fixed,
                "fixed mode regression changed anchor mode");
        require(recovered.simulated.anchor_observations.empty(),
                "fixed mode must not emit a tracked anchor path");

        phystwin::Trajectory moving_camera = exact;
        moving_camera.anchor_mode = phystwin::AnchorMode::tracked;
        moving_camera.anchor_track_coverage = 1.0;
        moving_camera.anchor_observations.clear();
        moving_camera.anchor_observations.reserve(
            moving_camera.observations.size());
        for (auto& point : moving_camera.observations) {
            const double camera_x = 28.0 * std::sin(0.8 * point.t) + 7.0 * point.t;
            const double camera_y = 19.0 * std::cos(0.55 * point.t) - 19.0;
            point.x += camera_x;
            point.y += camera_y;
            moving_camera.anchor_observations.push_back({
                .frame = point.frame,
                .t = point.t,
                .x = exact.pivot->x + camera_x,
                .y = exact.pivot->y + camera_y,
            });
        }
        const phystwin::PendulumReconstruction camera_recovered =
            phystwin::PendulumFitter{}.fit(moving_camera);
        require_near(
            camera_recovered.parameters.omega0, -0.35, 0.03, "camera omega0");
        require_near(
            camera_recovered.parameters.lambda, 7.2, 0.08, "camera lambda");
        require_near(
            camera_recovered.parameters.damping, 0.22, 0.025, "camera damping");
        require(camera_recovered.rmse < 0.15,
                "moving-camera RMSE exceeded 0.15 px");
        require(camera_recovered.environment.anchor_mode ==
                    phystwin::AnchorMode::tracked,
                "tracked anchor mode was not preserved");
        require(camera_recovered.simulated.anchor_observations.size() ==
                    moving_camera.observations.size(),
                "tracked reconstruction anchor path size");

        constexpr phystwin::PendulumParameters fast_parameters{
            .omega0 = -0.2,
            .lambda = 32.0,
            .damping = 0.04,
        };
        constexpr phystwin::PendulumEnvironment fast_environment{
            .pivot_x = 400.0,
            .pivot_y = 170.0,
            .radius = 220.0,
            .theta0 = -0.35,
            .integration_step = 1.0 / 240.0,
        };
        std::vector<double> fast_times;
        for (int frame = 0; frame <= 180; ++frame) {
            fast_times.push_back(static_cast<double>(frame) / 60.0);
        }
        phystwin::Trajectory fast =
            phystwin::PendulumSimulator{}.run(
                fast_parameters, fast_environment, fast_times);
        fast.fps = 60.0;
        const phystwin::PendulumReconstruction fast_recovered =
            phystwin::PendulumFitter{}.fit(fast);
        require_near(
            fast_recovered.parameters.lambda, 32.0, 0.2, "fast lambda");
        require(fast_recovered.rmse < 0.2,
                "fast pendulum RMSE exceeded 0.2 px");

        phystwin::Trajectory noisy = exact;
        std::mt19937 generator(0x4e4f4953U);
        std::normal_distribution<double> noise(0.0, 0.8);
        for (auto& point : noisy.observations) {
            point.x += noise(generator);
            point.y += noise(generator);
        }
        for (const int frame : {55, 121, 207}) {
            noisy.observations[static_cast<std::size_t>(frame)].x +=
                frame == 121 ? -18.0 : 16.0;
            noisy.observations[static_cast<std::size_t>(frame)].y +=
                frame == 207 ? -15.0 : 13.0;
        }
        const phystwin::PendulumReconstruction robust =
            phystwin::PendulumFitter{}.fit(noisy);
        require_near(robust.parameters.omega0, -0.35, 0.18, "noisy omega0");
        require_near(robust.parameters.lambda, 7.2, 0.45, "noisy lambda");
        require_near(robust.parameters.damping, 0.22, 0.12, "noisy damping");
        require(robust.rmse > 0.5,
                "noisy/outlier case must not report an apparently perfect RMSE");
        require(robust.rmse < 4.0, "noisy/outlier RMSE exceeded 4 px");
        require(robust.robust_cost > 0.0, "robust objective must be reported");
        require(robust.quality != "poor", "noisy/outlier fit became poor");

        phystwin::Trajectory too_few = exact;
        too_few.observations.resize(8);
        require_throws(
            [&] { (void)phystwin::PendulumFitter{}.fit(too_few); },
            "at least 12");

        phystwin::Trajectory stationary = exact;
        for (auto& point : stationary.observations) {
            point.x = exact.observations.front().x;
            point.y = exact.observations.front().y;
        }
        require_throws(
            [&] { (void)phystwin::PendulumFitter{}.fit(stationary); },
            "insufficient angular span");

        phystwin::Trajectory zero_radius = exact;
        for (auto& point : zero_radius.observations) {
            point.x = zero_radius.pivot->x + 1.0;
            point.y = zero_radius.pivot->y + 1.0;
        }
        require_throws(
            [&] { (void)phystwin::PendulumFitter{}.fit(zero_radius); },
            "below 5 px");

        phystwin::Trajectory unusable_pivot = exact;
        for (std::size_t i = 0; i < unusable_pivot.observations.size(); ++i) {
            const double angle = 0.65 * std::sin(0.08 * static_cast<double>(i));
            const double radius = 80.0 + 60.0 * static_cast<double>(i % 5);
            unusable_pivot.observations[i].x =
                unusable_pivot.pivot->x + radius * std::sin(angle);
            unusable_pivot.observations[i].y =
                unusable_pivot.pivot->y + radius * std::cos(angle);
        }
        require_throws(
            [&] { (void)phystwin::PendulumFitter{}.fit(unusable_pivot); },
            "varies too much");

        phystwin::Trajectory invalid_time = exact;
        invalid_time.observations[20].t =
            invalid_time.observations[19].t;
        require_throws(
            [&] { (void)phystwin::PendulumFitter{}.fit(invalid_time); },
            "strictly increasing");

        phystwin::Trajectory misaligned_anchor = moving_camera;
        misaligned_anchor.anchor_observations[20].frame += 1;
        require_throws(
            [&] { (void)phystwin::PendulumFitter{}.fit(misaligned_anchor); },
            "frame-aligned");

        phystwin::Trajectory low_anchor_coverage = moving_camera;
        low_anchor_coverage.anchor_track_coverage = 0.5;
        require_throws(
            [&] { (void)phystwin::PendulumFitter{}.fit(low_anchor_coverage); },
            "at least 60%");

        phystwin::Trajectory varying_relative_radius = moving_camera;
        for (std::size_t i = 0;
             i < varying_relative_radius.observations.size();
             ++i) {
            const auto& anchor =
                varying_relative_radius.anchor_observations[i];
            const double angle =
                0.65 * std::sin(0.08 * static_cast<double>(i));
            const double radius = 80.0 + 60.0 * static_cast<double>(i % 5);
            varying_relative_radius.observations[i].x =
                anchor.x + radius * std::sin(angle);
            varying_relative_radius.observations[i].y =
                anchor.y + radius * std::cos(angle);
        }
        require_throws(
            [&] {
                (void)phystwin::PendulumFitter{}.fit(varying_relative_radius);
            },
            "varies too much");

        std::cout << "pendulum synthetic recovery\n"
                  << "parameter     actual       recovered      abs_error\n"
                  << "omega0        -0.35        "
                  << recovered.parameters.omega0 << "          "
                  << std::abs(recovered.parameters.omega0 + 0.35) << "\n"
                  << "lambda        7.2          "
                  << recovered.parameters.lambda << "          "
                  << std::abs(recovered.parameters.lambda - 7.2) << "\n"
                  << "damping       0.22         "
                  << recovered.parameters.damping << "          "
                  << std::abs(recovered.parameters.damping - 0.22) << "\n"
                  << "RMSE: " << recovered.rmse << " px\n"
                  << "moving-camera RMSE: " << camera_recovered.rmse << " px\n"
                  << "moving-camera lambda: "
                  << camera_recovered.parameters.lambda << "\n"
                  << "noisy/outlier RMSE: " << robust.rmse << " px\n"
                  << "noisy lambda: " << robust.parameters.lambda << "\n"
                  << "noisy damping: " << robust.parameters.damping << "\n"
                  << "fast lambda: " << fast_recovered.parameters.lambda << "\n"
                  << "fast RMSE: " << fast_recovered.rmse << " px\n"
                  << "degenerate_checks: 8\n"
                  << "search_generations: "
                  << recovered.search_generations << "\n"
                  << "fit time: " << recovered.fit_seconds << " s\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pendulum_fit failed: " << error.what() << "\n";
        return 1;
    }
}
