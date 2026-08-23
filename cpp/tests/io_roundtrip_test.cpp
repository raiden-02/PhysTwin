#include "phystwin/io.hpp"
#include "phystwin/metrics.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

bool nearly_equal(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) <= tol;
}

template <typename Function>
void require_throws(Function&& function, const std::string& expected_text) {
    try {
        function();
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()).find(expected_text) != std::string::npos,
                "wrong failure text: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected runtime_error containing: " +
                             expected_text);
}

}  // namespace

int main() {
    try {
        phystwin::Trajectory original;
        original.version = 1;
        original.fps = 60.0;
        original.frame_width = 1920;
        original.frame_height = 1080;
        original.observations = {
            {.frame = 0, .t = 0.0, .x = 531.2, .y = 312.7, .confidence = 0.9},
            {.frame = 1, .t = 1.0 / 60.0, .x = 534.1, .y = 306.8},
        };

        const auto path = std::filesystem::temp_directory_path() / "phystwin_io_roundtrip.json";
        phystwin::save_tracking(original, path);
        const phystwin::Trajectory loaded = phystwin::load_tracking(path);

        require(loaded.version == 1, "version");
        require(nearly_equal(loaded.fps, 60.0), "fps");
        require(loaded.frame_width == 1920, "frame_width");
        require(loaded.frame_height == 1080, "frame_height");
        require(loaded.observations.size() == 2, "observation count");
        require(loaded.observations[0].frame == 0, "frame 0");
        require(nearly_equal(loaded.observations[0].x, 531.2), "x0");
        require(nearly_equal(loaded.observations[0].y, 312.7), "y0");
        require(loaded.observations[0].confidence.has_value(), "confidence present");
        require(nearly_equal(*loaded.observations[0].confidence, 0.9), "confidence value");
        require(!loaded.observations[1].confidence.has_value(), "confidence omitted");
        require(loaded.model == phystwin::DynamicsModel::projectile_bounce,
                "legacy/default model");

        phystwin::Trajectory same = loaded;
        const double zero = phystwin::rmse(loaded, same);
        require(nearly_equal(zero, 0.0), "RMSE of identical trajectories");

        std::filesystem::remove(path);

        phystwin::Trajectory pendulum = original;
        pendulum.model = phystwin::DynamicsModel::pendulum;
        pendulum.pivot = phystwin::ReferencePoint{500.0, 100.0};
        phystwin::save_tracking(pendulum, path);
        const phystwin::Trajectory loaded_pendulum =
            phystwin::load_tracking(path);
        require(loaded_pendulum.model == phystwin::DynamicsModel::pendulum,
                "pendulum model");
        require(loaded_pendulum.pivot.has_value(), "pendulum pivot present");
        require(nearly_equal(loaded_pendulum.pivot->x, 500.0), "pivot x");
        require(nearly_equal(loaded_pendulum.pivot->y, 100.0), "pivot y");
        std::filesystem::remove(path);

        phystwin::Trajectory tracked = pendulum;
        tracked.anchor_mode = phystwin::AnchorMode::tracked;
        tracked.anchor_track_coverage = 0.92;
        tracked.observations.clear();
        tracked.anchor_observations.clear();
        for (int frame = 0; frame < 12; ++frame) {
            const double t = static_cast<double>(frame) / 60.0;
            tracked.observations.push_back(
                {.frame = frame, .t = t, .x = 520.0 + frame, .y = 300.0});
            tracked.anchor_observations.push_back(
                {.frame = frame, .t = t, .x = 500.0 + frame, .y = 100.0});
        }
        phystwin::save_tracking(tracked, path);
        const phystwin::Trajectory loaded_tracked =
            phystwin::load_tracking(path);
        require(loaded_tracked.anchor_mode == phystwin::AnchorMode::tracked,
                "tracked anchor mode");
        require(loaded_tracked.anchor_observations.size() == 12,
                "anchor observation count");
        require(loaded_tracked.anchor_track_coverage.has_value(),
                "anchor coverage present");
        require(nearly_equal(*loaded_tracked.anchor_track_coverage, 0.92),
                "anchor coverage");

        tracked.anchor_observations[4].frame = 5;
        phystwin::save_tracking(tracked, path);
        require_throws(
            [&] { (void)phystwin::load_tracking(path); }, "frame-aligned");
        std::cout << "io_roundtrip: ok\n";
        std::filesystem::remove(path);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "io_roundtrip failed: " << ex.what() << "\n";
        return 1;
    }
}
