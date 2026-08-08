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

        phystwin::Trajectory same = loaded;
        const double zero = phystwin::rmse(loaded, same);
        require(nearly_equal(zero, 0.0), "RMSE of identical trajectories");

        std::filesystem::remove(path);
        std::cout << "io_roundtrip: ok\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "io_roundtrip failed: " << ex.what() << "\n";
        return 1;
    }
}
