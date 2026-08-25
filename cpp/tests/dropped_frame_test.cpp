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

        const phystwin::Trajectory full =
            phystwin::generate_synthetic(actual, environment, 241);

        phystwin::Trajectory gapped;
        gapped.version = full.version;
        gapped.fps = full.fps;
        gapped.frame_width = full.frame_width;
        gapped.frame_height = full.frame_height;
        for (const auto& point : full.observations) {
            // Drop a contiguous interior gap, as empty SAM 2 masks would.
            if (point.frame >= 40 && point.frame <= 46) {
                continue;
            }
            gapped.observations.push_back(point);
        }
        require(gapped.observations.size() == 234, "gapped observation count");
        for (std::size_t i = 1; i < gapped.observations.size(); ++i) {
            require(gapped.observations[i].t > gapped.observations[i - 1].t,
                    "gapped times must stay strictly increasing");
        }

        std::vector<double> gapped_times;
        gapped_times.reserve(gapped.observations.size());
        for (const auto& point : gapped.observations) {
            gapped_times.push_back(point.t);
        }
        const phystwin::Trajectory resampled =
            phystwin::Simulator{}.run(actual, environment, gapped_times);
        require_near(phystwin::rmse(gapped, resampled), 0.0, 1e-9,
                     "simulator samples observation times, not reconstructed indices");

        const phystwin::Reconstruction recovered = phystwin::Fitter{}.fit(gapped);
        require_near(recovered.parameters.vx0, actual.vx0, 0.5, "vx0");
        require_near(recovered.parameters.vy0, actual.vy0, 0.5, "vy0");
        require_near(recovered.parameters.g, actual.g, 1.0, "g");
        require_near(recovered.parameters.e, actual.e, 0.002, "restitution");
        require(recovered.rmse < 0.1, "gapped RMSE exceeded 0.1 px");
        require(recovered.n == 234, "fit observation count");

        std::cout << "dropped_frame: ok n=" << recovered.n
                  << " RMSE=" << recovered.rmse << " px\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "dropped_frame failed: " << ex.what() << "\n";
        return 1;
    }
}
