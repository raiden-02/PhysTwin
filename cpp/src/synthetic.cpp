#include "phystwin/synthetic.hpp"

#include "phystwin/simulator.hpp"

#include <stdexcept>
#include <vector>

namespace phystwin {

Trajectory generate_synthetic(const Parameters& theta,
                              const Environment& env,
                              int frame_count) {
    if (frame_count < 1) {
        throw std::invalid_argument("synthetic frame count must be >= 1");
    }

    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(frame_count));
    for (int frame = 0; frame < frame_count; ++frame) {
        times.push_back(static_cast<double>(frame) * env.dt);
    }

    Trajectory result = Simulator{}.run(theta, env, times);
    for (int frame = 0; frame < frame_count; ++frame) {
        result.observations[static_cast<std::size_t>(frame)].frame = frame;
    }
    return result;
}

}  // namespace phystwin
