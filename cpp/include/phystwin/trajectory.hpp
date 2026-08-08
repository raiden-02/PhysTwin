#pragma once

#include <optional>
#include <vector>

namespace phystwin {

// One tracked position. Image pixels, origin top-left, +x right, +y down.
struct Observation {
    int frame = 0;
    double t = 0.0;
    double x = 0.0;
    double y = 0.0;
    std::optional<double> confidence;
    std::optional<double> bbox_x;
    std::optional<double> bbox_y;
    std::optional<double> bbox_w;
    std::optional<double> bbox_h;
    std::optional<double> radius;
};

struct Trajectory {
    int version = 1;
    double fps = 0.0;
    int frame_width = 0;
    int frame_height = 0;
    std::vector<Observation> observations;
};

}  // namespace phystwin
