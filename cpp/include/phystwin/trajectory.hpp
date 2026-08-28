#pragma once

#include <optional>
#include <string_view>
#include <vector>

namespace phystwin {

enum class DynamicsModel {
    projectile_bounce,
    pendulum,
};

enum class AnchorMode {
    fixed,
    tracked,
};

std::string_view model_name(DynamicsModel model);
DynamicsModel parse_model(std::string_view name);
std::string_view anchor_mode_name(AnchorMode mode);
AnchorMode parse_anchor_mode(std::string_view name);

struct ReferencePoint {
    double x = 0.0;
    double y = 0.0;
};

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
    DynamicsModel model = DynamicsModel::projectile_bounce;
    double fps = 0.0;
    int frame_width = 0;
    int frame_height = 0;
    AnchorMode anchor_mode = AnchorMode::fixed;
    std::optional<ReferencePoint> pivot;
    std::optional<double> anchor_track_coverage;
    std::vector<Observation> anchor_observations;
    std::vector<Observation> observations;
};

}  // namespace phystwin
