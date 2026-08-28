#include "phystwin/io.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

namespace phystwin {
namespace {

nlohmann::json observation_to_json(const Observation& obs) {
    nlohmann::json j = {
        {"frame", obs.frame},
        {"t", obs.t},
        {"x", obs.x},
        {"y", obs.y},
    };
    if (obs.confidence) {
        j["confidence"] = *obs.confidence;
    }
    if (obs.bbox_x) {
        j["bbox_x"] = *obs.bbox_x;
    }
    if (obs.bbox_y) {
        j["bbox_y"] = *obs.bbox_y;
    }
    if (obs.bbox_w) {
        j["bbox_w"] = *obs.bbox_w;
    }
    if (obs.bbox_h) {
        j["bbox_h"] = *obs.bbox_h;
    }
    if (obs.radius) {
        j["radius"] = *obs.radius;
    }
    return j;
}

Observation observation_from_json(const nlohmann::json& j) {
    if (!j.is_object()) {
        throw std::runtime_error("observation must be an object");
    }
    for (const char* key : {"frame", "t", "x", "y"}) {
        if (!j.contains(key)) {
            throw std::runtime_error(std::string("observation missing field: ") + key);
        }
    }

    Observation obs;
    obs.frame = j.at("frame").get<int>();
    obs.t = j.at("t").get<double>();
    obs.x = j.at("x").get<double>();
    obs.y = j.at("y").get<double>();
    if (j.contains("confidence") && !j["confidence"].is_null()) {
        obs.confidence = j["confidence"].get<double>();
    }
    if (j.contains("bbox_x") && !j["bbox_x"].is_null()) {
        obs.bbox_x = j["bbox_x"].get<double>();
    }
    if (j.contains("bbox_y") && !j["bbox_y"].is_null()) {
        obs.bbox_y = j["bbox_y"].get<double>();
    }
    if (j.contains("bbox_w") && !j["bbox_w"].is_null()) {
        obs.bbox_w = j["bbox_w"].get<double>();
    }
    if (j.contains("bbox_h") && !j["bbox_h"].is_null()) {
        obs.bbox_h = j["bbox_h"].get<double>();
    }
    if (j.contains("radius") && !j["radius"].is_null()) {
        obs.radius = j["radius"].get<double>();
    }
    return obs;
}

nlohmann::json load_json_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open " + path.string());
    }
    nlohmann::json j;
    in >> j;
    return j;
}

void write_json_file(const std::filesystem::path& path, const nlohmann::json& j) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write " + path.string());
    }
    out << j.dump(2) << '\n';
}

}  // namespace

std::string_view model_name(const DynamicsModel model) {
    switch (model) {
    case DynamicsModel::projectile_bounce:
        return "projectile_bounce";
    case DynamicsModel::pendulum:
        return "pendulum";
    }
    throw std::invalid_argument("unknown dynamics model");
}

DynamicsModel parse_model(const std::string_view name) {
    if (name == "projectile_bounce") {
        return DynamicsModel::projectile_bounce;
    }
    if (name == "pendulum") {
        return DynamicsModel::pendulum;
    }
    throw std::invalid_argument("unknown dynamics model: " + std::string(name));
}

std::string_view anchor_mode_name(const AnchorMode mode) {
    switch (mode) {
    case AnchorMode::fixed:
        return "fixed";
    case AnchorMode::tracked:
        return "tracked";
    }
    throw std::invalid_argument("unknown anchor mode");
}

AnchorMode parse_anchor_mode(const std::string_view name) {
    if (name == "fixed") {
        return AnchorMode::fixed;
    }
    if (name == "tracked") {
        return AnchorMode::tracked;
    }
    throw std::invalid_argument("unknown anchor mode: " + std::string(name));
}

Trajectory load_tracking(const std::filesystem::path& path) {
    const nlohmann::json j = load_json_file(path);
    if (!j.is_object()) {
        throw std::runtime_error("tracking.json must be an object");
    }
    if (!j.contains("version") || j.at("version").get<int>() != 1) {
        throw std::runtime_error("tracking.json version must be 1");
    }
    if (!j.contains("fps") || j.at("fps").get<double>() <= 0.0) {
        throw std::runtime_error("tracking.json fps must be > 0");
    }
    if (!j.contains("observations") || !j.at("observations").is_array() ||
        j.at("observations").empty()) {
        throw std::runtime_error("tracking.json observations must be a non-empty array");
    }

    Trajectory traj;
    traj.version = 1;
    if (j.contains("model")) {
        traj.model = parse_model(j.at("model").get<std::string>());
    }
    traj.fps = j.at("fps").get<double>();
    if (j.contains("frame_width")) {
        traj.frame_width = j.at("frame_width").get<int>();
    }
    if (j.contains("frame_height")) {
        traj.frame_height = j.at("frame_height").get<int>();
    }
    if (j.contains("reference") && !j.at("reference").is_null()) {
        const auto& reference = j.at("reference");
        if (!reference.is_object() || !reference.contains("pivot_x") ||
            !reference.contains("pivot_y")) {
            throw std::runtime_error(
                "tracking.json reference must contain pivot_x and pivot_y");
        }
        if (reference.contains("mode")) {
            traj.anchor_mode =
                parse_anchor_mode(reference.at("mode").get<std::string>());
        }
        traj.pivot = ReferencePoint{
            .x = reference.at("pivot_x").get<double>(),
            .y = reference.at("pivot_y").get<double>(),
        };
        if (reference.contains("coverage") && !reference.at("coverage").is_null()) {
            traj.anchor_track_coverage = reference.at("coverage").get<double>();
            if (!std::isfinite(*traj.anchor_track_coverage) ||
                *traj.anchor_track_coverage < 0.0 ||
                *traj.anchor_track_coverage > 1.0) {
                throw std::runtime_error(
                    "tracking.json reference.coverage must be between 0 and 1");
            }
        }
    }
    if (traj.model == DynamicsModel::pendulum && !traj.pivot.has_value()) {
        throw std::runtime_error("pendulum tracking.json requires a pivot reference");
    }
    traj.observations.reserve(j.at("observations").size());
    for (const auto& item : j.at("observations")) {
        traj.observations.push_back(observation_from_json(item));
    }
    if (j.contains("anchor_observations")) {
        if (!j.at("anchor_observations").is_array()) {
            throw std::runtime_error(
                "tracking.json anchor_observations must be an array");
        }
        traj.anchor_observations.reserve(j.at("anchor_observations").size());
        for (const auto& item : j.at("anchor_observations")) {
            traj.anchor_observations.push_back(observation_from_json(item));
        }
    }
    if (traj.anchor_mode == AnchorMode::tracked) {
        if (traj.anchor_observations.size() < 12) {
            throw std::runtime_error(
                "tracked pendulum requires at least 12 anchor observations");
        }
        if (traj.anchor_observations.size() != traj.observations.size()) {
            throw std::runtime_error(
                "tracked pendulum target and anchor observations must be frame-aligned");
        }
        for (std::size_t i = 0; i < traj.observations.size(); ++i) {
            if (traj.observations[i].frame != traj.anchor_observations[i].frame ||
                std::abs(traj.observations[i].t -
                         traj.anchor_observations[i].t) > 1e-9) {
                throw std::runtime_error(
                    "tracked pendulum target and anchor observations must be frame-aligned");
            }
        }
    }
    return traj;
}

void save_tracking(const Trajectory& trajectory, const std::filesystem::path& path) {
    nlohmann::json observations = nlohmann::json::array();
    for (const auto& obs : trajectory.observations) {
        observations.push_back(observation_to_json(obs));
    }
    nlohmann::json j = {
        {"version", trajectory.version},
        {"model", model_name(trajectory.model)},
        {"fps", trajectory.fps},
        {"frame_width", trajectory.frame_width},
        {"frame_height", trajectory.frame_height},
        {"observations", observations},
    };
    if (trajectory.pivot.has_value()) {
        j["reference"] = {
            {"mode", anchor_mode_name(trajectory.anchor_mode)},
            {"pivot_x", trajectory.pivot->x},
            {"pivot_y", trajectory.pivot->y},
        };
        if (trajectory.anchor_track_coverage.has_value()) {
            j["reference"]["coverage"] = *trajectory.anchor_track_coverage;
        }
    }
    if (!trajectory.anchor_observations.empty()) {
        nlohmann::json anchors = nlohmann::json::array();
        for (const auto& obs : trajectory.anchor_observations) {
            anchors.push_back(observation_to_json(obs));
        }
        j["anchor_observations"] = std::move(anchors);
    }
    write_json_file(path, j);
}

void save_reconstruction(const Reconstruction& reconstruction,
                         const std::filesystem::path& path) {
    nlohmann::json simulated = nlohmann::json::array();
    for (const auto& obs : reconstruction.simulated.observations) {
        simulated.push_back(observation_to_json(obs));
    }
    const nlohmann::json j = {
        {"version", 1},
        {"model", model_name(DynamicsModel::projectile_bounce)},
        {"parameters",
         {{"vx0", reconstruction.parameters.vx0},
          {"vy0", reconstruction.parameters.vy0},
          {"g", reconstruction.parameters.g},
          {"e", reconstruction.parameters.e}}},
        {"environment",
         {{"x0", reconstruction.environment.x0},
          {"y0", reconstruction.environment.y0},
          {"y_ground", reconstruction.environment.y_ground},
          {"dt", reconstruction.environment.dt}}},
        {"units",
         {{"position", "pixels"},
          {"time", "seconds"},
          {"velocity", "pixels_per_second"},
          {"gravity", "pixels_per_second_squared"},
          {"restitution", "dimensionless"}}},
        {"metrics",
         {{"rmse", reconstruction.rmse},
          {"mae", reconstruction.mae},
          {"rmse_x", reconstruction.rmse_x},
          {"rmse_y", reconstruction.rmse_y},
          {"normalized_rmse", reconstruction.normalized_rmse},
          {"worst_axis_normalized_rmse",
           reconstruction.worst_axis_normalized_rmse},
          {"ground_violation", reconstruction.ground_violation},
          {"quality", reconstruction.quality},
          {"ground_source", reconstruction.ground_source},
          {"n", reconstruction.n},
          {"search_generations", reconstruction.search_generations},
          {"refinement_iterations", reconstruction.refinement_iterations},
          {"iterations", reconstruction.iterations},
          {"fit_seconds", reconstruction.fit_seconds}}},
        {"simulated", simulated},
    };
    write_json_file(path, j);
}

void save_reconstruction(const PendulumReconstruction& reconstruction,
                         const std::filesystem::path& path) {
    nlohmann::json simulated = nlohmann::json::array();
    for (const auto& obs : reconstruction.simulated.observations) {
        simulated.push_back(observation_to_json(obs));
    }
    nlohmann::json anchor_path = nlohmann::json::array();
    for (const auto& obs : reconstruction.simulated.anchor_observations) {
        anchor_path.push_back(observation_to_json(obs));
    }
    const nlohmann::json j = {
        {"version", 1},
        {"model", model_name(DynamicsModel::pendulum)},
        {"parameters",
         {{"omega0", reconstruction.parameters.omega0},
          {"lambda", reconstruction.parameters.lambda},
          {"damping", reconstruction.parameters.damping}}},
        {"environment",
         {{"pivot_x", reconstruction.environment.pivot_x},
          {"pivot_y", reconstruction.environment.pivot_y},
          {"radius", reconstruction.environment.radius},
          {"theta0", reconstruction.environment.theta0},
          {"integration_step", reconstruction.environment.integration_step},
          {"reference_mode",
           anchor_mode_name(reconstruction.environment.anchor_mode)},
          {"anchor_path", anchor_path}}},
        {"units",
         {{"position", "pixels"},
          {"time", "seconds"},
          {"angle", "radians"},
          {"angular_velocity", "radians_per_second"},
          {"lambda", "per_second_squared"},
          {"damping", "per_second"}}},
        {"metrics",
         {{"rmse", reconstruction.rmse},
          {"mae", reconstruction.mae},
          {"rmse_x", reconstruction.rmse_x},
          {"rmse_y", reconstruction.rmse_y},
          {"normalized_rmse", reconstruction.normalized_rmse},
          {"robust_cost", reconstruction.robust_cost},
          {"radial_mad", reconstruction.radial_mad},
          {"angular_span", reconstruction.angular_span},
          {"pivot_adjustment", reconstruction.pivot_adjustment},
          {"anchor_track_coverage", reconstruction.anchor_track_coverage},
          {"quality", reconstruction.quality},
          {"n", reconstruction.n},
          {"search_generations", reconstruction.search_generations},
          {"refinement_iterations", reconstruction.refinement_iterations},
          {"fit_seconds", reconstruction.fit_seconds}}},
        {"simulated", simulated},
    };
    write_json_file(path, j);
}

}  // namespace phystwin
