#include "phystwin/io.hpp"

#include <nlohmann/json.hpp>

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
    traj.fps = j.at("fps").get<double>();
    if (j.contains("frame_width")) {
        traj.frame_width = j.at("frame_width").get<int>();
    }
    if (j.contains("frame_height")) {
        traj.frame_height = j.at("frame_height").get<int>();
    }
    traj.observations.reserve(j.at("observations").size());
    for (const auto& item : j.at("observations")) {
        traj.observations.push_back(observation_from_json(item));
    }
    return traj;
}

void save_tracking(const Trajectory& trajectory, const std::filesystem::path& path) {
    nlohmann::json observations = nlohmann::json::array();
    for (const auto& obs : trajectory.observations) {
        observations.push_back(observation_to_json(obs));
    }
    const nlohmann::json j = {
        {"version", trajectory.version},
        {"fps", trajectory.fps},
        {"frame_width", trajectory.frame_width},
        {"frame_height", trajectory.frame_height},
        {"observations", observations},
    };
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
          {"iterations", reconstruction.iterations},
          {"fit_seconds", reconstruction.fit_seconds}}},
        {"simulated", simulated},
    };
    write_json_file(path, j);
}

}  // namespace phystwin
