#pragma once

#include "phystwin/pendulum.hpp"
#include "phystwin/physics_model.hpp"
#include "phystwin/trajectory.hpp"

#include <filesystem>

namespace phystwin {

Trajectory load_tracking(const std::filesystem::path& path);
void save_tracking(const Trajectory& trajectory, const std::filesystem::path& path);

void save_reconstruction(const Reconstruction& reconstruction,
                         const std::filesystem::path& path);
void save_reconstruction(const PendulumReconstruction& reconstruction,
                         const std::filesystem::path& path);

}  // namespace phystwin
