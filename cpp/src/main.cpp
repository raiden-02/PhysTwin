#include "phystwin/fitter.hpp"
#include "phystwin/io.hpp"
#include "phystwin/pendulum.hpp"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kVersion = "0.1.0";

void print_usage(std::ostream& out) {
    out << "PhysTwin " << kVersion << "\n"
        << "Usage:\n"
        << "  phystwin inspect <tracking.json>\n"
        << "  phystwin fit <tracking.json> [--output reconstruction.json]"
           " [--ground-y PIXELS]\n"
        << "  phystwin --help\n"
        << "  phystwin --version\n\n"
        << "tracking.json model selects projectile_bounce (default) or pendulum.\n"
        << "Pendulum input requires a fixed pivot or frame-aligned tracked anchor.\n"
        << "--ground-y is the object's center y at ground contact. If omitted,\n"
        << "the largest observed centroid y is used.\n";
}

int inspect(const std::filesystem::path& path) {
    const phystwin::Trajectory traj = phystwin::load_tracking(path);
    const auto& first = traj.observations.front();
    const auto& last = traj.observations.back();
    std::cout << "file: " << path.string() << "\n"
              << "version: " << traj.version << "\n"
              << "model: " << phystwin::model_name(traj.model) << "\n"
              << "fps: " << traj.fps << "\n"
              << "frame_size: " << traj.frame_width << "x" << traj.frame_height << "\n"
              << "n_observations: " << traj.observations.size() << "\n"
              << "anchor_mode: " << phystwin::anchor_mode_name(traj.anchor_mode)
              << "\n"
              << "n_anchor_observations: " << traj.anchor_observations.size()
              << "\n"
              << "first: frame=" << first.frame << " t=" << first.t << " x=" << first.x
              << " y=" << first.y << "\n"
              << "last:  frame=" << last.frame << " t=" << last.t << " x=" << last.x
              << " y=" << last.y << "\n";
    return 0;
}

double parse_number(const std::string& text, std::string_view option) {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size()) {
        throw std::invalid_argument(std::string(option) + " expects a number");
    }
    return value;
}

int fit(const std::filesystem::path& path,
        const std::filesystem::path& output,
        const std::optional<double> ground_y) {
    const phystwin::Trajectory traj = phystwin::load_tracking(path);
    if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path());
    }
    if (traj.model == phystwin::DynamicsModel::pendulum) {
        if (ground_y.has_value()) {
            throw std::invalid_argument("--ground-y is only valid for projectile_bounce");
        }
        const phystwin::PendulumReconstruction reconstruction =
            phystwin::PendulumFitter{}.fit(traj);
        phystwin::save_reconstruction(reconstruction, output);
        std::cout << std::fixed << std::setprecision(6)
                  << "input: " << path.string() << "\n"
                  << "output: " << output.string() << "\n"
                  << "model: pendulum\n"
                  << "anchor_mode: "
                  << phystwin::anchor_mode_name(
                         reconstruction.environment.anchor_mode)
                  << "\n"
                  << "anchor_coverage: "
                  << reconstruction.anchor_track_coverage * 100.0 << "%\n"
                  << "observations: " << reconstruction.n << "\n"
                  << "pivot: " << reconstruction.environment.pivot_x << ", "
                  << reconstruction.environment.pivot_y << " px\n"
                  << "radius: " << reconstruction.environment.radius << " px\n"
                  << "omega0: " << reconstruction.parameters.omega0 << " rad/s\n"
                  << "lambda: " << reconstruction.parameters.lambda << " 1/s^2\n"
                  << "damping: " << reconstruction.parameters.damping << " 1/s\n"
                  << "RMSE: " << reconstruction.rmse << " px\n"
                  << "RMSE_x: " << reconstruction.rmse_x << " px\n"
                  << "RMSE_y: " << reconstruction.rmse_y << " px\n"
                  << "radial_MAD: " << reconstruction.radial_mad << " px\n"
                  << "angular_span: " << reconstruction.angular_span << " rad\n"
                  << "pivot_adjustment: " << reconstruction.pivot_adjustment
                  << " px\n"
                  << "quality: " << reconstruction.quality << "\n"
                  << "fit_seconds: " << reconstruction.fit_seconds << "\n";
        if (reconstruction.quality == "poor") {
            std::cerr << "poor pendulum fit: do not treat the parameters as credible.\n";
            return 2;
        }
        if (reconstruction.quality == "fair") {
            std::cerr << "warning: fair pendulum fit. Inspect the synchronized twin.\n";
        }
        return 0;
    }

    phystwin::FitOptions options;
    options.ground_y = ground_y;
    const phystwin::Reconstruction reconstruction =
        phystwin::Fitter{}.fit(traj, options);

    phystwin::save_reconstruction(reconstruction, output);

    std::cout << std::fixed << std::setprecision(6)
              << "input: " << path.string() << "\n"
              << "output: " << output.string() << "\n"
              << "model: projectile_bounce\n"
              << "observations: " << reconstruction.n << "\n"
              << "ground_y: " << reconstruction.environment.y_ground
              << " (" << reconstruction.ground_source << ")\n"
              << "vx0: " << reconstruction.parameters.vx0 << " px/s\n"
              << "vy0: " << reconstruction.parameters.vy0 << " px/s\n"
              << "gravity_scale: " << reconstruction.parameters.g << " px/s^2\n"
              << "restitution: " << reconstruction.parameters.e << "\n"
              << "RMSE: " << reconstruction.rmse << " px\n"
              << "RMSE_x: " << reconstruction.rmse_x << " px\n"
              << "RMSE_y: " << reconstruction.rmse_y << " px\n"
              << "MAE: " << reconstruction.mae << " px\n"
              << "normalized_RMSE: " << reconstruction.normalized_rmse * 100.0
              << "% of trajectory extent\n"
              << "worst_axis_error: "
              << reconstruction.worst_axis_normalized_rmse * 100.0
              << "% of that axis travel\n"
              << "ground_violation: " << reconstruction.ground_violation
              << " px\n"
              << "quality: " << reconstruction.quality << "\n"
              << "search_generations: " << reconstruction.search_generations
              << " (fixed DE budget)\n"
              << "refinement_iterations: " << reconstruction.refinement_iterations
              << " (step-halving safety net)\n"
              << "fit_seconds: " << reconstruction.fit_seconds << "\n";

    if (reconstruction.quality == "poor") {
        std::cerr
            << "poor fit: error is too large or observations cross the chosen ground. "
               "Do not treat the fitted parameters as credible.\n";
        return 2;
    }
    if (reconstruction.quality == "fair") {
        std::cerr
            << "warning: fair fit. Inspect observed vs simulated motion before "
               "using the parameters.\n";
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty() || args[0] == "--help" || args[0] == "-h") {
        print_usage(std::cout);
        return args.empty() ? 1 : 0;
    }
    if (args[0] == "--version" || args[0] == "-v") {
        std::cout << kVersion << "\n";
        return 0;
    }

    try {
        if (args[0] == "inspect") {
            if (args.size() != 2) {
                print_usage(std::cerr);
                return 1;
            }
            return inspect(args[1]);
        }
        if (args[0] == "fit") {
            if (args.size() < 2) {
                print_usage(std::cerr);
                return 1;
            }
            std::filesystem::path output = "reconstruction.json";
            std::optional<double> ground_y;
            for (std::size_t i = 2; i < args.size(); ++i) {
                if (args[i] == "--output" && i + 1 < args.size()) {
                    output = args[++i];
                } else if (args[i] == "--ground-y" && i + 1 < args.size()) {
                    ground_y = parse_number(args[++i], "--ground-y");
                } else {
                    std::cerr << "unknown argument: " << args[i] << "\n";
                    print_usage(std::cerr);
                    return 1;
                }
            }
            return fit(args[1], output, ground_y);
        }
        std::cerr << "unknown command: " << args[0] << "\n";
        print_usage(std::cerr);
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
