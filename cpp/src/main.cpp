#include "phystwin/fitter.hpp"
#include "phystwin/io.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kVersion = "0.1.0";

void print_usage(std::ostream& out) {
    out << "PhysTwin " << kVersion << "\n"
        << "Usage:\n"
        << "  phystwin inspect <tracking.json>\n"
        << "  phystwin fit <tracking.json> [--output reconstruction.json]\n"
        << "  phystwin --help\n"
        << "  phystwin --version\n";
}

int inspect(const std::filesystem::path& path) {
    const phystwin::Trajectory traj = phystwin::load_tracking(path);
    const auto& first = traj.observations.front();
    const auto& last = traj.observations.back();
    std::cout << "file: " << path.string() << "\n"
              << "version: " << traj.version << "\n"
              << "fps: " << traj.fps << "\n"
              << "frame_size: " << traj.frame_width << "x" << traj.frame_height << "\n"
              << "n_observations: " << traj.observations.size() << "\n"
              << "first: frame=" << first.frame << " t=" << first.t << " x=" << first.x
              << " y=" << first.y << "\n"
              << "last:  frame=" << last.frame << " t=" << last.t << " x=" << last.x
              << " y=" << last.y << "\n";
    return 0;
}

int fit(const std::filesystem::path& path, const std::filesystem::path&) {
    const phystwin::Trajectory traj = phystwin::load_tracking(path);
    std::cerr << "loaded " << traj.observations.size() << " observations from " << path.string()
              << "\n"
              << "phystwin fit is not implemented yet (Checkpoint 1)\n";
    return 2;
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
            for (std::size_t i = 2; i < args.size(); ++i) {
                if (args[i] == "--output" && i + 1 < args.size()) {
                    output = args[++i];
                } else {
                    std::cerr << "unknown argument: " << args[i] << "\n";
                    print_usage(std::cerr);
                    return 1;
                }
            }
            return fit(args[1], output);
        }
        std::cerr << "unknown command: " << args[0] << "\n";
        print_usage(std::cerr);
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
