#include "phystwin/fitter.hpp"
#include "phystwin/simulator.hpp"

#include <iostream>
#include <vector>

// Checkpoint 1 will generate a trajectory from known (vx0, vy0, g, e),
// fit it, and assert recovery within an explicit tolerance.
// Checkpoint 0 only proves the test target links against the core library.
int main() {
    const phystwin::Simulator sim;
    const phystwin::Trajectory empty = sim.run({}, {}, std::vector<double>{});
    if (!empty.observations.empty()) {
        std::cerr << "synthetic_fit placeholder expected an empty simulator\n";
        return 1;
    }
    std::cout << "synthetic_fit: placeholder (parameter recovery is Checkpoint 1)\n";
    return 0;
}
