"""Plot observed and reconstructed trajectories from the JSON boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_metrics import contact_timing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tracking")
    parser.add_argument("reconstruction")
    parser.add_argument("--output", default="results/reconstruction_preview.png")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    tracking = json.loads(Path(args.tracking).read_text(encoding="utf-8"))
    reconstruction = json.loads(
        Path(args.reconstruction).read_text(encoding="utf-8")
    )
    observed = tracking["observations"]
    simulated = reconstruction["simulated"]
    if len(observed) != len(simulated):
        raise ValueError(
            f"length mismatch: observed={len(observed)} simulated={len(simulated)}"
        )

    t = [point["t"] for point in observed]
    obs_x = [point["x"] for point in observed]
    obs_y = [point["y"] for point in observed]
    sim_x = [point["x"] for point in simulated]
    sim_y = [point["y"] for point in simulated]

    metrics = reconstruction["metrics"]
    parameters = reconstruction["parameters"]
    model = reconstruction.get("model", "projectile_bounce")
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].plot(t, obs_x, label="observed", linewidth=1.5)
    axes[0].plot(t, sim_x, "--", label="simulated", linewidth=1.5)
    axes[0].set_title("horizontal position")
    axes[0].set_xlabel("t (s)")
    axes[0].set_ylabel("x (px)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, obs_y, label="observed", linewidth=1.5)
    axes[1].plot(t, sim_y, "--", label="simulated", linewidth=1.5)
    axes[1].set_title("vertical position (+y down)")
    axes[1].set_xlabel("t (s)")
    axes[1].set_ylabel("y (px)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(obs_x, obs_y, label="observed", linewidth=1.5)
    axes[2].plot(sim_x, sim_y, "--", label="simulated", linewidth=1.5)
    axes[2].invert_yaxis()
    axes[2].set_aspect("equal", adjustable="datalim")
    heading = args.title + "\n" if args.title else ""
    if model == "pendulum":
        parameter_line = (
            f"lambda={parameters['lambda']:.3f} s^-2, "
            f"damping={parameters['damping']:.3f} s^-1"
        )
    else:
        parameter_line = (
            f"g={parameters['g']:.2f} px/s², e={parameters['e']:.3f}"
        )
    axes[2].set_title(
        f"{heading}"
        f"RMSE {metrics['rmse']:.2f} px ({metrics['quality']})\n"
        f"x={metrics['rmse_x']:.2f}, y={metrics['rmse_y']:.2f} px\n"
        f"{parameter_line}"
    )
    axes[2].set_xlabel("x (px)")
    axes[2].set_ylabel("y (px)")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=130)
    plt.close(figure)
    print(f"wrote {output}")
    if model == "projectile_bounce":
        timing = contact_timing(tracking, reconstruction)
        print(f"observed contact frames: {timing['observed_contact_frames']}")
        print(f"simulated contact frames: {timing['simulated_contact_frames']}")
        if timing.get("paired"):
            print(
                f"mean contact timing error: {timing['mean_error_frames']:.2f} frames "
                f"({timing['mean_error_ms']:.2f} ms)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
