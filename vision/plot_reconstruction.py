"""Plot observed and reconstructed trajectories from the JSON boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def contact_frames(points: list[dict]) -> list[int]:
    """Detect high-y local maxima for a simple bounce-timing check."""
    y = [point["y"] for point in points]
    threshold = max(y) - 0.12 * (max(y) - min(y))
    contacts: list[int] = []
    for index in range(3, len(y) - 3):
        if y[index] != max(y[index - 3 : index + 4]) or y[index] < threshold:
            continue
        if not contacts or index - contacts[-1] >= 8:
            contacts.append(index)
        elif y[index] > y[contacts[-1]]:
            contacts[-1] = index
    return contacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tracking")
    parser.add_argument("reconstruction")
    parser.add_argument("--output", default="results/reconstruction_preview.png")
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
    axes[2].set_title(
        f"RMSE {metrics['rmse']:.2f} px ({metrics['quality']})\n"
        f"x={metrics['rmse_x']:.2f}, y={metrics['rmse_y']:.2f} px\n"
        f"g={parameters['g']:.2f} px/s², e={parameters['e']:.3f}"
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
    observed_contacts = contact_frames(observed)
    simulated_contacts = contact_frames(simulated)
    paired = min(len(observed_contacts), len(simulated_contacts))
    if paired:
        errors = [
            abs(observed_contacts[index] - simulated_contacts[index])
            for index in range(paired)
        ]
        mean_frames = sum(errors) / paired
        mean_ms = mean_frames * 1000.0 / float(tracking["fps"])
        print(f"observed contact frames: {observed_contacts}")
        print(f"simulated contact frames: {simulated_contacts}")
        print(
            f"mean contact timing error: {mean_frames:.2f} frames "
            f"({mean_ms:.2f} ms)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
