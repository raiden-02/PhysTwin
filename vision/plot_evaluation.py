"""One figure with observed vs simulated trajectories for every evaluation case."""

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
    parser.add_argument("--manifest", default="results/cases/manifest.json")
    parser.add_argument("--output", default="docs/demo/observed_vs_simulated.png")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    cases = []
    for item in manifest["cases"]:
        if "tracking" not in item:
            continue
        if item.get("kind") == "recorded_video_failure":
            continue
        tracking = json.loads(Path(item["tracking"]).read_text(encoding="utf-8"))
        reconstruction = json.loads(
            Path(item["reconstruction"]).read_text(encoding="utf-8")
        )
        cases.append(
            {
                "title": item["title"],
                "tracking": tracking,
                "reconstruction": reconstruction,
                "timing": contact_timing(tracking, reconstruction),
            }
        )
    if not cases:
        raise RuntimeError("no plottable cases in the evaluation manifest")

    n = len(cases)
    figure, axes = plt.subplots(2, n, figsize=(5.2 * n, 8.4), squeeze=False)
    figure.suptitle("PhysTwin: observed vs simulated trajectories", fontsize=14, y=0.98)

    for col, case in enumerate(cases):
        observed = case["tracking"]["observations"]
        simulated = case["reconstruction"]["simulated"]
        metrics = case["reconstruction"]["metrics"]
        parameters = case["reconstruction"]["parameters"]
        t = [point["t"] for point in observed]
        obs_x = [point["x"] for point in observed]
        obs_y = [point["y"] for point in observed]
        sim_x = [point["x"] for point in simulated]
        sim_y = [point["y"] for point in simulated]

        xy = axes[0][col]
        xy.plot(obs_x, obs_y, color="#1f77b4", linewidth=1.8, label="observed")
        xy.plot(sim_x, sim_y, color="#ff7f0e", linestyle="--", linewidth=1.8, label="simulated")
        xy.invert_yaxis()
        xy.set_aspect("equal", adjustable="datalim")
        xy.set_xlabel("x (px)")
        xy.set_ylabel("y (px, +down)")
        xy.grid(alpha=0.3)
        xy.legend(loc="best", fontsize=8)
        bounce = ""
        if case["timing"].get("paired"):
            bounce = (
                f"bounce err {case['timing']['mean_error_frames']:.2f} fr "
                f"({case['timing']['mean_error_ms']:.1f} ms)"
            )
        xy.set_title(
            f"{case['title']}\n"
            f"RMSE {metrics['rmse']:.2f} px · {metrics['quality']}\n"
            f"g={parameters['g']:.1f} px/s²  e={parameters['e']:.3f}\n"
            f"{bounce}".rstrip(),
            fontsize=10,
        )

        yt = axes[1][col]
        yt.plot(t, obs_y, color="#1f77b4", linewidth=1.8, label="observed y")
        yt.plot(t, sim_y, color="#ff7f0e", linestyle="--", linewidth=1.8, label="simulated y")
        yt.set_xlabel("t (s)")
        yt.set_ylabel("y (px, +down)")
        yt.grid(alpha=0.3)
        yt.legend(loc="best", fontsize=8)
        yt.invert_yaxis()
        yt.set_title("vertical position vs time", fontsize=10)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=140)
    plt.close(figure)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
