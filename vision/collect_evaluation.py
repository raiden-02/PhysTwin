"""Write docs/evaluation.json from measured case files. Does not invent numbers."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_metrics import contact_timing


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16")
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    return data.decode("utf-8")


def _load_json(path: str | None) -> dict | None:
    if not path:
        return None
    file = Path(path)
    if not file.exists():
        return None
    return json.loads(file.read_text(encoding="utf-8"))


def _parse_synthetic_stdout(text: str) -> dict:
    rows = {}
    for name in ("vx0", "vy0", "g", "e"):
        match = re.search(
            rf"^{name}\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)",
            text,
            re.MULTILINE,
        )
        if match:
            rows[name] = {
                "actual": float(match.group(1)),
                "recovered": float(match.group(2)),
                "abs_error": float(match.group(3)),
            }
    rmse_match = re.search(r"^RMSE:\s+([-\d.eE+]+)", text, re.MULTILINE)
    mae_match = re.search(r"^MAE:\s+([-\d.eE+]+)", text, re.MULTILINE)
    perturbed = re.search(r"^perturbed RMSE:\s+([-\d.eE+]+)", text, re.MULTILINE)
    iterations = re.search(r"^optimizer iterations:\s+(\d+)", text, re.MULTILINE)
    search_gen = re.search(r"^search_generations:\s+(\d+)", text, re.MULTILINE)
    refine = re.search(r"^refinement_iterations:\s+(\d+)", text, re.MULTILINE)
    fit_time = re.search(r"^fit time:\s+([-\d.eE+]+)", text, re.MULTILINE)
    return {
        "parameters": rows,
        "rmse_px": float(rmse_match.group(1)) if rmse_match else None,
        "mae_px": float(mae_match.group(1)) if mae_match else None,
        "perturbed_rmse_px": float(perturbed.group(1)) if perturbed else None,
        "search_generations": int(search_gen.group(1)) if search_gen else None,
        "refinement_iterations": int(refine.group(1)) if refine else None,
        "iterations": int(iterations.group(1)) if iterations else None,
        "fit_seconds": float(fit_time.group(1)) if fit_time else None,
        "stdout": text.strip(),
    }


def _parse_pendulum_synthetic_stdout(text: str) -> dict:
    rows = {}
    for name in ("omega0", "lambda", "damping"):
        match = re.search(
            rf"^{name}\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)",
            text,
            re.MULTILINE,
        )
        if match:
            rows[name] = {
                "actual": float(match.group(1)),
                "recovered": float(match.group(2)),
                "abs_error": float(match.group(3)),
            }
    patterns = {
        "rmse_px": r"^RMSE:\s+([-\d.eE+]+)",
        "noisy_outlier_rmse_px": r"^noisy/outlier RMSE:\s+([-\d.eE+]+)",
        "noisy_lambda": r"^noisy lambda:\s+([-\d.eE+]+)",
        "noisy_damping": r"^noisy damping:\s+([-\d.eE+]+)",
        "fit_seconds": r"^fit time:\s+([-\d.eE+]+)",
    }
    result = {"parameters": rows}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.MULTILINE)
        result[key] = float(match.group(1)) if match else None
    degenerate = re.search(r"^degenerate_checks:\s+(\d+)", text, re.MULTILINE)
    generations = re.search(r"^search_generations:\s+(\d+)", text, re.MULTILINE)
    result["degenerate_checks"] = int(degenerate.group(1)) if degenerate else None
    result["search_generations"] = (
        int(generations.group(1)) if generations else None
    )
    result["stdout"] = text.strip()
    return result


def _video_case(item: dict) -> dict:
    tracking = _load_json(item.get("tracking"))
    reconstruction = _load_json(item.get("reconstruction"))
    raw = _load_json(item.get("tracking_raw"))
    if tracking is None or reconstruction is None:
        raise FileNotFoundError(
            f"missing tracking/reconstruction for case {item.get('id')}"
        )
    model = reconstruction.get("model", tracking.get("model", "projectile_bounce"))
    timing = (
        contact_timing(tracking, reconstruction)
        if model == "projectile_bounce"
        else None
    )
    metrics = reconstruction["metrics"]
    parameters = reconstruction["parameters"]
    result = {
        "id": item["id"],
        "title": item["title"],
        "kind": item["kind"],
        "model": model,
        "video": item.get("video"),
        "point": item.get("point"),
        "pivot": item.get("pivot"),
        "source": item.get("source"),
        "notes": item.get("notes"),
        "fps": tracking["fps"],
        "frame_width": tracking["frame_width"],
        "frame_height": tracking["frame_height"],
        "n_observations": len(tracking["observations"]),
        "parameters": parameters,
        "metrics": metrics,
        "contact_timing": timing,
        "artifacts": {
            "tracking": item.get("tracking"),
            "reconstruction": item.get("reconstruction"),
            "plot": item.get("plot"),
            "overlay": item.get("overlay"),
            "gif": item.get("gif") if item.get("gif") and Path(item["gif"]).exists() else None,
            "still": item.get("still"),
        },
    }
    if raw:
        seconds = raw.get("end_to_end_seconds", raw.get("inference_seconds"))
        fps_rate = raw.get("end_to_end_fps", raw.get("inference_fps"))
        result["tracking_runtime"] = {
            "device": raw.get("device"),
            "n_frames": raw.get("n_frames"),
            "skipped_empty_masks": raw.get("skipped_empty_masks"),
            "end_to_end_seconds": seconds,
            "end_to_end_fps": fps_rate,
            "timing_includes": raw.get(
                "timing_includes", "model_load,init_state,jpeg_decode,propagate"
            ),
            "point": raw.get("point"),
            "pivot": raw.get("pivot"),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/cases/manifest.json")
    parser.add_argument("--output", default="docs/evaluation.json")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    cases = []
    for item in manifest["cases"]:
        if item["kind"] in {"cpp_synthetic", "cpp_pendulum_synthetic"}:
            stdout_path = Path(item["stdout"])
            text = _read_text(stdout_path)
            parsed = (
                _parse_pendulum_synthetic_stdout(text)
                if item["kind"] == "cpp_pendulum_synthetic"
                else _parse_synthetic_stdout(text)
            )
            cases.append(
                {
                    "id": item["id"],
                    "title": item["title"],
                    "kind": item["kind"],
                    "notes": item.get("notes"),
                    **parsed,
                }
            )
        else:
            cases.append(_video_case(item))

    payload = {
        "date": date.today().isoformat(),
        "notes": (
            "Numbers copied from measured reconstruction.json files and "
            "synthetic_fit stdout. Re-run scripts/run-eval.ps1 to refresh."
        ),
        "cases": cases,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} cases={len(cases)}")
    for case in cases:
        kind = case["kind"]
        if kind in {"cpp_synthetic", "cpp_pendulum_synthetic"}:
            print(f"  {case['id']}: RMSE={case.get('rmse_px')} px")
        else:
            print(
                f"  {case['id']}: RMSE={case['metrics']['rmse']:.4f} px "
                f"quality={case['metrics']['quality']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
