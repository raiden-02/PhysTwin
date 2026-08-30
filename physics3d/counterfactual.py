"""Clone a fitted PhysicalScene and reroll it under an edited parameter."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from physics3d.newton_runtime import simulate_physical_scene
from vision.reconstruction.contracts import (
    canonical_json_bytes,
    validate_physical_scene,
    validate_rollout_source,
)


MOON_GRAVITY_M_S2 = 1.62
PRESETS = {
    "moon": MOON_GRAVITY_M_S2,
    "fitted": None,
    "double": None,
}


def clone_scene_with_gravity(
    fitted_scene: Mapping[str, Any],
    *,
    gravity_magnitude_m_s2: float,
    label: str,
) -> dict[str, Any]:
    """Copy a fitted scene and change only gravity magnitude. Direction stays -Y."""

    if gravity_magnitude_m_s2 <= 0.0:
        raise ValueError("gravity_magnitude_m_s2 must be > 0")
    scene = copy.deepcopy(dict(fitted_scene))
    original = abs(float(scene["world"]["gravity_m_s2"][1]))
    scene["world"]["gravity_m_s2"] = [0.0, -float(gravity_magnitude_m_s2), 0.0]
    scene["scene_id"] = f"{scene['scene_id']}-{label}"
    provenance = dict(scene.get("provenance") or {})
    provenance["counterfactual"] = {
        "source_scene_id": fitted_scene["scene_id"],
        "source_scene_sha256": hashlib.sha256(canonical_json_bytes(fitted_scene)).hexdigest(),
        "parameter": "gravity_magnitude_m_s2",
        "original_value": original,
        "counterfactual_value": float(gravity_magnitude_m_s2),
        "observed": False,
        "note": "Simulated hypothesis. This motion was not observed.",
    }
    scene["provenance"] = provenance
    return dict(validate_physical_scene(scene))


def rollout_counterfactual(
    fitted_scene: Mapping[str, Any],
    *,
    gravity_magnitude_m_s2: float,
    label: str,
    output_dir: Path,
    repeat_check: bool = False,
) -> dict[str, Any]:
    """Run Newton on the edited scene and write provenance-stamped artifacts."""

    scene = clone_scene_with_gravity(
        fitted_scene,
        gravity_magnitude_m_s2=gravity_magnitude_m_s2,
        label=label,
    )
    rollout = simulate_physical_scene(scene, repeat_check=repeat_check)
    validate_rollout_source(rollout, scene)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = output_dir / f"counterfactual_{label}_physical_scene.json"
    rollout_path = output_dir / f"counterfactual_{label}_simulated_world_state.json"
    _write_json(scene_path, scene)
    _write_json(rollout_path, rollout)
    report = {
        "schema": "phystwin.counterfactual_rollout",
        "version": 1,
        "label": label,
        "parameter": "gravity_magnitude_m_s2",
        "original_value": scene["provenance"]["counterfactual"]["original_value"],
        "counterfactual_value": gravity_magnitude_m_s2,
        "observed": False,
        "source_fitted_scene_sha256": scene["provenance"]["counterfactual"]["source_scene_sha256"],
        "counterfactual_scene_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        "counterfactual_rollout_sha256": hashlib.sha256(rollout_path.read_bytes()).hexdigest(),
        "outputs": {
            "physical_scene": scene_path.name,
            "simulated_world_state": rollout_path.name,
        },
    }
    _write_json(output_dir / f"counterfactual_{label}.json", report)
    return {
        "report": report,
        "physical_scene": scene,
        "rollout": rollout,
    }


def main() -> int:
    import argparse
    import json
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fitted-scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--moon", action="store_true", default=True)
    args = parser.parse_args()
    scene = json.loads(args.fitted_scene.read_text(encoding="utf-8"))
    result = rollout_counterfactual(
        scene,
        gravity_magnitude_m_s2=MOON_GRAVITY_M_S2,
        label="moon",
        output_dir=args.output.resolve(),
        repeat_check=False,
    )
    print(json.dumps(result["report"], indent=2))
    return 0


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
