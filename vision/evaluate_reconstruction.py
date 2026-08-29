#!/usr/bin/env python3
"""Evaluate one human SceneObservation against EMDB or a synthetic check."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.reconstruction.contracts import load_contract
from vision.reconstruction.evaluation import (
    EMDB_DATASET_LICENSE,
    evaluate_observation,
    load_emdb_reference,
    reference_from_observation,
    save_evaluation,
)
from vision.reconstruction.humans import HUMANS_EXTENSION
from vision.reconstruction.tram import (
    TramHumanAdapter,
    normalize_human_options,
)
from vision.reconstruction.humans import HumanReconstructionRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, help="predicted SceneObservation JSON")
    parser.add_argument("--emdb-sequence", type=Path, help="approved EMDB sequence folder")
    parser.add_argument("--smpl-model-root", type=Path, help="registered SMPL/SMPL-X model root")
    parser.add_argument("--person-id", help="humans.v1 person id, defaults to the first person")
    parser.add_argument(
        "--accept-emdb-license",
        action="store_true",
        help="confirm this use follows the EMDB non-commercial academic license",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="run a synthetic alignment regression instead of claiming EMDB measurement",
    )
    parser.add_argument(
        "--fixture-body-offset-m",
        type=float,
        default=0.05,
        help="known X offset applied to the synthetic prediction",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "evaluation3d" / "latest",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fixture:
        if args.observation or args.emdb_sequence or args.smpl_model_root:
            raise SystemExit("--fixture cannot be combined with EMDB inputs")
        observation, reference = _fixture_case(args.fixture_body_offset_m)
    else:
        if not args.observation or not args.emdb_sequence or not args.smpl_model_root:
            raise SystemExit(
                "EMDB evaluation requires --observation, --emdb-sequence, "
                "--smpl-model-root, and --accept-emdb-license"
            )
        if not args.accept_emdb_license:
            raise SystemExit(
                f"EMDB dataset terms: {EMDB_DATASET_LICENSE}. "
                "Pass --accept-emdb-license only if your approved use complies."
            )
        observation = load_contract(args.observation.resolve())
        reference = load_emdb_reference(
            args.emdb_sequence.resolve(),
            args.smpl_model_root.resolve(),
        )

    evaluation = evaluate_observation(
        observation,
        reference,
        person_id=args.person_id,
    )
    output = args.output.resolve()
    report = save_evaluation(output, evaluation)
    print(
        json.dumps(
            {
                "status": report["status"],
                "evaluation_id": report["evaluation_id"],
                "output": str(output),
                "matched_good_samples": report["coverage"]["matched_good_samples"],
                "metrics": report["metrics"],
            },
            indent=2,
        )
    )
    return 0


def _fixture_case(body_offset_m: float):
    with tempfile.TemporaryDirectory() as temp:
        adapter = TramHumanAdapter()
        output = adapter.reconstruct_humans(
            HumanReconstructionRequest(
                options=normalize_human_options(
                    {
                        "source": "walk_fixture",
                        "walk_frames": 12,
                        "fps": 24.0,
                    }
                )
            ),
            Path(temp),
        )
    reference_observation = output.observation
    reference = reference_from_observation(reference_observation)
    prediction = copy.deepcopy(reference_observation)
    humans = prediction["extensions"][HUMANS_EXTENSION]
    for person in humans["people"]:
        for sample in person["samples"]:
            for joint in sample["joints"]:
                joint[0] += body_offset_m
            sample["root"] = list(sample["joints"][0])
    prediction["observation_id"] = "p3-offset-prediction"
    return prediction, reference


if __name__ == "__main__":
    raise SystemExit(main())
