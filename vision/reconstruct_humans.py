#!/usr/bin/env python3
"""Map TRAM-native body/camera output into a cached SceneObservation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.reconstruction.cache import (
    humans_cache_entry,
    is_complete,
    load_cached_observation,
    publish_observation,
    sha256_file,
)
from vision.reconstruction.contracts import load_contract
from vision.reconstruction.humans import HumanReconstructionRequest, human_cache_key
from vision.reconstruction.tram import (
    TramHumanAdapter,
    TramUnavailableError,
    make_descriptor,
    normalize_human_options,
    write_projected_skeleton_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-fixture",
        action="store_true",
        help="convert the committed TRAM c2w fixture (2 frames)",
    )
    parser.add_argument(
        "--walk-fixture",
        action="store_true",
        help="build a 12-frame synthetic walk in the same TRAM camera convention",
    )
    parser.add_argument("--tram-dir", type=Path, help="official TRAM results/<seq> folder")
    parser.add_argument(
        "--observation",
        type=Path,
        help="existing SceneObservation to keep (P1 world). Joints are lifted through its poses.",
    )
    parser.add_argument("--video", type=Path, help="source video used only for content identity")
    parser.add_argument("--write-video", type=Path, help="draw the skeleton back onto a debug mp4")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if sum(bool(item) for item in (args.from_fixture, args.walk_fixture, args.tram_dir)) != 1:
        raise SystemExit("choose exactly one of --from-fixture, --walk-fixture, or --tram-dir")

    if args.tram_dir:
        source = "tram_dir"
        tram_dir = str(args.tram_dir.resolve())
    elif args.walk_fixture:
        source = "walk_fixture"
        tram_dir = None
    else:
        source = "fixture"
        tram_dir = None

    options = normalize_human_options({"source": source, "tram_dir": tram_dir})
    parent = load_contract(args.observation) if args.observation else None
    video_hash = sha256_file(args.video.resolve()) if args.video else None
    request = HumanReconstructionRequest(
        options=options,
        parent_observation=parent,
        video_sha256=video_hash,
    )
    descriptor = make_descriptor()
    key = human_cache_key(descriptor, request)
    entry = humans_cache_entry(ROOT, key)
    if is_complete(entry) and not args.force:
        observation = load_cached_observation(entry)
        print(json.dumps({"cache": "hit", "key": key, "path": str(entry)}, indent=2))
    else:
        adapter = TramHumanAdapter()
        try:
            observation = publish_observation(
                entry,
                lambda work_dir: adapter.reconstruct_humans(request, work_dir).observation,
            )
        except TramUnavailableError as error:
            raise SystemExit(str(error)) from error
        print(json.dumps({"cache": "miss", "key": key, "path": str(entry)}, indent=2))

    if args.write_video:
        write_projected_skeleton_video(args.write_video, observation)
        print(json.dumps({"video": str(args.write_video.resolve())}, indent=2))

    people = observation["extensions"]["phystwin.humans.v1"]["people"]
    print(
        json.dumps(
            {
                "observation_id": observation["observation_id"],
                "samples": len(observation["timeline"]["samples"]),
                "people": len(people),
                "scale": observation["coordinates"]["scale"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    if os.environ.get("PHYSTWIN_TRAM_ROOT") and "--help" in sys.argv:
        print(f"PHYSTWIN_TRAM_ROOT={os.environ['PHYSTWIN_TRAM_ROOT']}", file=sys.stderr)
    raise SystemExit(main())
