#!/usr/bin/env python3
"""Run the P1 DA3 reconstruction adapter and publish a cached SceneObservation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.reconstruction.adapter import ReconstructionRequest, VideoInput, reconstruction_cache_key
from vision.reconstruction.cache import cache_entry, is_complete, load_cached_observation, publish_observation, sha256_file
from vision.reconstruction.da3 import Da3ReconstructionAdapter, make_descriptor, normalize_options, resolve_weights_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="local video path")
    parser.add_argument("--source-id", default="video0")
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--force", action="store_true", help="ignore an existing complete cache entry")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video = args.video.resolve()
    if not video.is_file():
        raise SystemExit(f"video not found: {video}")
    options = normalize_options(
        {
            "start_s": args.start_s,
            "duration_s": args.duration_s,
            "max_frames": args.max_frames,
        }
    )
    print("resolving DA3-BASE weights hash...", file=sys.stderr)
    weights = resolve_weights_sha256()
    descriptor = make_descriptor(weights)
    request = ReconstructionRequest(
        inputs=(VideoInput(args.source_id, video, sha256_file(video)),),
        options=options,
    )
    key = reconstruction_cache_key(descriptor, request)
    entry = cache_entry(ROOT, key)
    if is_complete(entry) and not args.force:
        observation = load_cached_observation(entry)
        print(json.dumps({"cache": "hit", "key": key, "path": str(entry)}, indent=2))
        print(json.dumps(observation["provenance"]["runtime"], indent=2))
        return 0

    adapter = Da3ReconstructionAdapter(weights)
    observation = publish_observation(
        entry,
        lambda work_dir: adapter.reconstruct(request, work_dir).observation,
    )
    print(json.dumps({"cache": "miss", "key": key, "path": str(entry)}, indent=2))
    print(json.dumps(observation["provenance"]["runtime"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
