"""Content-addressed reconstruction cache with atomic publish."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .contracts import ContractError, load_contract, validate_scene_observation

COMPLETE_NAME = "COMPLETE"
DOCUMENT_NAME = "scene_observation.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def cache_entry(root: Path, cache_key: str) -> Path:
    return root / "results" / "cache" / "reconstruction" / cache_key


def is_complete(entry: Path) -> bool:
    marker = entry / COMPLETE_NAME
    document = entry / DOCUMENT_NAME
    if not marker.is_file() or not document.is_file():
        return False
    try:
        load_contract(document)
    except (OSError, ContractError):
        return False
    return True


def load_cached_observation(entry: Path) -> Mapping[str, Any]:
    if not is_complete(entry):
        raise FileNotFoundError(f"incomplete reconstruction cache: {entry}")
    return load_contract(entry / DOCUMENT_NAME)


def publish_observation(
    entry: Path,
    build: Callable[[Path], Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Build into a sibling temp directory, validate, then publish."""

    entry.parent.mkdir(parents=True, exist_ok=True)
    tmp = entry.parent / f".tmp-{entry.name}-{uuid.uuid4().hex[:8]}"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        observation = dict(build(tmp))
        validate_scene_observation(observation)
        _validate_artifacts(tmp, observation)
        document = tmp / DOCUMENT_NAME
        document.write_text(
            json.dumps(observation, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        load_contract(document)
        (tmp / COMPLETE_NAME).write_text("ok\n", encoding="utf-8")
        if entry.exists():
            shutil.rmtree(entry)
        tmp.rename(entry)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return load_contract(entry / DOCUMENT_NAME)


def _validate_artifacts(work_dir: Path, observation: Mapping[str, Any]) -> None:
    for artifact in observation["artifacts"]:
        uri = artifact.get("uri")
        if not isinstance(uri, str) or not uri:
            raise ContractError("artifact.uri must be a relative path")
        if Path(uri).is_absolute() or ".." in Path(uri).parts:
            raise ContractError(f"artifact.uri is not a safe relative path: {uri}")
        path = work_dir / uri
        if not path.is_file():
            raise ContractError(f"missing artifact file {uri}")
        digest = sha256_file(path)
        if digest != artifact["sha256"]:
            raise ContractError(f"artifact {artifact['id']} hash mismatch")
