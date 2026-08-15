"""Shared evaluation helpers for observed vs simulated trajectories."""

from __future__ import annotations

from typing import Any


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


def contact_timing(
    tracking: dict[str, Any],
    reconstruction: dict[str, Any],
    max_pair_frames: int = 8,
) -> dict[str, Any]:
    observed = tracking["observations"]
    simulated = reconstruction["simulated"]
    observed_contacts = contact_frames(observed)
    simulated_contacts = contact_frames(simulated)
    used: set[int] = set()
    errors: list[int] = []
    pairs: list[list[int]] = []
    unpaired_observed = 0
    for observed_frame in observed_contacts:
        best_index = None
        best_error = None
        for index, simulated_frame in enumerate(simulated_contacts):
            if index in used:
                continue
            error = abs(observed_frame - simulated_frame)
            if best_error is None or error < best_error:
                best_error = error
                best_index = index
        if best_index is None or best_error is None or best_error > max_pair_frames:
            unpaired_observed += 1
            continue
        used.add(best_index)
        errors.append(best_error)
        pairs.append([observed_frame, simulated_contacts[best_index]])
    fps = float(tracking["fps"])
    result: dict[str, Any] = {
        "observed_contact_frames": observed_contacts,
        "simulated_contact_frames": simulated_contacts,
        "paired": len(errors),
        "unpaired_observed": unpaired_observed,
        "unpaired_simulated": len(simulated_contacts) - len(used),
        "pairs": pairs,
    }
    if errors:
        mean_frames = sum(errors) / len(errors)
        result["mean_error_frames"] = mean_frames
        result["mean_error_ms"] = mean_frames * 1000.0 / fps
    return result
