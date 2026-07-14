"""Deterministic fixed-width sketches of native residual event tapes.

CRCS is deliberately a transport primitive, not a trainable memory module.
Each event tape is assigned to one signed bucket per independent sketch row
using only its public ordinal.  The model later receives every lane; no
controller semantically retrieves, unbinds, parses, or answers on its behalf.
"""
from __future__ import annotations

import hashlib

import torch


def _validate(events: torch.Tensor, rows: int, buckets: int) -> None:
    if events.ndim != 4:
        raise ValueError("events must have shape [batch, events, tape, d_model]")
    if events.shape[1] <= 0 or rows <= 0 or buckets <= 0:
        raise ValueError("events, rows, and buckets must be positive")


def assignment(event_count: int, rows: int, buckets: int, seed: int = 20260714) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic public bucket and sign codes for event ordinals.

    The result is independent of tensor values and global random state.  Every
    event appears in exactly one bucket in every row, so each row is a standard
    signed CountSketch and the full tensor has ``rows * buckets`` lanes.
    """
    if event_count <= 0 or rows <= 0 or buckets <= 0:
        raise ValueError("event_count, rows, and buckets must be positive")
    locations = torch.empty((rows, event_count), dtype=torch.long)
    signs = torch.empty((rows, event_count), dtype=torch.int8)
    for row in range(rows):
        for event in range(event_count):
            digest = hashlib.blake2b(
                "crcs:{}:{}:{}".format(seed, row, event).encode("ascii"), digest_size=8
            ).digest()
            value = int.from_bytes(digest, "little")
            locations[row, event] = value % buckets
            signs[row, event] = 1 if ((value >> 32) & 1) else -1
    return locations, signs


def sketch_events(events: torch.Tensor, rows: int = 4, buckets: int = 4, seed: int = 20260714) -> torch.Tensor:
    """Bundle event tapes into fixed signed lanes without trainable parameters.

    Args:
        events: Native tapes ``[batch, event_count, tape_len, d_model]``.
        rows/buckets: Fixed CountSketch geometry.  Width is independent of
            event count: ``rows * buckets * tape_len`` residual positions.
    """
    _validate(events, rows, buckets)
    batch, event_count, tape_len, d_model = events.shape
    locations, signs = assignment(event_count, rows, buckets, seed)
    lanes = torch.zeros((batch, rows, buckets, tape_len, d_model), device=events.device, dtype=events.dtype)
    for row in range(rows):
        for event in range(event_count):
            lane = int(locations[row, event])
            lanes[:, row, lane].add_(events[:, event], alpha=int(signs[row, event]))
    return lanes.reshape(batch, rows * buckets * tape_len, d_model)


def flat_sum(events: torch.Tensor) -> torch.Tensor:
    """Matched one-lane additive baseline with the same native tape type."""
    _validate(events, rows=1, buckets=1)
    return events.sum(dim=1)


def collision_summary(event_count: int, rows: int = 4, buckets: int = 4, seed: int = 20260714) -> dict:
    """Expose nonsemantic sketch capacity before a corpus is admitted."""
    locations, _ = assignment(event_count, rows, buckets, seed)
    occupancies = []
    for row in range(rows):
        counts = torch.bincount(locations[row], minlength=buckets)
        occupancies.append([int(value) for value in counts])
    return {
        "event_count": event_count,
        "rows": rows,
        "buckets": buckets,
        "lanes": rows * buckets,
        "occupancies": occupancies,
        "max_bucket_load": max(max(row) for row in occupancies),
        "colliding_assignments": sum(sum(max(0, value - 1) for value in row) for row in occupancies),
    }
