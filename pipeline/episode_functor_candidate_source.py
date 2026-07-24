"""Minimal source-only object admitted to an EFC compiler process."""

from __future__ import annotations

from dataclasses import dataclass


class CandidateSourceError(ValueError):
    """A candidate source crossed the declared source-only boundary."""


@dataclass(frozen=True, slots=True)
class CandidateSource:
    """The only source object admissible to a neural forward path."""

    source: bytes

    def __post_init__(self) -> None:
        if not self.source:
            raise CandidateSourceError("candidate source is empty")
        if any(
            marker in self.source.lower()
            for marker in (b"renderer", b"family", b"split")
        ):
            raise CandidateSourceError(
                "candidate source contains forbidden metadata"
            )


__all__ = ["CandidateSource", "CandidateSourceError"]
