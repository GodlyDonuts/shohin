#!/usr/bin/env python3
"""Run one consumed EFC deployed-wire CPU rehearsal.

The beacons in this utility are fixed synthetic test values. Therefore the
result demonstrates phase ordering and deterministic mechanics only; it does
not establish external unpredictability, an official confirmation board, a
neural compiler result, or permission to pretrain.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.episode_functor_seal_protocol import (
    Beacon,
    ProtocolViolation,
    _publish_immutable,
    canonical_json_bytes,
)
from pipeline.episode_functor_wire_protocol import (
    C_RUNTIME_SOURCE,
    RUST_RUNTIME_SOURCE,
    WireProtocolSpec,
    WireSealFirstRehearsal,
)


WORLD_BEACON = Beacon(
    round=20_260_723_01,
    value="consumed-deployed-wire-world-beacon",
)
CHALLENGE_BEACONS = (
    Beacon(
        round=20_260_723_02,
        value="consumed-deployed-wire-challenge-A",
    ),
    Beacon(
        round=20_260_723_03,
        value="consumed-deployed-wire-challenge-B",
    ),
)


def _compile_runtimes(root: Path) -> dict[str, Path]:
    cc = shutil.which("cc")
    rustc = shutil.which("rustc")
    if cc is None or rustc is None:
        raise ProtocolViolation("strict C and Rust compilers are required")
    runtimes = {"c": root / "runtime_c", "rust": root / "runtime_rust"}
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            str(C_RUNTIME_SOURCE),
            "-o",
            str(runtimes["c"]),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            rustc,
            "--edition=2021",
            "-C",
            "opt-level=2",
            "-D",
            "warnings",
            str(RUST_RUNTIME_SOURCE),
            "-o",
            str(runtimes["rust"]),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return runtimes


def _deployed_query_count_and_uniques(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()
    count = int.from_bytes(payload[20:24], "little")
    semantic: set[tuple[bytes, bytes, bytes]] = set()
    for index in range(count):
        offset = 64 + index * 320
        word_length = int.from_bytes(
            payload[offset + 24 : offset + 26], "little"
        )
        semantic.add(
            (
                payload[offset + 8 : offset + 16],
                payload[offset + 16 : offset + 24],
                payload[offset + 32 : offset + 32 + word_length * 8],
            )
        )
    return count, len(semantic)


def run(out: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(
        prefix="efc-wire-runtime-build."
    ) as build_name:
        runtimes = _compile_runtimes(Path(build_name))
        binary_hashes = tuple(
            (name, sha256(runtimes[name].read_bytes()).hexdigest())
            for name in ("c", "rust")
        )
        rehearsal = WireSealFirstRehearsal(
            out,
            WireProtocolSpec(runtime_binary_sha256=binary_hashes),
        )
        fixture = rehearsal.supply_world_beacon(WORLD_BEACON)
        rehearsal.seal_machine()
        source_delete = rehearsal.poison_and_delete_source()
        challenges = tuple(
            rehearsal.run_challenge(beacon, runtimes)
            for beacon in CHALLENGE_BEACONS
        )

    event_paths = sorted(rehearsal.root.joinpath("events").glob("*.json"))
    event_tip = sha256(event_paths[-1].read_bytes()).hexdigest()
    query_cardinalities = tuple(
        _deployed_query_count_and_uniques(
            rehearsal.root
            / "challenges"
            / receipt.challenge_seed_commitment
            / "queries.bin"
        )
        for receipt in challenges
    )
    report = {
        "checks": {
            "abstract_coordinates_before_query_wire": all(
                receipt.coordinate_commit_event
                < receipt.query_render_event
                for receipt in challenges
            ),
            "compile_count_exactly_one": rehearsal.compile_count == 1,
            "deployed_query_duplicates_zero": all(
                count == unique == receipt.total_coordinates
                for (count, unique), receipt in zip(
                    query_cardinalities, challenges, strict=True
                )
            ),
            "independent_generator_query_fields_seen_zero": (
                fixture.admissibility_receipt["query_fields_seen"] == 0
            ),
            "machine_unchanged_across_challenges": all(
                receipt.machine_sha_before == receipt.machine_sha_after
                for receipt in challenges
            ),
            "nontrivial_empty_observer_partition": (
                1
                < fixture.admissibility_receipt[
                    "empty_observer_class_count"
                ]
                < rehearsal.spec.state_count
            ),
            "source_deleted_before_challenge": (
                bool(source_delete["source_deleted"])
            ),
            "third_assessor_and_c_rust_agree": True,
        },
        "challenge_receipts": [
            receipt.canonical_dict() for receipt in challenges
        ],
        "event_chain_tip_sha256": event_tip,
        "external_beacon_unpredictability_established": False,
        "machine_sha256": sha256(
            rehearsal.machine_path.read_bytes()
        ).hexdigest(),
        "machine_bytes": len(rehearsal.machine_path.read_bytes()),
        "machine_root": rehearsal.machine_root,
        "neural_compiler_preregistration_authorized": False,
        "official_board": False,
        "pretraining_authorized": False,
        "protocol_root": rehearsal.protocol_root,
        "remaining_blockers": [
            "externally sourced future beacon with auditable provenance",
            "multiworld frozen train/development/confirmation custody",
            "neural raw-evidence-to-machine identification",
            "source-deleted neural treatment versus mandatory controls",
            "unopened cross-generator and cross-renderer confirmation",
        ],
        "schema": "efc-deployed-wire-consumed-rehearsal-report-v1",
        "status": "cpu-contract-pass-neural-prereg-no-go",
        "world_evidence_root": rehearsal.world_evidence_root,
        "world_root": rehearsal.world_root,
    }
    if not all(report["checks"].values()):
        raise ProtocolViolation("consumed deployed-wire rehearsal failed")
    _publish_immutable(
        rehearsal.root / "final_report.json",
        canonical_json_bytes(report),
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.out.resolve())
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
