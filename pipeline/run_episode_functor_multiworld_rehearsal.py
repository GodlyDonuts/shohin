#!/usr/bin/env python3
"""Run a consumed EFC multiworld custody rehearsal.

This utility uses fixed synthetic beacons and a placeholder candidate root. It
validates deterministic custody mechanics only. It is not an official
confirmation opening, a neural score, or proof of external unpredictability.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.episode_functor_multiworld_custody import (
    MultiworldCustodyRehearsal,
    MultiworldCustodySpec,
    structural_overlap,
)
from pipeline.episode_functor_seal_protocol import (
    Beacon,
    ProtocolViolation,
    _publish_immutable,
    canonical_json_bytes,
)


OPEN_BEACON = Beacon(
    round=20_260_723_10,
    value="consumed-multiworld-open-beacon",
)
CONFIRMATION_BEACON = Beacon(
    round=20_260_723_11,
    value="consumed-multiworld-confirmation-beacon",
)
CANDIDATE_ROOT = sha256(b"consumed-placeholder-no-neural-candidate").hexdigest()


def run(out: Path) -> dict[str, object]:
    rehearsal = MultiworldCustodyRehearsal(out, MultiworldCustodySpec())
    train, development = rehearsal.freeze_open_splits(OPEN_BEACON)
    open_root_before_candidate = rehearsal.open_splits_root
    rehearsal.seal_candidate(CANDIDATE_ROOT)
    confirmation = rehearsal.open_confirmation(CONFIRMATION_BEACON)
    overlaps = structural_overlap((train, development, confirmation))
    rehearsal.verify_published_state()
    all_records = (
        *train.records,
        *development.records,
        *confirmation.records,
    )
    event_paths = sorted(rehearsal.root.joinpath("events").glob("*.json"))
    events = [json.loads(path.read_bytes()) for path in event_paths]
    event_order = {
        event["event"]: int(event["event_id"]) for event in events
    }
    report = {
        "assessor_secrets_candidate_inaccessible": False,
        "candidate_root": CANDIDATE_ROOT,
        "checks": {
            "all_structural_overlaps_zero": all(
                overlap == 0 for overlap in overlaps.values()
            ),
            "candidate_sealed_before_confirmation": (
                event_order["candidate_sealed"]
                < event_order["confirmation_opened"]
            ),
            "confirmation_beacon_strictly_later": (
                CONFIRMATION_BEACON.round > OPEN_BEACON.round
            ),
            "every_split_has_both_source_renderers": all(
                {record.source_renderer for record in manifest.records}
                == {
                    "canonical-json-events-v2",
                    "strict-line-events-v1",
                }
                for manifest in (train, development, confirmation)
            ),
            "future_behavior_separates_all_states": all(
                record.future_behavior_class_count == rehearsal.spec.state_count
                for record in all_records
            ),
            "nontrivial_empty_observer_partitions": all(
                1 < record.empty_observer_class_count < rehearsal.spec.state_count
                for record in all_records
            ),
            "open_root_unchanged_after_confirmation": (
                rehearsal.open_splits_root == open_root_before_candidate
            ),
        },
        "confirmation_manifest_root": confirmation.manifest_root,
        "counts": {
            "confirmation": len(confirmation.records),
            "development": len(development.records),
            "train": len(train.records),
        },
        "event_chain_tip_sha256": sha256(event_paths[-1].read_bytes()).hexdigest(),
        "external_beacon_unpredictability_established": False,
        "filesystem_process_isolation_established": False,
        "neural_preregistration_authorized": False,
        "official_confirmation": False,
        "open_splits_root": rehearsal.open_splits_root,
        "pretraining_authorized": False,
        "protocol_root": rehearsal.protocol_root,
        "schema": "efc-multiworld-consumed-rehearsal-report-v2",
        "status": "cpu-phase-rehearsal-pass-process-custody-no-go",
        "structural_overlaps": overlaps,
    }
    if not all(report["checks"].values()):
        raise ProtocolViolation("consumed multiworld rehearsal failed")
    _publish_immutable(
        rehearsal.root / "final_report.json",
        canonical_json_bytes(report),
    )
    rehearsal.verify_published_state(
        allowed_top_level_files=frozenset({"final_report.json"})
    )
    if rehearsal.root.joinpath("final_report.json").read_bytes() != (
        canonical_json_bytes(report)
    ):
        raise ProtocolViolation("consumed multiworld report changed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.out.resolve()), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
