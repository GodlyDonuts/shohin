import copy
import hashlib
import json
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from pipeline import adjudicate_acw_hidden_basis as adjudicator

_REAL_VALIDATE_FROZEN_DEVELOPMENT_BASELINE = (
    adjudicator._validate_frozen_development_baseline
)

FINAL_STATE_EXACTNESS = {
    "acw": 0.96,
    "dense_categorical": 0.80,
    "addressed_continuous": 0.81,
    "gru": 0.82,
    "packet_token_transformer": 0.84,
    "uniform_query_acw": 0.91,
    "answer_motor": 0.83,
    "source_retained": 0.99,
    "direct_state_acw": 1.0,
}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _encoded(payload: dict) -> bytes:
    return adjudicator.canonical_json_bytes(payload) + b"\n"


def _write_json(path: Path, payload: dict) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_encoded(payload))
    return {"path": str(path), "sha256": adjudicator.sha256_file(path)}


def _relative_reference(root: Path, reference: dict[str, str]) -> dict[str, str]:
    return {
        "path": str(Path(reference["path"]).relative_to(root)),
        "sha256": reference["sha256"],
    }


def _accuracy(
    histories: int,
    queries: int,
    *,
    scalar: float = 1.0,
    state: float = 1.0,
) -> dict:
    scalar_total = histories * queries
    scalar_correct = round(scalar_total * scalar)
    state_exact = round(histories * state)
    return {
        "scalar_correct": scalar_correct,
        "scalar_total": scalar_total,
        "scalar_accuracy": scalar_correct / scalar_total,
        "state_exact": state_exact,
        "state_total": histories,
        "state_exactness": state_exact / histories,
    }


def _set_accuracy(
    metric: dict, *, scalar: float | None = None, state: float | None = None
) -> None:
    histories = metric["state_total"]
    queries = metric["scalar_total"] // histories
    replacement = _accuracy(
        histories,
        queries,
        scalar=metric["scalar_accuracy"] if scalar is None else scalar,
        state=metric["state_exactness"] if state is None else state,
    )
    metric.clear()
    metric.update(replacement)


def _identity(split: str, index: int) -> dict:
    if split == "development":
        return {"kind": "development", "seed": adjudicator.DEVELOPMENT_SEEDS[index]}
    return {
        "kind": "confirmation",
        "index": index,
        "commitment": adjudicator.CONFIRMATION_COMMITMENTS[index],
    }


def _dataset_manifest(split: str, index: int) -> dict:
    arrays = {}
    for relative in sorted(adjudicator._required_dataset_arrays()):
        arrays[relative] = {
            "bytes": 1,
            "dtype": "uint8",
            "shape": [1],
            "sha256": _digest(f"array:{relative}"),
        }
    return adjudicator.with_payload_hash(
        {
            "protocol": adjudicator.GENERATOR_PROTOCOL,
            "seed_identity": _identity(split, index),
            "seed_fingerprint": _digest(f"seed:{split}:{index}"),
            "field_size": 17,
            "dimension": 3,
            "source_dim": 96,
            "event_dim": 96,
            "event_count": 48,
            "event_address_counts": {"0": 16, "1": 16, "2": 16},
            "public_queries": 24,
            "new_queries": 8,
            "counts": {
                "train": 4096,
                "adaptation": 1024,
                "evaluation_per_depth": 2048,
            },
            "evaluation_depths": list(adjudicator.EVALUATION_DEPTHS),
            "visited_buckets": {
                "train": {"train": 1},
                "adaptation": {"adaptation": 1},
                "evaluation": {
                    str(depth): {"evaluation": 1}
                    for depth in adjudicator.EVALUATION_DEPTHS
                },
            },
            "depth_counts": {
                "train": {str(depth): 1 for depth in range(9)},
                "adaptation": {"8": 1},
            },
            "arrays": arrays,
        }
    )


def _evaluation_report(
    arm: str,
    split: str,
    index: int,
    dataset_payload_sha256: str,
    checkpoint_sha256: str,
) -> dict:
    checkpoint_arm, model_arm = adjudicator._checkpoint_arms(arm)
    final_state = FINAL_STATE_EXACTNESS[arm]
    public_depths = {
        str(depth): _accuracy(
            adjudicator.PUBLIC_HISTORIES,
            adjudicator.PUBLIC_QUERIES,
            state=final_state if depth == 64 else 1.0,
        )
        for depth in adjudicator.EVALUATION_DEPTHS
    }
    new_reader_depths = {
        str(depth): _accuracy(
            adjudicator.PUBLIC_HISTORIES,
            adjudicator.NEW_READER_QUERIES,
        )
        for depth in adjudicator.EVALUATION_DEPTHS
    }
    report = {
        "protocol": adjudicator.EVALUATION_PROTOCOL,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_arm": checkpoint_arm,
        "model_arm": model_arm,
        "parameters": adjudicator.ARM_PARAMETERS[arm],
        "dataset_manifest_payload_sha256": dataset_payload_sha256,
        "seed_identity": _identity(split, index),
        "optimizer_seed": adjudicator._expected_optimizer_seed(_identity(split, index)),
        "query_schedule_kind": (
            "uniform_schedule.jsonl"
            if arm == "uniform_query_acw"
            else "cgb_schedule.jsonl"
        ),
        "pilot_report_payload_sha256": _digest("frozen-pilot-report-v2"),
        "training_evidence": _native_training_evidence(arm, split, index),
        "scientific_identity": {
            "scientific_commit": "d" * 40,
            "scientific_path_sha256": {
                "R12_ADDRESSED_CATEGORICAL_WORKSPACE_PREREG.md": "b" * 64,
                "pipeline/evaluate_acw_hidden_basis.py": "c" * 64,
            },
        },
        "public_depths": public_depths,
        "new_reader": {
            "updates": 500,
            "state_dim": 32,
            "reader_parameters": 4096,
            "loss_first": 1.0,
            "loss_last": 0.01,
            "depths": new_reader_depths,
        },
        "compiled_sparse_control": _compiled_sparse_control(),
        "claim_boundary": adjudicator.EVALUATOR_CLAIM_BOUNDARY,
    }
    if arm != adjudicator.DIRECT_STATE_ARM:
        report["label_efficiency"] = _native_label_efficiency(
            arm, split, index, public_depths["64"]
        )
    if model_arm == "acw":
        report.update(
            {
                "packet_interventions": {
                    "donor_following": _accuracy(
                        adjudicator.PUBLIC_HISTORIES,
                        adjudicator.PUBLIC_QUERIES,
                    ),
                    "shuffled_against_original": _accuracy(
                        adjudicator.PUBLIC_HISTORIES,
                        adjudicator.PUBLIC_QUERIES,
                        scalar=0.06,
                        state=0.0,
                    ),
                    "held_packet_source_swap_predictions_identical": True,
                    "source_swap_basis": adjudicator.SOURCE_SWAP_BASIS,
                    "donor_different_truth_fraction": 1.0,
                },
                "write_legality": {
                    "unaddressed_registers_checked": 262_144,
                    "illegal_writes": 0,
                },
                "event_words": {
                    "histories": adjudicator.EVENT_WORD_HISTORIES,
                    "equivalent_prediction_query_equivalence": 1.0,
                    "equivalent_a": _accuracy(
                        adjudicator.EVENT_WORD_HISTORIES,
                        adjudicator.PUBLIC_QUERIES,
                    ),
                    "equivalent_b": _accuracy(
                        adjudicator.EVENT_WORD_HISTORIES,
                        adjudicator.PUBLIC_QUERIES,
                    ),
                    "non_equivalent_target_separator_rate": 1.0,
                    "non_equivalent_prediction_separator_rate": 1.0,
                    "non_equivalent_a": _accuracy(
                        adjudicator.EVENT_WORD_HISTORIES,
                        adjudicator.PUBLIC_QUERIES,
                    ),
                    "non_equivalent_b": _accuracy(
                        adjudicator.EVENT_WORD_HISTORIES,
                        adjudicator.PUBLIC_QUERIES,
                    ),
                },
            }
        )
    return adjudicator.with_payload_hash(report)


def _native_resource_ledger(arm: str) -> dict:
    (
        semantic_bits,
        persistent_bytes,
        persistent_dtype,
        training_bytes,
        transient_bytes,
        _,
        parameter_matched,
    ) = adjudicator.ARM_RESOURCES[arm]
    return {
        "trainable_parameters": adjudicator.ARM_PARAMETERS[arm],
        "semantic_state_bits": semantic_bits,
        "persistent_evaluation_bytes": persistent_bytes,
        "persistent_evaluation_dtype": persistent_dtype,
        "persistent_training_state_bytes": training_bytes,
        "declared_transient_token_bytes": transient_bytes,
        "parameter_matched_primary": parameter_matched,
    }


def _resource_profile(
    scope: str,
    *,
    training: bool = False,
    direct_training: bool = False,
) -> dict:
    profile = {
        "scope": scope,
        "batch_size": 256,
        "active_events": 2_048,
        "wall_seconds": 1.0,
        "process_peak_rss_bytes": 1_000_000,
        "profiler_event_count": 2,
        "operator_inventory": [
            {
                "name": "aten::addmm",
                "calls": 1,
                "operator_reported_flops": 1_000_000,
                "positive_allocation_bytes": 16_384,
                "positive_self_allocation_bytes": 2_048,
            },
            {
                "name": "aten::view",
                "calls": 1,
                "operator_reported_flops": 0,
                "positive_allocation_bytes": 0,
                "positive_self_allocation_bytes": 0,
            },
        ],
        "uncounted_operator_names": ["aten::view"],
        "operator_inventory_complete": True,
        "operator_reported_flops": 1_000_000,
        "largest_operator_allocation_bytes": 4_096,
        "largest_self_operator_allocation_bytes": 2_048,
        "total_positive_operator_allocations_bytes": 16_384,
        "flop_counting_contract": adjudicator.FLOP_COUNTING_CONTRACT,
        "transient_memory_contract": adjudicator.TRANSIENT_MEMORY_CONTRACT,
    }
    if training:
        profile["optimizer_included"] = True
    if direct_training:
        profile["state_auxiliary_weight"] = 4.0
    return profile


def _native_training_evidence(arm: str, split: str, index: int) -> dict:
    direct = arm == adjudicator.DIRECT_STATE_ARM
    schedule_family = "uniform" if arm == "uniform_query_acw" else "cgb"
    return {
        "trainer_bundle_manifest_payload_sha256": _digest(
            f"trainer-bundle:{schedule_family}:{split}:{index}"
        ),
        "curriculum_sha256": _digest(f"curriculum:{schedule_family}:{split}:{index}"),
        "query_schedule_sha256": _digest(f"query-schedule:{schedule_family}:v2"),
        "updates": adjudicator.OPTIMIZER_UPDATES,
        "labels": adjudicator.FINAL_SCALAR_LABELS,
        "resource_ledger": _native_resource_ledger(arm),
        "resource_measurements": {
            "training": _resource_profile(
                (
                    adjudicator.DIRECT_TRAINING_PROFILE_SCOPE
                    if direct
                    else adjudicator.TRAINING_PROFILE_SCOPE
                ),
                training=True,
                direct_training=direct,
            ),
            "inference": _resource_profile(adjudicator.INFERENCE_PROFILE_SCOPE),
        },
    }


def _native_label_efficiency(
    arm: str,
    split: str,
    index: int,
    final_metric: dict,
) -> list[dict]:
    records = []
    final_round = len(adjudicator.LABEL_CHECKPOINTS) - 1
    for round_index, labels in enumerate(adjudicator.LABEL_CHECKPOINTS):
        final = round_index == final_round
        records.append(
            {
                "round": round_index,
                "labels": labels,
                "optimizer_updates": (
                    adjudicator.OPTIMIZER_UPDATES if final else 200 * (round_index + 1)
                ),
                "model_tensor_sha256": _digest(
                    f"label-model:{arm}:{split}:{index}:{round_index}"
                ),
                "depth_64": (
                    copy.deepcopy(final_metric)
                    if final
                    else _accuracy(
                        adjudicator.PUBLIC_HISTORIES,
                        adjudicator.PUBLIC_QUERIES,
                        scalar=min(0.95, 0.25 + 0.05 * round_index),
                        state=min(0.85, 0.10 + 0.06 * round_index),
                    )
                ),
            }
        )
    return records


def _compiled_sparse_control() -> dict:
    event_updates = adjudicator.PUBLIC_HISTORIES * sum(adjudicator.EVALUATION_DEPTHS)
    query_reads = (
        adjudicator.PUBLIC_HISTORIES
        * adjudicator.PUBLIC_QUERIES
        * len(adjudicator.EVALUATION_DEPTHS)
    )
    depths = {}
    for depth in adjudicator.EVALUATION_DEPTHS:
        depths[str(depth)] = {
            **_accuracy(
                adjudicator.PUBLIC_HISTORIES,
                adjudicator.PUBLIC_QUERIES,
            ),
            "transition_state_exact": adjudicator.PUBLIC_HISTORIES,
            "transition_state_total": adjudicator.PUBLIC_HISTORIES,
            "transition_state_exactness": 1.0,
        }
    return {
        "depths": depths,
        "external_event_updates": event_updates,
        "event_arithmetic": {
            "multiplications": 2 * event_updates,
            "additions": 2 * event_updates,
            "modulo": event_updates,
        },
        "external_query_reads": query_reads,
        "query_arithmetic": {
            "multiplications": 3 * query_reads,
            "additions": 3 * query_reads,
            "modulo": query_reads,
            "permutation_lookups": query_reads,
        },
        "resource_ledger": {
            "trainable_parameters": 0,
            "persistent_state_bytes": 3,
            "event_table_bytes": 192,
            "query_table_bytes": 1_024,
            "runtime": "NumPy/Python exact F_17 replay",
        },
        "claim_boundary": "Known exact compilation; not neural learnability evidence.",
    }


class SyntheticEvidence:
    def __init__(self, root: Path):
        self.root = root
        self.manifest_path = root / "manifest.json"
        self.datasets: dict[tuple[str, int], dict] = {}
        self.bundles: dict[tuple[str, int, str], dict] = {}
        self.runs: dict[tuple[str, str, int], dict] = {}
        self.manifest: dict = {}
        self._build()

    def _relative_reference(self, reference: dict[str, str]) -> dict[str, str]:
        return _relative_reference(self.root, reference)

    def _rooted_reference(self, directory: Path, payload: dict) -> dict:
        reference = _write_json(directory / "manifest.json", payload)
        return {
            "root": str(directory.relative_to(self.root)),
            "manifest": self._relative_reference(reference),
        }

    def _build(self) -> None:
        for split in ("development", "confirmation"):
            for index in range(3):
                payload = _dataset_manifest(split, index)
                dataset_root = self.root / "datasets" / f"{split}_{index}"
                for relative, record in payload["arrays"].items():
                    path = dataset_root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"x")
                    record["bytes"] = 1
                    record["sha256"] = adjudicator.sha256_file(path)
                payload = adjudicator.with_payload_hash(payload)
                self.datasets[(split, index)] = self._rooted_reference(
                    dataset_root, payload
                )
                for schedule_family in ("cgb", "uniform"):
                    bundle_payload = adjudicator.with_payload_hash(
                        {
                            "trusted_payload_sha256": _digest(
                                f"trainer-bundle:{schedule_family}:{split}:{index}"
                            ),
                            "trusted_curriculum_sha256": _digest(
                                f"curriculum:{schedule_family}:{split}:{index}"
                            ),
                            "trusted_query_schedule_sha256": _digest(
                                f"query-schedule:{schedule_family}:v2"
                            ),
                            "trusted_query_schedule_kind": (
                                "uniform_schedule.jsonl"
                                if schedule_family == "uniform"
                                else "cgb_schedule.jsonl"
                            ),
                            "trusted_pilot_report_payload_sha256": _digest(
                                "frozen-pilot-report-v2"
                            ),
                        }
                    )
                    self.bundles[(split, index, schedule_family)] = (
                        self._rooted_reference(
                            self.root
                            / "bundles"
                            / f"{split}_{index}_{schedule_family}",
                            bundle_payload,
                        )
                    )

        reports = []
        for arm in (*adjudicator.SCORED_ARMS, adjudicator.DIRECT_STATE_ARM):
            splits = (
                ("development",)
                if arm == adjudicator.DIRECT_STATE_ARM
                else (
                    "development",
                    "confirmation",
                )
            )
            for split in splits:
                for index in range(3):
                    dataset_reference = self.datasets[(split, index)]
                    dataset_path = self.root / dataset_reference["manifest"]["path"]
                    dataset = json.loads(dataset_path.read_text())
                    dataset_payload = dataset["payload_sha256"]
                    checkpoint_label = f"checkpoint:{arm}:{split}:{index}"
                    checkpoint_path = (
                        self.root / "checkpoints" / f"{arm}_{split}_{index}.pt"
                    )
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint_path.write_bytes(checkpoint_label.encode("ascii"))
                    checkpoint_reference = {
                        "path": str(checkpoint_path.relative_to(self.root)),
                        "sha256": adjudicator.sha256_file(checkpoint_path),
                    }
                    checkpoint = checkpoint_reference["sha256"]
                    evaluation = _evaluation_report(
                        arm,
                        split,
                        index,
                        dataset_payload,
                        checkpoint,
                    )
                    stem = f"{arm}_{split}_{index}"
                    primary = _write_json(
                        self.root / "evaluations" / f"{stem}.json", evaluation
                    )
                    replay = _write_json(
                        self.root / "replays" / f"{stem}.json", evaluation
                    )
                    run = {
                        "arm": arm,
                        "checkpoint": checkpoint_reference,
                        "dataset": copy.deepcopy(dataset_reference),
                        "trainer_bundle": copy.deepcopy(
                            self.bundles[
                                (
                                    split,
                                    index,
                                    "uniform" if arm == "uniform_query_acw" else "cgb",
                                )
                            ]
                        ),
                        "evaluation_report": self._relative_reference(primary),
                        "replay_report": self._relative_reference(replay),
                    }
                    reports.append(run)
                    self.runs[(arm, split, index)] = run
        self.manifest = {
            "schema": adjudicator.MANIFEST_SCHEMA,
            "protocol": adjudicator.MANIFEST_PROTOCOL,
            "development_baseline": {
                "path": "trusted-development-baseline.json",
                "sha256": _digest("trusted-development-baseline-file"),
                "payload_sha256": _digest("trusted-development-baseline-payload"),
            },
            "reports": reports,
        }
        self.write_manifest()

    def write_manifest(self, *, bind_payload: bool = True) -> None:
        self.manifest.pop("payload_sha256", None)
        if bind_payload:
            self.manifest = adjudicator.with_payload_hash(self.manifest)
        self.manifest_path.write_bytes(_encoded(self.manifest))

    def mutate_report(
        self,
        key: tuple[str, str, int],
        mutation,
        *,
        bind_payload: bool = True,
        targets: tuple[str, ...] = ("evaluation_report", "replay_report"),
    ) -> None:
        run = self.runs[key]
        for target in targets:
            reference = run[target]
            path = self.root / reference["path"]
            report = json.loads(path.read_text())
            mutation(report)
            if bind_payload:
                report = adjudicator.with_payload_hash(report)
            path.write_bytes(_encoded(report))
            reference["sha256"] = adjudicator.sha256_file(path)
        self.write_manifest()

    def point_to_identity(self, key: tuple[str, str, int], identity: dict) -> None:
        source = self.root / self.runs[key]["dataset"]["manifest"]["path"]
        payload = json.loads(source.read_text())
        payload["seed_identity"] = identity
        payload["seed_fingerprint"] = _digest(f"replacement:{identity}")
        payload = adjudicator.with_payload_hash(payload)
        replacement = (
            self.root
            / "datasets"
            / (f"replacement_{len(list((self.root / 'datasets').iterdir()))}")
        )
        self.runs[key]["dataset"] = self._rooted_reference(replacement, payload)
        self.write_manifest()

    def set_depth_64_metric(
        self,
        key: tuple[str, str, int],
        *,
        state: float,
    ) -> None:
        def mutation(report: dict) -> None:
            _set_accuracy(report["public_depths"]["64"], state=state)
            report["label_efficiency"][-1]["depth_64"] = copy.deepcopy(
                report["public_depths"]["64"]
            )

        self.mutate_report(key, mutation)


def _trusted_dataset_tree(_root: Path, manifest: dict, _label: str) -> dict:
    return {
        "arrays_hashed_and_opened": len(manifest["arrays"]),
        "required_array_shapes_verified": len(adjudicator._required_dataset_specs()),
    }


def _trusted_bundle_summary(
    _root: Path,
    manifest: dict,
    _dataset_manifest: dict,
    dataset_summary: dict,
    _label: str,
) -> dict:
    return {
        "payload_sha256": manifest["trusted_payload_sha256"],
        "source_manifest_payload_sha256": dataset_summary["payload_sha256"],
        "seed_identity": dataset_summary["seed_identity"],
        "query_schedule_sha256": manifest["trusted_query_schedule_sha256"],
        "query_schedule_kind": manifest["trusted_query_schedule_kind"],
        "pilot_report_payload_sha256": manifest["trusted_pilot_report_payload_sha256"],
        "pilot_replay_comparison_payload_sha256": _digest("trusted-pilot-comparison"),
        "pilot_replay_comparison_sha256": _digest("trusted-pilot-comparison-file"),
        "pilot_scientific_identity": {
            "scientific_commit": "a" * 40,
            "scientific_path_sha256": {
                "R12_ADDRESSED_CATEGORICAL_WORKSPACE_PREREG.md": "b" * 64,
                "pipeline/evaluate_acw_hidden_basis.py": "c" * 64,
            },
        },
        "activation_commit": "d" * 40,
        "activation_scientific_identity": {
            "scientific_commit": "d" * 40,
            "scientific_path_sha256": {
                "R12_ADDRESSED_CATEGORICAL_WORKSPACE_PREREG.md": "b" * 64,
                "pipeline/evaluate_acw_hidden_basis.py": "c" * 64,
            },
        },
        "curriculum_sha256": manifest["trusted_curriculum_sha256"],
        "arrays_hashed_and_opened": len(adjudicator.BUNDLE_ARRAYS),
        "pilot_artifacts_opened": len(adjudicator.BUNDLE_PILOT_ARTIFACTS),
    }


def _trusted_checkpoint_summary(
    _path: Path,
    file_sha256: str,
    *,
    logical_arm: str,
    dataset_summary: dict,
    bundle_summary: dict,
    label: str,
    **_kwargs,
) -> dict:
    del label
    identity = dataset_summary["seed_identity"]
    split = identity["kind"]
    index = (
        adjudicator.DEVELOPMENT_SEEDS.index(identity["seed"])
        if split == "development"
        else identity["index"]
    )
    checkpoint_arm, model_arm = adjudicator._checkpoint_arms(logical_arm)
    training_evidence = _native_training_evidence(logical_arm, split, index)
    return {
        "sha256": file_sha256,
        "checkpoint_arm": checkpoint_arm,
        "model_arm": model_arm,
        "parameters": adjudicator.ARM_PARAMETERS[logical_arm],
        "scientific_identity": {
            "scientific_commit": "d" * 40,
            "scientific_path_sha256": {
                "R12_ADDRESSED_CATEGORICAL_WORKSPACE_PREREG.md": "b" * 64,
                "pipeline/evaluate_acw_hidden_basis.py": "c" * 64,
            },
        },
        "training_evidence": training_evidence,
        "dataset_manifest_payload_sha256": bundle_summary["payload_sha256"],
        "source_manifest_payload_sha256": dataset_summary["payload_sha256"],
        "curriculum_sha256": bundle_summary["curriculum_sha256"],
        "query_schedule_sha256": bundle_summary["query_schedule_sha256"],
        "query_schedule_kind": bundle_summary["query_schedule_kind"],
        "pilot_report_payload_sha256": bundle_summary["pilot_report_payload_sha256"],
    }


def _trusted_independent_replay(
    _checkpoint: Path,
    _dataset: Path,
    expected_bytes: bytes,
    _label: str,
    **_kwargs,
) -> dict:
    report = json.loads(expected_bytes)
    return {
        "sha256": hashlib.sha256(expected_bytes).hexdigest(),
        "payload_sha256": report["payload_sha256"],
        "byte_identical": True,
        "process_isolation": True,
    }


def _trusted_frozen_baseline(path: Path) -> dict:
    full_manifest = json.loads(Path(path).read_text())
    development_manifest = adjudicator.with_payload_hash(
        {
            "schema": adjudicator.DEVELOPMENT_MANIFEST_SCHEMA,
            "protocol": adjudicator.DEVELOPMENT_MANIFEST_PROTOCOL,
            "reports": [
                report
                for report in full_manifest["reports"]
                if "_development_" in report["evaluation_report"]["path"]
            ],
        }
    )
    runs, verification = adjudicator.verify_evidence(
        development_manifest, Path(path).parent, scope="development"
    )
    selection = adjudicator._development_baseline(runs)
    return {
        **selection,
        "selection": selection,
        "record": {
            "path": "trusted-development-baseline.json",
            "sha256": _digest("trusted-development-baseline-file"),
            "payload_sha256": _digest("trusted-development-baseline-payload"),
        },
        "development_manifest": {
            "path": "trusted-development-manifest.json",
            "sha256": _digest("trusted-development-manifest-file"),
            "payload_sha256": development_manifest["payload_sha256"],
        },
        "development_verification": verification,
        "development_run_bindings": adjudicator._development_run_bindings(runs),
        "activation_scientific_identity": verification["scientific_identity"],
        "confirmation_authorization": adjudicator._confirmation_authorization(),
        "source_checkpoint": selection["selected_checkpoint"]["checkpoint"],
        "copied_checkpoint": {
            **selection["selected_checkpoint"]["checkpoint"],
            "mode": "0444",
        },
        "confirmation_evidence_opened_when_frozen": False,
        "retention_independent_of_promotion": True,
        "can_override_promotion_gates": False,
        "claim_boundary": adjudicator.DEVELOPMENT_BASELINE_CLAIM,
    }


class ACWHiddenBasisAdjudicatorTests(unittest.TestCase):
    def setUp(self) -> None:
        patchers = (
            mock.patch.object(
                adjudicator, "_validate_dataset_tree", _trusted_dataset_tree
            ),
            mock.patch.object(
                adjudicator, "_validate_trainer_bundle", _trusted_bundle_summary
            ),
            mock.patch.object(
                adjudicator,
                "_validate_checkpoint_artifact",
                _trusted_checkpoint_summary,
            ),
            mock.patch.object(
                adjudicator,
                "_independent_evaluator_replay",
                _trusted_independent_replay,
            ),
            mock.patch.object(
                adjudicator,
                "_validate_frozen_development_baseline",
                _trusted_frozen_baseline,
            ),
        )
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _fixture(self, temporary: str) -> SyntheticEvidence:
        return SyntheticEvidence(Path(temporary))

    def _assert_payload_hash(self, decision: dict) -> None:
        recorded = decision["payload_sha256"]
        payload = dict(decision)
        payload.pop("payload_sha256")
        self.assertEqual(
            recorded,
            hashlib.sha256(adjudicator.canonical_json_bytes(payload)).hexdigest(),
        )

    def _assert_no_go(
        self, fixture: SyntheticEvidence, code: str | None = None
    ) -> dict:
        decision = adjudicator.adjudicate_manifest(
            fixture.manifest_path, fixture.manifest_path
        )
        self.assertFalse(decision["go"])
        self.assertEqual(decision["decision"], "NO_GO")
        if code is not None:
            self.assertIn(code, decision["reasons"])
        self._assert_payload_hash(decision)
        return decision

    def test_all_pass_reports_every_seed_median_and_bounded_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(temporary)
            decision = adjudicator.adjudicate_manifest(
                fixture.manifest_path, fixture.manifest_path
            )

        self.assertTrue(decision["go"])
        self.assertEqual(decision["decision"], "GO")
        self.assertEqual(decision["protocol"], adjudicator.DECISION_PROTOCOL)
        self.assertTrue(adjudicator.EVALUATION_PROTOCOL.endswith("-v2"))
        self.assertTrue(adjudicator.GENERATOR_PROTOCOL.endswith("-v3"))
        self.assertEqual(len(decision["seed_results"]), 51)
        self.assertEqual(
            set(decision["confirmation_medians"]), set(adjudicator.SCORED_ARMS)
        )
        for arm in adjudicator.SCORED_ARMS:
            rows = [row for row in decision["seed_results"] if row["arm"] == arm]
            self.assertEqual(len(rows), 6)
            self.assertEqual(
                {row["split"] for row in rows}, {"development", "confirmation"}
            )
        direct = [
            row
            for row in decision["seed_results"]
            if row["arm"] == adjudicator.DIRECT_STATE_ARM
        ]
        self.assertEqual(len(direct), 3)
        self.assertTrue(all(row["frozen_gate"]["passed"] for row in direct))
        self.assertEqual(
            decision["primary_endpoint"]["strongest_valid_equal_label_control"],
            "packet_token_transformer",
        )
        self.assertGreaterEqual(
            decision["primary_endpoint"]["absolute_margin"],
            adjudicator.CONTROL_MARGIN_FLOOR,
        )
        self.assertIn(
            "not evidence for an autonomous controller", decision["bounded_claim"]
        )
        self.assertEqual(
            decision["requirements"]["label_efficiency_checkpoints"],
            list(adjudicator.LABEL_CHECKPOINTS),
        )
        scored = next(
            row
            for row in decision["seed_results"]
            if row["arm"] == "acw" and row["split"] == "development"
        )
        self.assertEqual(scored["label_efficiency"][-1]["optimizer_updates"], 3_400)
        self.assertNotIn(
            "wall_seconds",
            scored["compiled_sparse_control"]["resource_ledger"],
        )
        self.assertEqual(
            set(decision["verification"]["query_schedule_sha256"]),
            {"cgb_schedule.jsonl", "uniform_schedule.jsonl"},
        )
        baseline = decision["development_baseline"]
        self.assertEqual(baseline["status"], "retained_baseline")
        self.assertEqual(baseline["selected_arm"], "acw")
        self.assertEqual(baseline["candidate_count"], len(adjudicator.SCORED_ARMS) * 3)
        self.assertTrue(baseline["retention_independent_of_promotion"])
        self.assertFalse(baseline["can_override_promotion_gates"])
        self._assert_payload_hash(decision)

    def test_best_valid_development_checkpoint_is_retained_on_no_go(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(temporary)
            for index in range(3):
                fixture.set_depth_64_metric(("acw", "development", index), state=0.70)
            decision = adjudicator.adjudicate_manifest(
                fixture.manifest_path, fixture.manifest_path
            )

        self.assertFalse(decision["go"])
        baseline = decision["development_baseline"]
        self.assertEqual(baseline["status"], "retained_baseline")
        self.assertEqual(baseline["selected_arm"], "uniform_query_acw")
        self.assertEqual(baseline["selected_checkpoint"]["index"], 0)
        self.assertEqual(baseline["candidate_count"], len(adjudicator.SCORED_ARMS) * 3)
        self.assertTrue(baseline["retention_independent_of_promotion"])
        self.assertIn("acw_all_development_seed_rule_failed", decision["reasons"])
        self._assert_payload_hash(decision)

    def test_full_adjudication_requires_a_preconfirmation_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(temporary)
            decision = adjudicator.adjudicate_manifest(fixture.manifest_path)

        self.assertFalse(decision["go"])
        self.assertIn("development_baseline_required", decision["reasons"])

    def test_baseline_is_validated_before_full_evidence_is_opened(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(temporary)
            failure = adjudicator.EvidenceError(
                "development_baseline_mutable", "baseline was not frozen"
            )
            with (
                mock.patch.object(
                    adjudicator,
                    "_validate_frozen_development_baseline",
                    side_effect=failure,
                ),
                mock.patch.object(
                    adjudicator,
                    "verify_evidence",
                    wraps=adjudicator.verify_evidence,
                ) as verify,
                mock.patch.object(
                    adjudicator,
                    "_read_regular_file",
                    wraps=adjudicator._read_regular_file,
                ) as read_regular,
            ):
                decision = adjudicator.adjudicate_manifest(
                    fixture.manifest_path, fixture.manifest_path
                )

        self.assertFalse(decision["go"])
        self.assertIn("development_baseline_mutable", decision["reasons"])
        verify.assert_not_called()
        read_regular.assert_not_called()

    def test_full_manifest_must_bind_the_exact_frozen_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(temporary)
            fixture.manifest["development_baseline"]["sha256"] = _digest(
                "different-baseline"
            )
            fixture.write_manifest()
            decision = adjudicator.adjudicate_manifest(
                fixture.manifest_path, fixture.manifest_path
            )

        self.assertFalse(decision["go"])
        self.assertIn("development_baseline_binding_mismatch", decision["reasons"])

    def test_development_only_freeze_preserves_checkpoint_immutably(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(temporary)
            development_reports = [
                fixture.runs[key]
                for key in sorted(fixture.runs)
                if key[1] == "development"
            ]
            development_manifest = adjudicator.with_payload_hash(
                {
                    "schema": adjudicator.DEVELOPMENT_MANIFEST_SCHEMA,
                    "protocol": adjudicator.DEVELOPMENT_MANIFEST_PROTOCOL,
                    "reports": development_reports,
                }
            )
            manifest_path = Path(temporary) / "development_manifest.json"
            manifest_path.write_bytes(_encoded(development_manifest))
            checkpoint_path = Path(temporary) / "retained" / "checkpoint.pt"
            baseline_path = Path(temporary) / "retained" / "baseline.json"

            baseline = adjudicator.freeze_development_baseline(
                manifest_path, checkpoint_path
            )
            file_sha256 = adjudicator.write_immutable_development_baseline(
                baseline_path, baseline
            )

            self.assertEqual(baseline["selection"]["selected_arm"], "acw")
            self.assertFalse(baseline["confirmation_evidence_opened"])
            self.assertEqual(stat.S_IMODE(checkpoint_path.stat().st_mode), 0o444)
            self.assertEqual(stat.S_IMODE(baseline_path.stat().st_mode), 0o444)
            self.assertEqual(file_sha256, adjudicator.sha256_file(baseline_path))
            self.assertEqual(
                baseline["source_checkpoint"]["sha256"],
                baseline["copied_checkpoint"]["sha256"],
            )
            reopened = _REAL_VALIDATE_FROZEN_DEVELOPMENT_BASELINE(baseline_path)
            self.assertEqual(reopened["selected_arm"], "acw")
            self.assertFalse(reopened["confirmation_evidence_opened_when_frozen"])
            with self.assertRaises(FileExistsError):
                adjudicator.write_immutable_development_baseline(
                    baseline_path, baseline
                )
            baseline_path.chmod(0o644)
            with self.assertRaises(adjudicator.EvidenceError) as raised:
                _REAL_VALIDATE_FROZEN_DEVELOPMENT_BASELINE(baseline_path)
            self.assertEqual(raised.exception.code, "development_baseline_mutable")

    def test_self_attested_empty_development_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(temporary)
            development_reports = [
                fixture.runs[key]
                for key in sorted(fixture.runs)
                if key[1] == "development"
            ]
            development_manifest = adjudicator.with_payload_hash(
                {
                    "schema": adjudicator.DEVELOPMENT_MANIFEST_SCHEMA,
                    "protocol": adjudicator.DEVELOPMENT_MANIFEST_PROTOCOL,
                    "reports": development_reports,
                }
            )
            manifest_path = Path(temporary) / "development_manifest.json"
            manifest_path.write_bytes(_encoded(development_manifest))
            checkpoint_path = Path(temporary) / "retained" / "checkpoint.pt"
            baseline = adjudicator.freeze_development_baseline(
                manifest_path, checkpoint_path
            )

            empty_manifest = adjudicator.with_payload_hash(
                {
                    "schema": adjudicator.DEVELOPMENT_MANIFEST_SCHEMA,
                    "protocol": adjudicator.DEVELOPMENT_MANIFEST_PROTOCOL,
                    "reports": [],
                }
            )
            empty_path = Path(temporary) / "empty_development_manifest.json"
            empty_path.write_bytes(_encoded(empty_manifest))
            forged = copy.deepcopy(baseline)
            forged["development_manifest"] = {
                "path": str(empty_path.resolve()),
                "sha256": adjudicator.sha256_file(empty_path),
                "payload_sha256": empty_manifest["payload_sha256"],
            }
            forged = adjudicator.with_payload_hash(forged)
            forged_path = Path(temporary) / "retained" / "forged_baseline.json"
            forged_path.write_bytes(_encoded(forged))
            forged_path.chmod(0o444)

            with self.assertRaises(adjudicator.EvidenceError) as raised:
                _REAL_VALIDATE_FROZEN_DEVELOPMENT_BASELINE(forged_path)
            self.assertEqual(raised.exception.code, "report_count_mismatch")

    def test_cli_requires_an_explicit_preconfirmation_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(temporary)
            output = Path(temporary) / "decision.json"
            with self.assertRaisesRegex(SystemExit, "requires --development-baseline"):
                adjudicator.main(
                    [
                        "--manifest",
                        str(fixture.manifest_path),
                        "--out",
                        str(output),
                    ]
                )
            self.assertFalse(output.exists())

    def test_hash_protocol_schema_and_replay_mutations_fail_closed(self) -> None:
        cases = []

        def stale_manifest(fixture: SyntheticEvidence) -> None:
            fixture.manifest["protocol"] = "changed-after-hash"
            fixture.manifest_path.write_bytes(_encoded(fixture.manifest))

        cases.append(("manifest payload", stale_manifest, "payload_hash_mismatch"))

        def stale_report(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("acw", "development", 0),
                lambda report: report.__setitem__("protocol", "changed-after-hash"),
                bind_payload=False,
            )

        cases.append(("evaluation payload", stale_report, "payload_hash_mismatch"))

        def wrong_protocol(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("acw", "development", 0),
                lambda report: report.__setitem__("protocol", "wrong"),
            )

        cases.append(
            ("evaluation protocol", wrong_protocol, "evaluation_protocol_mismatch")
        )

        def extra_schema_key(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("acw", "development", 0),
                lambda report: report.__setitem__("unfrozen", True),
            )

        cases.append(("evaluation schema", extra_schema_key, "schema_mismatch"))

        def changed_replay(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("acw", "development", 0),
                lambda report: report["new_reader"].__setitem__("loss_last", 0.02),
                targets=("replay_report",),
            )

        cases.append(("replay bytes", changed_replay, "replay_hash_mismatch"))

        def stale_dataset(fixture: SyntheticEvidence) -> None:
            reference = fixture.runs[("acw", "development", 0)]["dataset"]["manifest"]
            path = fixture.root / reference["path"]
            payload = json.loads(path.read_text())
            payload["field_size"] = 19
            path.write_bytes(_encoded(payload))
            reference["sha256"] = adjudicator.sha256_file(path)
            fixture.write_manifest()

        cases.append(("dataset payload", stale_dataset, "payload_hash_mismatch"))

        for name, mutation, code in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = self._fixture(temporary)
                mutation(fixture)
                self._assert_no_go(fixture, code)

    def test_seed_matrix_rejects_missing_duplicate_pilot_and_unregistered_identities(
        self,
    ) -> None:
        cases = []

        def missing(fixture: SyntheticEvidence) -> None:
            fixture.manifest["reports"].pop()
            fixture.write_manifest()

        cases.append(("missing", missing, "report_count_mismatch"))

        def duplicate(fixture: SyntheticEvidence) -> None:
            reports = fixture.manifest["reports"]
            reports[1] = copy.deepcopy(reports[0])
            fixture.write_manifest()

        cases.append(("duplicate", duplicate, "duplicate_seed_identity"))

        def pilot(fixture: SyntheticEvidence) -> None:
            fixture.point_to_identity(
                ("acw", "development", 0),
                {"kind": "pilot", "seed": 2026071600},
            )

        cases.append(("pilot", pilot, "pilot_seed_forbidden"))

        def unregistered(fixture: SyntheticEvidence) -> None:
            fixture.point_to_identity(
                ("acw", "development", 0),
                {"kind": "development", "seed": 2026071699},
            )

        cases.append(("unregistered", unregistered, "unregistered_seed_identity"))

        def wrong_commitment(fixture: SyntheticEvidence) -> None:
            fixture.point_to_identity(
                ("acw", "confirmation", 0),
                {"kind": "confirmation", "index": 0, "commitment": "f" * 64},
            )

        cases.append(
            (
                "confirmation commitment",
                wrong_commitment,
                "confirmation_commitment_mismatch",
            )
        )

        def direct_confirmation(fixture: SyntheticEvidence) -> None:
            fixture.runs[(adjudicator.DIRECT_STATE_ARM, "development", 0)][
                "dataset"
            ] = copy.deepcopy(fixture.datasets[("confirmation", 0)])
            fixture.write_manifest()

        cases.append(
            (
                "direct confirmation",
                direct_confirmation,
                "direct_state_confirmation_forbidden",
            )
        )

        for name, mutation, code in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = self._fixture(temporary)
                mutation(fixture)
                self._assert_no_go(fixture, code)

    def test_direct_state_and_three_plus_two_seed_rules(self) -> None:
        with self.subTest("direct-state"), tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(temporary)
            fixture.mutate_report(
                (adjudicator.DIRECT_STATE_ARM, "development", 0),
                lambda report: _set_accuracy(report["public_depths"]["8"], state=0.94),
            )
            self._assert_no_go(fixture, "direct_state_diagnostic_gate_failed")

        with (
            self.subTest("all development"),
            tempfile.TemporaryDirectory() as temporary,
        ):
            fixture = self._fixture(temporary)
            fixture.mutate_report(
                ("acw", "development", 0),
                lambda report: _set_accuracy(
                    report["packet_interventions"]["donor_following"], scalar=0.98
                ),
            )
            self._assert_no_go(fixture, "acw_all_development_seed_rule_failed")

        with (
            self.subTest("two confirmation pass"),
            tempfile.TemporaryDirectory() as temporary,
        ):
            fixture = self._fixture(temporary)
            fixture.mutate_report(
                ("acw", "confirmation", 0),
                lambda report: _set_accuracy(
                    report["packet_interventions"]["donor_following"], scalar=0.98
                ),
            )
            decision = adjudicator.adjudicate_manifest(
                fixture.manifest_path, fixture.manifest_path
            )
            self.assertTrue(decision["go"])
            self.assertEqual(decision["acw_seed_rule"]["confirmation_passes"], 2)

        with (
            self.subTest("only one confirmation pass"),
            tempfile.TemporaryDirectory() as temporary,
        ):
            fixture = self._fixture(temporary)
            for index in (0, 1):
                fixture.mutate_report(
                    ("acw", "confirmation", index),
                    lambda report: _set_accuracy(
                        report["packet_interventions"]["donor_following"], scalar=0.98
                    ),
                )
            self._assert_no_go(fixture, "acw_two_of_three_confirmation_rule_failed")

    def test_every_acw_causal_gate_is_decision_relevant(self) -> None:
        mutations = {
            "depth scalar": (
                lambda report: _set_accuracy(
                    report["public_depths"]["32"], scalar=0.98
                ),
                "depth_32_scalar_below_0.99",
            ),
            "depth state": (
                lambda report: _set_accuracy(report["public_depths"]["65"], state=0.84),
                "depth_65_state_below_0.85",
            ),
            "donor": (
                lambda report: _set_accuracy(
                    report["packet_interventions"]["donor_following"], scalar=0.98
                ),
                "donor_following_scalar_below_0.99",
            ),
            "shuffle": (
                lambda report: _set_accuracy(
                    report["packet_interventions"]["shuffled_against_original"],
                    scalar=0.09,
                ),
                "shuffled_scalar_above_chance_plus_0.02",
            ),
            "source swap": (
                lambda report: report["packet_interventions"].__setitem__(
                    "held_packet_source_swap_predictions_identical", False
                ),
                "held_packet_source_swap_changed_predictions",
            ),
            "donor support": (
                lambda report: report["packet_interventions"].__setitem__(
                    "donor_different_truth_fraction", 0.0
                ),
                "donor_map_has_no_truth_change",
            ),
            "new reader": (
                lambda report: _set_accuracy(
                    report["new_reader"]["depths"]["64"], state=0.89
                ),
                "new_reader_depth_64_state_below_0.90",
            ),
            "illegal write": (
                lambda report: report["write_legality"].__setitem__(
                    "illegal_writes", 1
                ),
                "illegal_multi_register_write",
            ),
            "equivalent words": (
                lambda report: report["event_words"].__setitem__(
                    "equivalent_prediction_query_equivalence", 0.99
                ),
                "equivalent_event_words_not_query_equivalent",
            ),
            "non-equivalent words": (
                lambda report: report["event_words"].__setitem__(
                    "non_equivalent_prediction_separator_rate", 0.99
                ),
                "non_equivalent_event_words_lack_prediction_separator",
            ),
        }
        for name, (mutation, seed_failure) in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = self._fixture(temporary)
                fixture.mutate_report(("acw", "development", 0), mutation)
                decision = self._assert_no_go(
                    fixture, "acw_all_development_seed_rule_failed"
                )
                row = next(
                    result
                    for result in decision["seed_results"]
                    if result["arm"] == "acw"
                    and result["split"] == "development"
                    and result["index"] == 0
                )
                self.assertIn(seed_failure, row["frozen_gate"]["failures"])

    def test_label_and_resource_ledger_mutations_fail_closed(self) -> None:
        cases = []

        def wrong_labels(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("acw", "development", 0),
                lambda report: report["training_evidence"].__setitem__(
                    "labels", adjudicator.FINAL_SCALAR_LABELS - 1
                ),
            )

        cases.append(("label count", wrong_labels, "label_count_mismatch"))

        def missing_train_field(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("acw", "development", 0),
                lambda report: report["training_evidence"]["resource_measurements"][
                    "training"
                ].pop("operator_reported_flops"),
            )

        cases.append(("train schema", missing_train_field, "schema_mismatch"))

        def incomplete_inference(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("acw", "development", 0),
                lambda report: report["training_evidence"]["resource_measurements"][
                    "inference"
                ].__setitem__("wall_seconds", 0.0),
            )

        cases.append(
            ("inference complete", incomplete_inference, "incomplete_resource_ledger")
        )

        def resource_drift(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("acw", "development", 0),
                lambda report: report["training_evidence"][
                    "resource_ledger"
                ].__setitem__("persistent_evaluation_bytes", 4),
            )

        cases.append(("resource drift", resource_drift, "resource_ledger_mismatch"))

        def hidden_oracle_field(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("acw", "development", 0),
                lambda report: report["training_evidence"].__setitem__(
                    "oracle_access", True
                ),
            )

        cases.append(("hidden oracle field", hidden_oracle_field, "schema_mismatch"))

        def optimizer_mismatch(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("acw", "confirmation", 0),
                lambda report: report.__setitem__(
                    "optimizer_seed", report["optimizer_seed"] + 1
                ),
            )

        cases.append(("optimizer seed", optimizer_mismatch, "optimizer_seed_mismatch"))

        def malformed_curriculum_hash(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("dense_categorical", "development", 0),
                lambda report: report["training_evidence"].__setitem__(
                    "curriculum_sha256", "not-a-hash"
                ),
            )

        cases.append(("curriculum hash", malformed_curriculum_hash, "invalid_sha256"))

        def compiled_wall_time(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("acw", "development", 0),
                lambda report: report["compiled_sparse_control"][
                    "resource_ledger"
                ].__setitem__("wall_seconds", 0.1),
            )

        cases.append(("compiled wall time", compiled_wall_time, "schema_mismatch"))

        def compiled_transition_failure(fixture: SyntheticEvidence) -> None:
            def mutation(report: dict) -> None:
                depth = report["compiled_sparse_control"]["depths"]["64"]
                depth["transition_state_exact"] -= 1
                depth["transition_state_exactness"] = (
                    depth["transition_state_exact"] / depth["transition_state_total"]
                )

            fixture.mutate_report(("acw", "development", 0), mutation)

        cases.append(
            (
                "compiled transition",
                compiled_transition_failure,
                "compiled_sparse_control_failed",
            )
        )

        for name, mutation, code in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = self._fixture(temporary)
                mutation(fixture)
                self._assert_no_go(fixture, code)

    def test_training_artifact_hash_relationships_fail_closed(self) -> None:
        cases = []

        def missing_bundle_hash(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("acw", "development", 0),
                lambda report: report["training_evidence"].pop(
                    "trainer_bundle_manifest_payload_sha256"
                ),
            )

        cases.append(("missing bundle hash", missing_bundle_hash, "schema_mismatch"))

        def cgb_schedule_fork(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("acw", "development", 0),
                lambda report: report["training_evidence"].__setitem__(
                    "query_schedule_sha256", _digest("forked-cgb-schedule")
                ),
            )

        cases.append(
            (
                "CGB schedule fork",
                cgb_schedule_fork,
                "evaluation_checkpoint_training_mismatch",
            )
        )

        def schedule_hash_reuse(fixture: SyntheticEvidence) -> None:
            cgb_hash = _digest("query-schedule:cgb:v2")
            for split in ("development", "confirmation"):
                for index in range(3):
                    fixture.mutate_report(
                        ("uniform_query_acw", split, index),
                        lambda report, value=cgb_hash: report[
                            "training_evidence"
                        ].__setitem__("query_schedule_sha256", value),
                    )

        cases.append(
            (
                "schedule hash reuse",
                schedule_hash_reuse,
                "evaluation_checkpoint_training_mismatch",
            )
        )

        def trainer_bundle_fork(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("dense_categorical", "development", 0),
                lambda report: report["training_evidence"].__setitem__(
                    "trainer_bundle_manifest_payload_sha256",
                    _digest("forked-bundle"),
                ),
            )

        cases.append(
            (
                "bundle fork",
                trainer_bundle_fork,
                "evaluation_checkpoint_training_mismatch",
            )
        )

        def trainer_bundle_domain_reuse(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("acw", "development", 1),
                lambda report: report["training_evidence"].__setitem__(
                    "trainer_bundle_manifest_payload_sha256",
                    _digest("trainer-bundle:cgb:development:0"),
                ),
            )

        cases.append(
            (
                "bundle domain reuse",
                trainer_bundle_domain_reuse,
                "evaluation_checkpoint_training_mismatch",
            )
        )

        def curriculum_fork(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("dense_categorical", "development", 0),
                lambda report: report["training_evidence"].__setitem__(
                    "curriculum_sha256", _digest("forked-curriculum")
                ),
            )

        cases.append(
            (
                "curriculum fork",
                curriculum_fork,
                "evaluation_checkpoint_training_mismatch",
            )
        )

        def curriculum_domain_reuse(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("acw", "confirmation", 0),
                lambda report: report["training_evidence"].__setitem__(
                    "curriculum_sha256",
                    _digest("curriculum:cgb:development:0"),
                ),
            )

        cases.append(
            (
                "curriculum domain reuse",
                curriculum_domain_reuse,
                "evaluation_checkpoint_training_mismatch",
            )
        )

        for name, mutation, code in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = self._fixture(temporary)
                mutation(fixture)
                self._assert_no_go(fixture, code)

    def test_label_efficiency_requires_every_frozen_checkpoint_and_final_binding(
        self,
    ) -> None:
        cases = []

        def missing_record(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("acw", "development", 0),
                lambda report: report["label_efficiency"].pop(3),
            )

        cases.append(
            ("missing", missing_record, "label_efficiency_checkpoint_mismatch")
        )

        def wrong_label(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("acw", "development", 0),
                lambda report: report["label_efficiency"][3].__setitem__(
                    "labels", report["label_efficiency"][3]["labels"] + 1
                ),
            )

        cases.append(
            (
                "wrong cumulative labels",
                wrong_label,
                "label_efficiency_checkpoint_mismatch",
            )
        )

        def duplicate_checkpoint(fixture: SyntheticEvidence) -> None:
            def mutation(report: dict) -> None:
                records = report["label_efficiency"]
                records[3]["model_tensor_sha256"] = records[2]["model_tensor_sha256"]

            fixture.mutate_report(("acw", "development", 0), mutation)

        cases.append(("duplicate", duplicate_checkpoint, "duplicate_label_checkpoint"))

        def wrong_final_updates(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("acw", "development", 0),
                lambda report: report["label_efficiency"][-1].__setitem__(
                    "optimizer_updates", 2_600
                ),
            )

        cases.append(
            (
                "final 3400-update binding",
                wrong_final_updates,
                "label_efficiency_checkpoint_mismatch",
            )
        )

        def wrong_final_metric(fixture: SyntheticEvidence) -> None:
            fixture.mutate_report(
                ("acw", "development", 0),
                lambda report: _set_accuracy(
                    report["label_efficiency"][-1]["depth_64"], state=0.1
                ),
            )

        cases.append(
            (
                "final metric",
                wrong_final_metric,
                "label_efficiency_final_metric_mismatch",
            )
        )

        for name, mutation, code in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = self._fixture(temporary)
                mutation(fixture)
                self._assert_no_go(fixture, code)

    def test_primary_confirmation_floor_and_control_margin_are_gates(self) -> None:
        with self.subTest("control margin"), tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(temporary)
            for index in range(3):
                fixture.set_depth_64_metric(
                    ("packet_token_transformer", "confirmation", index),
                    state=0.90,
                )
            decision = self._assert_no_go(
                fixture,
                "primary_equal_label_control_margin_below_0.10",
            )
            self.assertEqual(
                decision["primary_endpoint"]["strongest_valid_equal_label_control"],
                "packet_token_transformer",
            )

        with (
            self.subTest("ACW median floor"),
            tempfile.TemporaryDirectory() as temporary,
        ):
            fixture = self._fixture(temporary)
            for index in (0, 1):
                fixture.set_depth_64_metric(("acw", "confirmation", index), state=0.89)
            decision = self._assert_no_go(
                fixture,
                "primary_confirmation_depth_64_state_below_0.90",
            )
            self.assertIn(
                "acw_two_of_three_confirmation_rule_failed", decision["reasons"]
            )

    def test_immutable_hash_bound_decision_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(temporary)
            decision = adjudicator.adjudicate_manifest(
                fixture.manifest_path, fixture.manifest_path
            )
            destination = Path(temporary) / "decision.json"
            file_sha256 = adjudicator.write_immutable_json(destination, decision)
            self.assertEqual(file_sha256, adjudicator.sha256_file(destination))
            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o444)
            self.assertEqual(json.loads(destination.read_text()), decision)
            with self.assertRaises(FileExistsError):
                adjudicator.write_immutable_json(destination, decision)

            stale = dict(decision)
            stale["go"] = False
            other = Path(temporary) / "stale.json"
            with self.assertRaises(adjudicator.EvidenceError):
                adjudicator.write_immutable_json(other, stale)
            self.assertFalse(other.exists())


class ACWAdjudicatorArtifactSecurityTests(unittest.TestCase):
    def test_activation_constants_match_independent_trainer(self) -> None:
        from pipeline import acw_hidden_basis_training as trainer

        for name in (
            "PILOT_ARTIFACT_REGISTRY_PROTOCOL",
            "PILOT_INDEPENDENT_VERIFICATION_PROTOCOL",
            "PILOT_SCIENTIFIC_COMMIT",
            "PILOT_ANCHOR_COMMIT",
            "PILOT_EXECUTION_COMMIT",
            "PILOT_REGISTRY_RAW_SHA256",
            "PILOT_REGISTRY_PATH",
            "PILOT_ANCHORED_FILES",
            "PILOT_CANONICAL_REMOTE_URL",
            "PILOT_OFFLINE_BUNDLE_TEMPLATE",
            "PILOT_ACTIVATION_ALLOWLIST",
            "PILOT_CUSTODY_ALLOWLIST",
            "PILOT_CANONICAL_PATHS",
            "PILOT_REGISTRY_CLAIM",
            "PILOT_INDEPENDENT_VERIFICATION_CLAIM",
        ):
            self.assertEqual(getattr(adjudicator, name), getattr(trainer, name))
        self.assertEqual(
            adjudicator.PILOT_SCIENTIFIC_PATHS,
            trainer.ACW_SCIENTIFIC_PATHS,
        )

    def test_one_byte_synthetic_arrays_cannot_yield_go(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticEvidence(Path(temporary))
            with mock.patch.object(
                adjudicator,
                "_validate_frozen_development_baseline",
                return_value={"record": fixture.manifest["development_baseline"]},
            ):
                decision = adjudicator.adjudicate_manifest(
                    fixture.manifest_path, fixture.manifest_path
                )

        self.assertFalse(decision["go"])
        self.assertEqual(decision["decision"], "NO_GO")
        self.assertIn("evidence_contract_failed", decision["reasons"])
        self.assertIn("invalid_npy_artifact", decision["reasons"])

    def test_rehashed_forged_score_fails_independent_evaluator_replay(self) -> None:
        forged = adjudicator.with_payload_hash(
            {"protocol": adjudicator.EVALUATION_PROTOCOL, "fabricated_score": 1.0}
        )
        observed = adjudicator.with_payload_hash(
            {"protocol": adjudicator.EVALUATION_PROTOCOL, "fabricated_score": 0.0}
        )
        forged_bytes = _encoded(forged)
        observed_bytes = _encoded(observed)

        def fake_evaluator(argv, **_kwargs):
            output = Path(argv[argv.index("--out") + 1])
            output.write_bytes(observed_bytes)
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(
                adjudicator.subprocess, "run", side_effect=fake_evaluator
            ),
        ):
            with self.assertRaises(adjudicator.EvidenceError) as raised:
                adjudicator._independent_evaluator_replay(
                    Path(temporary) / "checkpoint.pt",
                    Path(temporary) / "dataset",
                    forged_bytes,
                    "forged",
                    checkpoint_bytes=b"checkpoint",
                    dataset_manifest={"arrays": {}},
                )
        self.assertEqual(raised.exception.code, "independent_evaluator_mismatch")

    def test_real_checkpoint_tensors_and_bundle_bindings_are_opened(self) -> None:
        from pipeline.acw_hidden_basis_training import model_for_arm

        identity = {
            "kind": "development",
            "seed": adjudicator.DEVELOPMENT_SEEDS[0],
        }
        dataset_summary = {
            "payload_sha256": _digest("real-dataset"),
            "seed_identity": identity,
        }
        evidence = _native_training_evidence("acw", "development", 0)
        bundle_summary = {
            "payload_sha256": evidence["trainer_bundle_manifest_payload_sha256"],
            "source_manifest_payload_sha256": dataset_summary["payload_sha256"],
            "seed_identity": identity,
            "query_schedule_sha256": evidence["query_schedule_sha256"],
            "query_schedule_kind": "cgb_schedule.jsonl",
            "pilot_report_payload_sha256": _digest("frozen-pilot-report-v2"),
            "curriculum_sha256": evidence["curriculum_sha256"],
            "arrays_hashed_and_opened": len(adjudicator.BUNDLE_ARRAYS),
        }
        model = model_for_arm("acw")
        state = {
            name: tensor.detach().clone() for name, tensor in model.state_dict().items()
        }
        checkpoint = {
            "protocol": adjudicator.TRAINING_PROTOCOL,
            "arm": "acw",
            "seed": identity["seed"],
            "dataset_manifest_payload_sha256": bundle_summary["payload_sha256"],
            "source_manifest_payload_sha256": dataset_summary["payload_sha256"],
            "curriculum_sha256": bundle_summary["curriculum_sha256"],
            "query_schedule_sha256": bundle_summary["query_schedule_sha256"],
            "query_schedule_kind": bundle_summary["query_schedule_kind"],
            "pilot_report_payload_sha256": bundle_summary[
                "pilot_report_payload_sha256"
            ],
            "parameters": adjudicator.ARM_PARAMETERS["acw"],
            "training_report": {
                "updates": evidence["updates"],
                "labels": evidence["labels"],
                "resource_ledger": evidence["resource_ledger"],
                "resource_measurements": evidence["resource_measurements"],
            },
            "label_efficiency_models": [copy.deepcopy(state) for _ in range(13)],
            "scientific_identity": {
                "scientific_commit": "a" * 40,
                "scientific_path_sha256": {
                    "pipeline/evaluate_acw_hidden_basis.py": "b" * 64
                },
            },
            "model": state,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            torch.save(checkpoint, path)
            digest = adjudicator.sha256_file(path)
            summary = adjudicator._validate_checkpoint_artifact(
                path,
                digest,
                logical_arm="acw",
                dataset_summary=dataset_summary,
                bundle_summary=bundle_summary,
                label="checkpoint",
            )
            self.assertEqual(summary["sha256"], digest)
            self.assertEqual(summary["training_evidence"], evidence)

            checkpoint["dataset_manifest_payload_sha256"] = _digest("forked-bundle")
            fork = Path(temporary) / "fork.pt"
            torch.save(checkpoint, fork)
            with self.assertRaises(adjudicator.EvidenceError) as raised:
                adjudicator._validate_checkpoint_artifact(
                    fork,
                    adjudicator.sha256_file(fork),
                    logical_arm="acw",
                    dataset_summary=dataset_summary,
                    bundle_summary=bundle_summary,
                    label="fork",
                )
        self.assertEqual(raised.exception.code, "checkpoint_bundle_binding_mismatch")

    def test_real_bundle_arrays_and_curriculum_are_hashed_and_opened(self) -> None:
        def write_array(root: Path, relative: str, value: np.ndarray) -> dict:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                np.save(handle, value, allow_pickle=False)
            return {
                "bytes": path.stat().st_size,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": adjudicator.sha256_file(path),
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            values = {
                "public/event_features.npy": np.zeros((48, 96), dtype=np.float32),
                "public/event_addresses.npy": np.repeat(
                    np.arange(3, dtype=np.int8), 16
                ),
                "public/train/source_features.npy": np.zeros(
                    (4096, 96), dtype=np.float32
                ),
                "public/train/event_ids.npy": np.full((4096, 8), -1, dtype=np.int16),
                "public/train/lengths.npy": np.zeros(4096, dtype=np.int16),
                "public/train/initial_queries.npy": np.tile(
                    np.asarray([[0, 1]], dtype=np.int8), (4096, 1)
                ),
                "public/train/initial_answers.npy": np.zeros((4096, 2), dtype=np.int8),
            }
            arrays = {
                relative: write_array(root, relative, value)
                for relative, value in values.items()
            }
            rows = []
            for history_id in range(4096):
                rows.extend(
                    (
                        {
                            "history_id": history_id,
                            "query_id": 0,
                            "answer": 0,
                            "round": 0,
                        },
                        {
                            "history_id": history_id,
                            "query_id": 1,
                            "answer": 0,
                            "round": 0,
                        },
                    )
                )
                rows.extend(
                    {
                        "history_id": history_id,
                        "query_id": round_index + 1,
                        "answer": 0,
                        "round": round_index,
                    }
                    for round_index in range(1, 13)
                )
            curriculum = root / "curriculum.jsonl"
            curriculum.write_bytes(b"".join(_encoded(row) for row in rows))
            curriculum_record = {
                "bytes": curriculum.stat().st_size,
                "rows": len(rows),
                "sha256": adjudicator.sha256_file(curriculum),
            }
            schedule_rows = [
                {
                    "history_id": row["history_id"],
                    "query_id": row["query_id"],
                    "round": row["round"],
                }
                for row in rows
            ]
            schedule_raw = b"".join(_encoded(row) for row in schedule_rows)
            pilot_root = root / "pilot"
            pilot_root.mkdir()
            schedule_records = {}
            for name in ("cgb_schedule.jsonl", "uniform_schedule.jsonl"):
                path = pilot_root / name
                path.write_bytes(schedule_raw)
                schedule_records[name] = {
                    "bytes": len(schedule_raw),
                    "rows": len(schedule_rows),
                    "sha256": adjudicator.sha256_file(path),
                }
            pilot_identity = {
                "scientific_commit": "a" * 40,
                "scientific_path_sha256": {
                    "R12_ADDRESSED_CATEGORICAL_WORKSPACE_PREREG.md": "b" * 64,
                    "pipeline/evaluate_acw_hidden_basis.py": "c" * 64,
                },
            }
            pilot_report = adjudicator.with_payload_hash(
                {
                    "protocol": adjudicator.PILOT_PROTOCOL,
                    "dataset_manifest_payload_sha256": _digest("pilot-dataset"),
                    "scientific_identity": pilot_identity,
                    "schedules": schedule_records,
                }
            )
            pilot_report_path = pilot_root / "report.json"
            pilot_report_path.write_bytes(_encoded(pilot_report))
            common_files = {
                "report.json": {
                    "bytes": pilot_report_path.stat().st_size,
                    "sha256": adjudicator.sha256_file(pilot_report_path),
                },
                **{
                    name: {
                        "bytes": (pilot_root / name).stat().st_size,
                        "sha256": adjudicator.sha256_file(pilot_root / name),
                    }
                    for name in schedule_records
                },
            }
            pilot_comparison = adjudicator.with_payload_hash(
                {
                    "protocol": adjudicator.PILOT_COMPARISON_PROTOCOL,
                    "reports_byte_identical": True,
                    "schedules_byte_identical": True,
                    "independently_recomputed": True,
                    "dataset_manifest_payload_sha256": pilot_report[
                        "dataset_manifest_payload_sha256"
                    ],
                    "scientific_identity": pilot_identity,
                    "common_files": common_files,
                    "independent_recomputation_sha256": {
                        name: record["sha256"] for name, record in common_files.items()
                    },
                }
            )
            pilot_comparison_path = pilot_root / "replay_comparison.json"
            pilot_comparison_path.write_bytes(_encoded(pilot_comparison))
            pilot_artifacts = {
                relative: {
                    "bytes": (root / relative).stat().st_size,
                    "sha256": adjudicator.sha256_file(root / relative),
                }
                for relative in adjudicator.BUNDLE_PILOT_ARTIFACTS
            }
            identity = {
                "kind": "development",
                "seed": adjudicator.DEVELOPMENT_SEEDS[0],
            }
            dataset_summary = {
                "payload_sha256": _digest("source-dataset"),
                "seed_identity": identity,
            }
            manifest = adjudicator.with_payload_hash(
                {
                    "protocol": adjudicator.TRAINER_BUNDLE_PROTOCOL,
                    "source_manifest_payload_sha256": dataset_summary["payload_sha256"],
                    "seed_identity": identity,
                    "data_replay_verification": {
                        "protocol": "R12-ACW-DATA-REPLAY-v1",
                        "seed_identity": identity,
                        "seed_fingerprint": _digest("seed"),
                        "source_manifest_payload_sha256": dataset_summary[
                            "payload_sha256"
                        ],
                        "regenerated_manifest_payload_sha256": dataset_summary[
                            "payload_sha256"
                        ],
                        "array_registry_sha256": _digest("array-registry"),
                        "arrays_verified": 9,
                        "public_arrays_verified": 7,
                        "oracle_arrays_verified": 2,
                    },
                    "query_schedule_sha256": hashlib.sha256(schedule_raw).hexdigest(),
                    "query_schedule_kind": "cgb_schedule.jsonl",
                    "pilot_report_payload_sha256": pilot_report["payload_sha256"],
                    "pilot_report_sha256": adjudicator.sha256_file(pilot_report_path),
                    "pilot_replay_comparison_payload_sha256": pilot_comparison[
                        "payload_sha256"
                    ],
                    "pilot_replay_comparison_sha256": adjudicator.sha256_file(
                        pilot_comparison_path
                    ),
                    "pilot_artifacts": pilot_artifacts,
                    "arrays": arrays,
                    "files": {"curriculum.jsonl": curriculum_record},
                    "oracle_paths_exported": 0,
                }
            )
            bundle_sources = {
                "pilot/report.json": ("artifacts/r12/acw_cgbr_pilot_v6/report.json"),
                "pilot/replay_comparison.json": (
                    "artifacts/r12/acw_cgbr_pilot_v6/replay_comparison.json"
                ),
                "pilot/cgb_schedule.jsonl": (
                    "artifacts/r12/acw_cgbr_pilot_v6/cgb_schedule.jsonl"
                ),
                "pilot/uniform_schedule.jsonl": (
                    "artifacts/r12/acw_cgbr_pilot_v6/uniform_schedule.jsonl"
                ),
            }
            anchor = {
                "activation_commit": "d" * 40,
                "anchor_commit": "e" * 40,
                "scientific_identity": pilot_identity,
                "activation_scientific_identity": {
                    "scientific_commit": "d" * 40,
                    "scientific_path_sha256": pilot_identity["scientific_path_sha256"],
                },
                "registry_raw_sha256": "f" * 64,
                "artifact_files": {
                    source: pilot_artifacts[bundle]
                    for bundle, source in bundle_sources.items()
                },
                "bundle_sources": bundle_sources,
            }
            with mock.patch.object(
                adjudicator,
                "_load_adjudicator_pilot_anchor",
                return_value=anchor,
            ):
                anchored_summary = adjudicator._validate_trainer_bundle(
                    root,
                    manifest,
                    {
                        "arrays": {
                            relative: arrays[relative]
                            for relative in adjudicator.BUNDLE_ARRAYS[:5]
                        }
                    },
                    dataset_summary,
                    "bundle",
                )
            self.assertEqual(
                anchored_summary["activation_commit"], anchor["activation_commit"]
            )
            self.assertEqual(
                anchored_summary["pilot_registry_raw_sha256"],
                anchor["registry_raw_sha256"],
            )

            summary = adjudicator._validate_unanchored_trainer_bundle_structure(
                root,
                manifest,
                {
                    "arrays": {
                        relative: arrays[relative]
                        for relative in adjudicator.BUNDLE_ARRAYS[:5]
                    }
                },
                dataset_summary,
                "bundle",
            )
            self.assertEqual(summary["arrays_hashed_and_opened"], 7)
            self.assertEqual(summary["pilot_artifacts_opened"], 4)
            self.assertEqual(summary["curriculum_sha256"], curriculum_record["sha256"])

            forked = copy.deepcopy(manifest)
            forked["query_schedule_sha256"] = _digest("unbound-schedule")
            forked = adjudicator.with_payload_hash(forked)
            with self.assertRaises(adjudicator.EvidenceError) as schedule_error:
                adjudicator._validate_unanchored_trainer_bundle_structure(
                    root,
                    forked,
                    {
                        "arrays": {
                            relative: arrays[relative]
                            for relative in adjudicator.BUNDLE_ARRAYS[:5]
                        }
                    },
                    dataset_summary,
                    "bundle",
                )
            self.assertEqual(
                schedule_error.exception.code, "bundle_schedule_binding_mismatch"
            )

            (root / "public/event_addresses.npy").write_bytes(b"corrupt")
            with self.assertRaises(adjudicator.EvidenceError) as raised:
                adjudicator._validate_unanchored_trainer_bundle_structure(
                    root,
                    manifest,
                    {
                        "arrays": {
                            relative: arrays[relative]
                            for relative in adjudicator.BUNDLE_ARRAYS[:5]
                        }
                    },
                    dataset_summary,
                    "bundle",
                )
        self.assertEqual(raised.exception.code, "array_artifact_mismatch")


if __name__ == "__main__":
    unittest.main()
