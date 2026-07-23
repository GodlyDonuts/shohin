"""Generate the split-isolated EPISODE action-binding mechanics corpus.

Model packets, optimizer labels, and assessor-only ledgers are written to
separate files. The model-visible payload remains token IDs plus an attention
mask. This artifact qualifies a mechanics board; it is not neural evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

from pipeline.episode_action_binding_board import (
    ACTION_COUNT,
    CyclicOrderCluster,
    EpisodeCase,
    action_agnostic_baseline,
    all_actions_union_baseline,
    binding_enumerator_oracle,
    erase_demonstration_actions,
    generate_cyclic_order_cluster,
    model_packet_payload,
    query_order_bagging_baseline,
    raw_token_histogram,
    validate_cyclic_order_cluster,
    visible_table_oracle,
    world_commitment,
)


DEFAULT_SEED = 202_607_230_9
DEFAULT_TRAIN_CLUSTERS = 256
DEFAULT_DEVELOPMENT_CLUSTERS = 64
CASES_PER_CLUSTER = 6


class EpisodeCorpusError(ValueError):
    """A corpus custody or split-isolation invariant failed."""


@dataclass(frozen=True)
class PacketRow:
    packet_sha256: str
    partition: str
    tokens: tuple[int, ...]
    attention_mask: tuple[int, ...]


@dataclass(frozen=True)
class TargetRow:
    packet_sha256: str
    target_token: int


@dataclass(frozen=True)
class OfflineRow:
    packet_sha256: str
    partition: str
    cluster_id: str
    cluster_index: int
    query_variant: str
    binding_shift: int
    target_token: int
    query_start_state: int
    query_action_indices: tuple[int, ...]
    physical_operators: tuple[tuple[int, ...], ...]
    state_tokens: tuple[int, ...]
    action_tokens: tuple[int, ...]
    world_commitment: str


@dataclass(frozen=True)
class CorpusManifest:
    schema: str
    seed: int
    train_clusters: int
    development_clusters: int
    total_clusters: int
    cases_per_cluster: int
    train_packets: int
    development_packets: int
    total_packets: int
    unique_packet_count: int
    complete_cluster_count: int
    oracle_agreement_count: int
    exact_operator_family_overlap: int
    exact_packet_overlap: int
    cyclic_histogram_invariant_count: int
    cyclic_action_erasure_invariant_count: int
    late_query_world_commitment_count: int
    action_agnostic_correct: int
    all_actions_union_correct: int
    query_order_bagging_correct: int
    action_agnostic_ceiling: float
    query_order_bagging_ceiling: float
    model_payload_sha256: str
    target_labels_sha256: str
    offline_ledger_sha256: str


@dataclass(frozen=True)
class EpisodeCorpus:
    packets: tuple[PacketRow, ...]
    targets: tuple[TargetRow, ...]
    offline: tuple[OfflineRow, ...]
    manifest: CorpusManifest


def generate_corpus(
    *,
    seed: int = DEFAULT_SEED,
    train_clusters: int = DEFAULT_TRAIN_CLUSTERS,
    development_clusters: int = DEFAULT_DEVELOPMENT_CLUSTERS,
) -> EpisodeCorpus:
    """Generate and audit one deterministic split-isolated corpus."""

    if train_clusters <= 0 or development_clusters <= 0:
        raise EpisodeCorpusError("both partitions require at least one cluster")

    packets: list[PacketRow] = []
    targets: list[TargetRow] = []
    offline: list[OfflineRow] = []
    signatures: dict[str, set[str]] = {"train": set(), "development": set()}
    packet_digests: dict[str, set[str]] = {"train": set(), "development": set()}
    complete_clusters = 0
    oracle_agreement = 0
    histogram_invariants = 0
    erasure_invariants = 0
    world_commitments = 0
    action_agnostic_correct = 0
    union_correct = 0
    order_bagging_correct = 0

    global_signatures: set[str] = set()
    for partition, cluster_count in (
        ("train", train_clusters),
        ("development", development_clusters),
    ):
        accepted = 0
        candidate_index = 0
        while accepted < cluster_count:
            candidate_seed = (
                seed
                + (0 if partition == "train" else 1_000_000_007)
                + candidate_index * 104_729
            )
            candidate_index += 1
            query_depth = 2 + accepted % 3 if partition == "train" else 5 + accepted % 2
            cluster = generate_cyclic_order_cluster(
                candidate_seed,
                query_depth=query_depth,
            )
            signature = _operator_family_signature(cluster)
            if signature in global_signatures:
                continue
            global_signatures.add(signature)
            signatures[partition].add(signature)

            cluster_id = _cluster_id(partition, accepted, cluster)
            (
                cluster_packets,
                cluster_targets,
                cluster_offline,
                cluster_counts,
            ) = _audit_and_flatten_cluster(
                cluster,
                partition=partition,
                cluster_index=accepted,
                cluster_id=cluster_id,
            )
            for row in cluster_packets:
                if row.packet_sha256 in packet_digests[partition]:
                    raise EpisodeCorpusError("duplicate packet within a partition")
                packet_digests[partition].add(row.packet_sha256)
            packets.extend(cluster_packets)
            targets.extend(cluster_targets)
            offline.extend(cluster_offline)
            complete_clusters += 1
            oracle_agreement += cluster_counts["oracle_agreement"]
            histogram_invariants += cluster_counts["histogram_invariants"]
            erasure_invariants += cluster_counts["erasure_invariants"]
            world_commitments += cluster_counts["world_commitments"]
            action_agnostic_correct += cluster_counts["action_agnostic_correct"]
            union_correct += cluster_counts["union_correct"]
            order_bagging_correct += cluster_counts["order_bagging_correct"]
            accepted += 1

    operator_overlap = len(signatures["train"] & signatures["development"])
    packet_overlap = len(packet_digests["train"] & packet_digests["development"])
    if operator_overlap or packet_overlap:
        raise EpisodeCorpusError("train/development split isolation failed")

    model_payload = [
        {
            "tokens": row.tokens,
            "attention_mask": row.attention_mask,
        }
        for row in packets
    ]
    target_payload = [asdict(row) for row in targets]
    offline_payload = [asdict(row) for row in offline]
    train_packets = train_clusters * CASES_PER_CLUSTER
    development_packets = development_clusters * CASES_PER_CLUSTER
    total_packets = train_packets + development_packets
    manifest = CorpusManifest(
        schema="episode_action_binding_corpus_v1",
        seed=seed,
        train_clusters=train_clusters,
        development_clusters=development_clusters,
        total_clusters=train_clusters + development_clusters,
        cases_per_cluster=CASES_PER_CLUSTER,
        train_packets=train_packets,
        development_packets=development_packets,
        total_packets=total_packets,
        unique_packet_count=len(set(row.packet_sha256 for row in packets)),
        complete_cluster_count=complete_clusters,
        oracle_agreement_count=oracle_agreement,
        exact_operator_family_overlap=operator_overlap,
        exact_packet_overlap=packet_overlap,
        cyclic_histogram_invariant_count=histogram_invariants,
        cyclic_action_erasure_invariant_count=erasure_invariants,
        late_query_world_commitment_count=world_commitments,
        action_agnostic_correct=action_agnostic_correct,
        all_actions_union_correct=union_correct,
        query_order_bagging_correct=order_bagging_correct,
        action_agnostic_ceiling=1.0 / ACTION_COUNT,
        query_order_bagging_ceiling=0.5,
        model_payload_sha256=_json_sha256(model_payload),
        target_labels_sha256=_json_sha256(target_payload),
        offline_ledger_sha256=_json_sha256(offline_payload),
    )
    corpus = EpisodeCorpus(
        packets=tuple(packets),
        targets=tuple(targets),
        offline=tuple(offline),
        manifest=manifest,
    )
    validate_corpus(corpus)
    return corpus


def validate_corpus(corpus: EpisodeCorpus) -> None:
    """Fail closed on every frozen corpus-level invariant."""

    manifest = corpus.manifest
    if len(corpus.packets) != manifest.total_packets:
        raise EpisodeCorpusError("packet count does not match manifest")
    if len(corpus.targets) != manifest.total_packets:
        raise EpisodeCorpusError("target count does not match manifest")
    if len(corpus.offline) != manifest.total_packets:
        raise EpisodeCorpusError("offline ledger count does not match manifest")
    if manifest.unique_packet_count != manifest.total_packets:
        raise EpisodeCorpusError("packet digests are not unique")
    if manifest.complete_cluster_count != manifest.total_clusters:
        raise EpisodeCorpusError("one or more six-case clusters is incomplete")
    if manifest.oracle_agreement_count != manifest.total_packets:
        raise EpisodeCorpusError("independent oracles do not agree on every packet")
    if manifest.exact_operator_family_overlap != 0:
        raise EpisodeCorpusError("operator family leaks across partitions")
    if manifest.exact_packet_overlap != 0:
        raise EpisodeCorpusError("packet leaks across partitions")
    expected_triples = manifest.total_clusters * 2
    expected_pairs = manifest.total_clusters * ACTION_COUNT
    if manifest.cyclic_histogram_invariant_count != expected_triples:
        raise EpisodeCorpusError("cyclic token histogram invariant failed")
    if manifest.cyclic_action_erasure_invariant_count != expected_triples:
        raise EpisodeCorpusError("cyclic action-erasure invariant failed")
    if manifest.late_query_world_commitment_count != expected_pairs:
        raise EpisodeCorpusError("late-query world commitment invariant failed")
    if manifest.action_agnostic_correct > expected_triples:
        raise EpisodeCorpusError("action-agnostic control exceeds one-third")
    if manifest.all_actions_union_correct > expected_triples:
        raise EpisodeCorpusError("all-actions control exceeds one-third")
    if manifest.query_order_bagging_correct > expected_pairs:
        raise EpisodeCorpusError("query-order control exceeds one-half")
    target_by_digest = {row.packet_sha256: row.target_token for row in corpus.targets}
    if len(target_by_digest) != manifest.total_packets:
        raise EpisodeCorpusError("target labels contain duplicate packet digests")
    if {row.packet_sha256 for row in corpus.packets} != set(target_by_digest):
        raise EpisodeCorpusError("model packets and target labels do not join exactly")
    for row in corpus.packets:
        if set(asdict(row)) != {
            "packet_sha256",
            "partition",
            "tokens",
            "attention_mask",
        }:
            raise EpisodeCorpusError("model packet row contains an unexpected field")


def write_corpus(corpus: EpisodeCorpus, output_dir: Path) -> dict[str, object]:
    """Write one fsync'd atomic corpus bundle and return its manifest."""

    validate_corpus(corpus)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        paths = {
            "model_packets.jsonl": [asdict(row) for row in corpus.packets],
            "target_labels.jsonl": [asdict(row) for row in corpus.targets],
            "offline_ledger.jsonl": [asdict(row) for row in corpus.offline],
        }
        artifact_hashes: dict[str, str] = {}
        for filename, rows in paths.items():
            path = staging / filename
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(
                        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            artifact_hashes[filename] = _file_sha256(path)

        manifest_path = staging / "manifest.json"
        manifest_payload = asdict(corpus.manifest)
        _write_json_fsync(manifest_path, manifest_payload)
        artifact_hashes["manifest.json"] = _file_sha256(manifest_path)
        bundle = {
            "schema": "episode_action_binding_bundle_v1",
            "artifacts": artifact_hashes,
        }
        _write_json_fsync(staging / "bundle_manifest.json", bundle)
        os.replace(staging, output_dir)
        _fsync_directory(output_dir.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_bundle(output_dir)


def verify_bundle(output_dir: Path) -> dict[str, object]:
    """Verify every artifact named by the bundle manifest."""

    bundle_path = output_dir / "bundle_manifest.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("schema") != "episode_action_binding_bundle_v1":
        raise EpisodeCorpusError("unknown bundle schema")
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise EpisodeCorpusError("bundle artifact map is missing")
    for filename, expected in artifacts.items():
        path = output_dir / filename
        if not path.is_file() or _file_sha256(path) != expected:
            raise EpisodeCorpusError(f"artifact hash mismatch: {filename}")
    return bundle


def _audit_and_flatten_cluster(
    cluster: CyclicOrderCluster,
    *,
    partition: str,
    cluster_index: int,
    cluster_id: str,
) -> tuple[
    list[PacketRow],
    list[TargetRow],
    list[OfflineRow],
    dict[str, int],
]:
    validate_cyclic_order_cluster(cluster)
    packets: list[PacketRow] = []
    targets: list[TargetRow] = []
    offline: list[OfflineRow] = []
    counts = {
        "oracle_agreement": 0,
        "histogram_invariants": 0,
        "erasure_invariants": 0,
        "world_commitments": 0,
        "action_agnostic_correct": 0,
        "union_correct": 0,
        "order_bagging_correct": 0,
    }

    for query_variant, group in (
        ("primary", cluster.primary),
        ("reordered", cluster.reordered),
    ):
        histograms = [raw_token_histogram(case.packet) for case in group.variants]
        erasures = [erase_demonstration_actions(case.packet) for case in group.variants]
        if not all(item == histograms[0] for item in histograms):
            raise EpisodeCorpusError("cyclic histogram mismatch")
        if not all(item == erasures[0] for item in erasures):
            raise EpisodeCorpusError("cyclic action-erasure mismatch")
        counts["histogram_invariants"] += 1
        counts["erasure_invariants"] += 1

        for case in group.variants:
            digest = _packet_sha256(case)
            visible = visible_table_oracle(case.packet)
            enumerated = binding_enumerator_oracle(case.packet, group.system)
            if visible != enumerated or visible != case.target_token:
                raise EpisodeCorpusError("independent oracle disagreement")
            counts["oracle_agreement"] += 1
            counts["action_agnostic_correct"] += int(
                action_agnostic_baseline(case.packet) == case.target_token
            )
            counts["union_correct"] += int(
                all_actions_union_baseline(case.packet) == case.target_token
            )
            counts["order_bagging_correct"] += int(
                query_order_bagging_baseline(case.packet) == case.target_token
            )
            payload = model_packet_payload(case.packet)
            packets.append(
                PacketRow(
                    packet_sha256=digest,
                    partition=partition,
                    tokens=payload["tokens"],
                    attention_mask=payload["attention_mask"],
                )
            )
            targets.append(
                TargetRow(packet_sha256=digest, target_token=case.target_token)
            )
            offline.append(
                OfflineRow(
                    packet_sha256=digest,
                    partition=partition,
                    cluster_id=cluster_id,
                    cluster_index=cluster_index,
                    query_variant=query_variant,
                    binding_shift=case.binding_shift,
                    target_token=case.target_token,
                    query_start_state=group.query_start_state,
                    query_action_indices=group.query_action_indices,
                    physical_operators=group.system.physical_operators,
                    state_tokens=group.system.state_tokens,
                    action_tokens=group.system.action_tokens,
                    world_commitment=world_commitment(case.packet),
                )
            )

    for left, right in zip(
        cluster.primary.variants,
        cluster.reordered.variants,
        strict=True,
    ):
        if world_commitment(left.packet) != world_commitment(right.packet):
            raise EpisodeCorpusError("paired query orders change the committed world")
        counts["world_commitments"] += 1
    return packets, targets, offline, counts


def _operator_family_signature(cluster: CyclicOrderCluster) -> str:
    operators = sorted(cluster.primary.system.physical_operators)
    return _json_sha256(operators)


def _cluster_id(
    partition: str,
    index: int,
    cluster: CyclicOrderCluster,
) -> str:
    return hashlib.sha256(
        (
            f"{partition}:{index}:"
            f"{cluster.primary.group_digest}:{cluster.reordered.group_digest}"
        ).encode("ascii")
    ).hexdigest()


def _packet_sha256(case: EpisodeCase) -> str:
    return _json_sha256(model_packet_payload(case.packet))


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_fsync(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--train-clusters",
        type=int,
        default=DEFAULT_TRAIN_CLUSTERS,
    )
    parser.add_argument(
        "--development-clusters",
        type=int,
        default=DEFAULT_DEVELOPMENT_CLUSTERS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    corpus = generate_corpus(
        seed=args.seed,
        train_clusters=args.train_clusters,
        development_clusters=args.development_clusters,
    )
    bundle = write_corpus(corpus, args.output_dir)
    print(json.dumps({"manifest": asdict(corpus.manifest), "bundle": bundle}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
