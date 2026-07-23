from __future__ import annotations

from dataclasses import replace
import json

import pytest

from pipeline.generate_episode_action_binding_corpus import (
    CASES_PER_CLUSTER,
    EpisodeCorpusError,
    generate_corpus,
    validate_corpus,
    verify_bundle,
    write_corpus,
)


def test_small_corpus_passes_split_and_symmetry_receipts() -> None:
    corpus = generate_corpus(seed=2026072391, train_clusters=8, development_clusters=4)
    manifest = corpus.manifest
    assert manifest.total_clusters == 12
    assert manifest.total_packets == 12 * CASES_PER_CLUSTER
    assert manifest.unique_packet_count == manifest.total_packets
    assert manifest.complete_cluster_count == manifest.total_clusters
    assert manifest.oracle_agreement_count == manifest.total_packets
    assert manifest.exact_operator_family_overlap == 0
    assert manifest.exact_packet_overlap == 0
    assert manifest.action_agnostic_correct <= manifest.total_clusters * 2
    assert manifest.all_actions_union_correct <= manifest.total_clusters * 2
    assert manifest.query_order_bagging_correct <= manifest.total_clusters * 3


def test_corpus_is_deterministic() -> None:
    left = generate_corpus(seed=2026072392, train_clusters=4, development_clusters=2)
    right = generate_corpus(seed=2026072392, train_clusters=4, development_clusters=2)
    assert left == right


def test_model_packet_file_has_no_target_or_offline_fields(tmp_path) -> None:
    corpus = generate_corpus(seed=2026072393, train_clusters=3, development_clusters=2)
    output_dir = tmp_path / "corpus"
    bundle = write_corpus(corpus, output_dir)
    assert verify_bundle(output_dir) == bundle
    rows = [
        json.loads(line)
        for line in (output_dir / "model_packets.jsonl").read_text().splitlines()
    ]
    assert rows
    assert set(rows[0]) == {
        "packet_sha256",
        "partition",
        "tokens",
        "attention_mask",
    }
    forbidden = {"target", "binding", "operator", "latent", "cluster", "oracle"}
    assert not forbidden & set(rows[0])


def test_bundle_tamper_is_detected(tmp_path) -> None:
    corpus = generate_corpus(seed=2026072394, train_clusters=2, development_clusters=1)
    output_dir = tmp_path / "corpus"
    write_corpus(corpus, output_dir)
    with (output_dir / "target_labels.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(EpisodeCorpusError, match="hash mismatch"):
        verify_bundle(output_dir)


def test_manifest_count_tamper_fails_closed() -> None:
    corpus = generate_corpus(seed=2026072395, train_clusters=2, development_clusters=1)
    tampered = replace(
        corpus,
        manifest=replace(
            corpus.manifest,
            oracle_agreement_count=corpus.manifest.oracle_agreement_count - 1,
        ),
    )
    with pytest.raises(EpisodeCorpusError, match="oracles"):
        validate_corpus(tampered)


def test_refuses_to_overwrite_bundle(tmp_path) -> None:
    corpus = generate_corpus(seed=2026072396, train_clusters=2, development_clusters=1)
    output_dir = tmp_path / "corpus"
    write_corpus(corpus, output_dir)
    with pytest.raises(FileExistsError):
        write_corpus(corpus, output_dir)


def test_both_partitions_are_required() -> None:
    with pytest.raises(EpisodeCorpusError, match="both partitions"):
        generate_corpus(train_clusters=0, development_clusters=1)
    with pytest.raises(EpisodeCorpusError, match="both partitions"):
        generate_corpus(train_clusters=1, development_clusters=0)
