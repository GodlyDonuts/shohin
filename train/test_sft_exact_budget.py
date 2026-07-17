import base64
import copy
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import models

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sft  # noqa: E402


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class _Tokenizer:
    def encode(self, text):
        words = text.split()
        return _Encoding([sum(word.encode("utf-8")) % 251 + 1 for word in words])


def test_frozen_factorial_budget_and_completion_contract():
    assert sft.CANONICAL_EXACT_BUDGET == {
        "updates": 1560,
        "batch_size": 16,
        "pack_len": 2048,
        "seed": 1337,
        "epochs": 1,
        "lr_muon": 1e-3,
        "lr_adam": 2e-4,
        "warmup": 50,
        "clip": 1.0,
    }
    assert 1560 * 16 == sft.FACTORIAL_REQUIRED_PACKS == 24_960
    assert 1560 * 16 * 2048 == 51_118_080
    _, token_ids, mask = sft.encode_supervised_example(
        _Tokenizer(), "Question: q\nAnswer:", " response", 999
    )
    assert token_ids[-1] == 999
    assert mask[-1] == 1


def _write_rows(path, count=12):
    with path.open("w") as sink:
        for index in range(count):
            row = {"question": f"q{index} x", "response": f"r{index} y"}
            sink.write(json.dumps(row, sort_keys=True) + "\n")


def test_build_packed_is_deterministic_and_reports_unpacked_tail(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_rows(corpus)
    args = ([str(corpus)], _Tokenizer(), 8, ["question"], ["response"], 999)

    first = sft.build_packed(*args, return_stats=True)
    second = sft.build_packed(*args, return_stats=True)
    x1, y1, groups1, stats1 = first
    x2, y2, groups2, stats2 = second

    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(groups1, groups2)
    assert stats1 == stats2
    assert stats1["packed_forward_positions"] == len(x1) * 8
    assert stats1["packed_supervised_tokens"] == int(np.count_nonzero(y1 != -1))
    assert stats1["groups"]["default"]["unpacked_tail_tokens"] >= 0
    assert len(stats1["packing_sha256"]) == 64


def test_exact_packing_includes_a_complete_window_with_one_lookahead(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    with corpus.open("w") as sink:
        for index in range(2):
            sink.write(
                json.dumps({"question": f"q{index}", "response": f"r{index}"}) + "\n"
            )

    x, y, _, stats = sft.build_packed(
        [str(corpus)],
        _Tokenizer(),
        9,
        ["question"],
        ["response"],
        999,
        return_stats=True,
    )
    assert x.shape == (1, 9)
    assert y.shape == (1, 9)
    assert stats["groups"]["default"]["unpacked_tail_tokens"] == 0


def test_exact_budget_plan_consumes_prefix_and_explicitly_ledgers_drops():
    supervised = np.asarray([3, 5, 7, 11, 13, 17, 19], dtype=np.int64)
    order1, plan1 = sft.make_exact_budget_plan(supervised, 3, 2, 8, 1337)
    order2, plan2 = sft.make_exact_budget_plan(supervised, 3, 2, 8, 1337)

    np.testing.assert_array_equal(order1, order2)
    assert plan1 == plan2
    assert len(order1) == 6
    assert plan1["optimizer_updates"] == 3
    assert plan1["consumed_packs"] == 6
    assert plan1["forward_token_positions"] == 3 * 2 * 8
    assert plan1["dropped_packs_count"] == 1
    assert len(plan1["dropped_packs"]) == 1
    assert plan1["target_thinning"] is False
    assert plan1["supervised_token_equality_enforced"] is False
    assert plan1["consumed_supervised_tokens"] + plan1[
        "dropped_supervised_tokens"
    ] == int(supervised.sum())


def test_exact_budget_plan_fails_closed_when_corpus_is_short():
    with pytest.raises(ValueError, match="needs 8 packs"):
        sft.make_exact_budget_plan(np.ones(7, dtype=np.int64), 2, 4, 2048, 1337)


def test_exact_packing_rejects_any_silently_skipped_row():
    stats = {"packed_sequences": 4, "skipped": {"invalid_fields": 1, "too_long": 0}}
    with pytest.raises(ValueError, match="rejected 1 rows"):
        sft.validate_exact_packing_stats(stats)


def test_actual_receipt_requires_every_exact_counter_to_match():
    _, plan = sft.make_exact_budget_plan(np.arange(1, 9), 2, 3, 8, 1337)
    consumed = plan["consumed_supervised_tokens"]
    update_counts = plan["supervised_tokens_per_update"]
    actual = sft.validate_exact_budget_actual(plan, update_counts, 6, 48)
    assert actual["optimizer_updates"] == 2
    assert actual["supervised_tokens"] == consumed
    assert len(actual["supervised_tokens_per_update_sha256"]) == 64

    with pytest.raises(RuntimeError, match="execution mismatch"):
        sft.validate_exact_budget_actual(plan, [consumed], 6, 48)
    with pytest.raises(RuntimeError, match="execution mismatch"):
        sft.validate_exact_budget_actual(plan, update_counts, 6, 47)
    with pytest.raises(RuntimeError, match="execution mismatch"):
        sft.validate_exact_budget_actual(
            plan, [update_counts[0] + 1, update_counts[1] - 1], 6, 48
        )


def _canonical_args():
    return SimpleNamespace(
        arm="",
        batch_size=16,
        clip=1.0,
        compile=False,
        data=["corpus.jsonl"],
        eos="<|endoftext|>",
        epochs=1,
        exact_updates=1560,
        expected_data_sha256="a" * 64,
        expected_init_sha256="b" * 64,
        expected_production_admission_sha256="e" * 64,
        expected_reviewed_commit="f" * 40,
        expected_reviewed_source_manifest_sha256="1" * 64,
        expected_source_sha256=[
            f"{path}={'d' * 64}" for path in sft.CANONICAL_SOURCE_PATHS
        ],
        expected_tokenizer_sha256="c" * 64,
        freeze_lexicon=False,
        group_field=None,
        lr_adam=2e-4,
        lr_muon=1e-3,
        max_examples=0,
        pack_len=2048,
        prompt_override_field=None,
        production_admission="production_admission.json",
        q_fields=sft.DEFAULT_Q_FIELDS,
        r_fields=sft.DEFAULT_R_FIELDS,
        reference="",
        replay_batch_size=4,
        replay_max_tokens=128,
        replay_prompts="",
        replay_weight=0.0,
        reviewed_source_manifest="reviewed_source_manifest.json",
        sample_weights=[],
        seed=1337,
        snapshot_root="/private/snapshot",
        warmup=50,
    )


def test_canonical_settings_reject_every_scientific_override():
    args = _canonical_args()
    cfg = SimpleNamespace(seq_len=2048)
    sft.validate_canonical_exact_settings(args, cfg, ["corpus.jsonl"])

    args.compile = True
    with pytest.raises(ValueError, match="forbids: --compile"):
        sft.validate_canonical_exact_settings(args, cfg, ["corpus.jsonl"])
    args.compile = False
    args.lr_muon = 0.002
    with pytest.raises(ValueError, match="override rejected"):
        sft.validate_canonical_exact_settings(args, cfg, ["corpus.jsonl"])
    args.lr_muon = 1e-3
    args.pack_len = 0
    with pytest.raises(ValueError, match="--pack-len=0"):
        sft.validate_canonical_exact_settings(args, cfg, ["corpus.jsonl"])

    args = _canonical_args()
    args.expected_source_sha256.pop()
    with pytest.raises(ValueError, match="expected source set mismatch"):
        sft.validate_canonical_exact_settings(args, cfg, ["corpus.jsonl"])

    args = _canonical_args()
    args.arm = "iid"
    with pytest.raises(ValueError, match="forbids caller arm labels"):
        sft.validate_canonical_exact_settings(args, cfg, ["corpus.jsonl"])


def _immutable_snapshot(logical_path, payload):
    digest = sft._sha256_bytes(payload)
    return sft.immutable_snapshot_from_bytes(
        logical_path,
        payload,
        digest,
        origin="test_exact_bytes_v1",
    )


def _admission_fixture():
    repo_root = Path(__file__).resolve().parent.parent
    source_snapshots = {
        relative: _immutable_snapshot(relative, (repo_root / relative).read_bytes())
        for relative in sft.CANONICAL_SOURCE_PATHS
    }
    source_hashes = {
        relative: snapshot.sha256 for relative, snapshot in source_snapshots.items()
    }
    commit = "f" * 40
    reviewed_manifest_value = {
        "clean_source_tree": True,
        "remote_attestation": False,
        "review_status": "approved",
        "reviewed_clean_commit": commit,
        "schema": sft.REVIEWED_SOURCE_MANIFEST_SCHEMA,
        "sources": source_hashes,
    }
    reviewed_manifest_snapshot = _immutable_snapshot(
        "reviewed_source_manifest.json",
        sft.canonical_json_bytes(reviewed_manifest_value),
    )
    rows = []
    for index, source in enumerate(sorted(sft.FACTORIAL_ROW_SOURCES)):
        rows.append(
            {
                "arm": "iid",
                "question": f"q{index}",
                "response": f"r{index}",
                "schema": sft.FACTORIAL_SCHEMA,
                "seed": sft.FACTORIAL_ARM_SEEDS["iid"],
                "source": source,
                "split": "train",
                "term_factor": False,
                "training_group": sft.FACTORIAL_TRAINING_GROUP,
                "width_factor": False,
            }
        )
    corpus_snapshot = _immutable_snapshot(
        "corpus.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode(),
    )
    tokenizer_snapshot = _immutable_snapshot("tokenizer.json", b'{"tokenizer":true}\n')

    generator_paths = (
        "pipeline/generate_digitwise_factorial_v4.py",
        "pipeline/generate_digitwise_recurrent_v1.py",
        "pipeline/test_generate_digitwise_factorial_v4.py",
        "train/digitwise_protocol.py",
    )
    auditor_paths = (
        "pipeline/audit_digitwise_factorial_v4.py",
        "pipeline/test_audit_digitwise_factorial_v4.py",
        "train/digitwise_protocol.py",
        "train/sft.py",
        "train/sft_encoding.py",
        "train/test_sft_exact_budget.py",
        "train/jobs/sft_factorial.sbatch",
    )

    def manifest(paths):
        return {
            "capture": "one_pass_exact_bytes_consumed_by_frozen_cli_v1",
            "sources": {
                relative: {
                    "bytes": len(source_snapshots[relative].payload),
                    "sha256": source_snapshots[relative].sha256,
                }
                for relative in paths
            },
        }

    generator_manifest = manifest(generator_paths)
    auditor_manifest = manifest(auditor_paths)
    packing = {
        "examples": len(rows),
        "groups": {"default": {"packed_sequences": 1}},
        "pack_len": 2048,
        "packing_sha256": "9" * 64,
        "packed_sequences": sft.FACTORIAL_REQUIRED_PACKS,
        "skipped": {"blank_lines": 0, "invalid_fields": 0, "too_long": 0},
    }
    test_target = {"episodes": 1, "rows": len(rows), "transitions": 2}
    frozen_input = {
        "bytes": 1,
        "capture": "one_pass_stable_descriptor_to_immutable_bytes_v1",
        "sha256": "2" * 64,
    }
    receipt = {
        "admission_pass": True,
        "admitted_arm": "iid",
        "audit": sft.PRODUCTION_ADMISSION_AUDIT,
        "checks": {"all_scientific_gates": True},
        "contamination": {
            "examples": [],
            "gates": {
                "exact_normalized_prompt_clear": True,
                "literal_normalized_word_13gram_clear": True,
                "reserved_signature_clear": True,
            },
            "heldout_answer_boundary": {
                "answer_values_retained_for_training": False,
                "training_rows_constructed_by_auditor": False,
            },
            "train_heldout_exact_normalized_prompt_hits": 0,
            "train_heldout_literal_13gram_hits": 0,
            "train_heldout_reserved_signature_hits": 0,
        },
        "data_sha256": corpus_snapshot.sha256,
        "declared_arm": "iid",
        "declared_factors": {"term": False, "width": False},
        "failures": {},
        "generator_reports": {
            role: {
                "bytes": 100,
                "runtime": {
                    "python": "3.test",
                    "python_implementation": "CPython",
                },
                "sha256": ("a" if role == "primary" else "b") * 64,
                "source_manifest": generator_manifest,
                "validated": True,
            }
            for role in ("primary", "counterpart")
        },
        "heldout": {
            **sft.FROZEN_FACTORIAL_HELDOUT_COUNTS,
            "blank_lines": 0,
            "frozen_sha256_required": sft.FROZEN_FACTORIAL_HELDOUT_SHA256,
            "identity_digests": {
                "branch_ids_sha256": "3" * 64,
                "normalized_prompts_sha256": "4" * 64,
                "reserved_signatures_sha256": "5" * 64,
            },
            "regimes": sft.FROZEN_FACTORIAL_HELDOUT_REGIMES,
            "sha256": sft.FROZEN_FACTORIAL_HELDOUT_SHA256,
        },
        "independent_audit": {
            "auditor_recomputed_solver_and_contract": True,
            "generator_implementation_imported": False,
            "generator_reports_used_only_as_bound_provenance": True,
            "remote_attestation": False,
        },
        "inputs": {
            "data": {
                "bytes": len(corpus_snapshot.payload),
                "sha256": corpus_snapshot.sha256,
            },
            "episodes": dict(frozen_input),
            "heldout": {
                "bytes": 1,
                "capture": frozen_input["capture"],
                "sha256": sft.FROZEN_FACTORIAL_HELDOUT_SHA256,
            },
            "paired_data": {**frozen_input, "sha256": "6" * 64},
            "paired_episodes": {**frozen_input, "sha256": "7" * 64},
            "tokenizer": {
                "bytes": len(tokenizer_snapshot.payload),
                "sha256": tokenizer_snapshot.sha256,
            },
        },
        "mechanical_pass": True,
        "mode": "production",
        "board": "narrow",
        "paired_arm": "term",
        "production_admission": True,
        "production_contract": True,
        "receipt_schema": sft.PRODUCTION_ADMISSION_SCHEMA,
        "rows_observed": {"raw": len(rows), "valid": len(rows)},
        "runtime": {
            "python": "3.test",
            "python_implementation": "CPython",
            "tokenizers": sft.tokenizers.__version__,
        },
        "schema": sft.FACTORIAL_SCHEMA,
        "seed": sft.FACTORIAL_ARM_SEEDS["iid"],
        "scientific_contract": {
            "heldout_splits": [
                "fit_w4",
                "fit_w6",
                "value_ood_w4",
                "value_ood_w6",
                "width_ood_w8",
            ],
            "row_sources": sorted(sft.FACTORIAL_ROW_SOURCES),
            "schema": sft.FACTORIAL_SCHEMA,
            "training_group": sft.FACTORIAL_TRAINING_GROUP,
            "training_split": "train",
        },
        "source_manifests": {
            "auditor": auditor_manifest,
            "generator": generator_manifest,
        },
        "tokenizer_accounting": {
            "encoding_boundary": "Canonical completion encoding; EOS is supervised.",
            "overall": {"rows_seen": len(rows)},
            "pack_length": 2048,
            "production_build_packed": packing,
            "tokenizer_bytes": len(tokenizer_snapshot.payload),
            "tokenizer_sha256": tokenizer_snapshot.sha256,
            "tokenizers_version": sft.tokenizers.__version__,
        },
        "target": test_target,
        "test_scale": None,
    }

    def receipt_snapshot(value):
        return _immutable_snapshot(
            "production_admission.json", sft.canonical_json_bytes(value)
        )

    return {
        "commit": commit,
        "corpus": corpus_snapshot,
        "manifest": reviewed_manifest_snapshot,
        "manifest_value": reviewed_manifest_value,
        "packing": packing,
        "receipt": receipt,
        "receipt_snapshot": receipt_snapshot,
        "sources": source_snapshots,
        "tokenizer": tokenizer_snapshot,
        "target": test_target,
    }


def test_production_admission_derives_arm_and_rejects_every_no_go_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _admission_fixture()
    monkeypatch.setattr(sft, "FACTORIAL_PRODUCTION_TARGET", fixture["target"])
    reviewed = sft.validate_reviewed_source_manifest(
        fixture["manifest"], fixture["commit"], fixture["sources"]
    )
    arm, receipt, corpus = sft.validate_production_admission(
        fixture["receipt_snapshot"](fixture["receipt"]),
        fixture["corpus"],
        fixture["tokenizer"],
        fixture["sources"],
        reviewed,
    )
    assert arm == corpus["arm"] == "iid"
    assert receipt["admitted_arm"] == "iid"
    assert sft.validate_admitted_packing(receipt, fixture["packing"]) == "9" * 64

    mutations = (
        (
            "wrong_arm",
            lambda value: value.update(admitted_arm="term", declared_arm="term"),
        ),
        (
            "production_admission_false",
            lambda value: value.update(production_admission=False),
        ),
        (
            "stale_data",
            lambda value: value["inputs"]["data"].update(sha256="0" * 64),
        ),
        (
            "contamination_not_clear",
            lambda value: value["contamination"].update(
                train_heldout_reserved_signature_hits=1,
                examples=[{"kind": "reserved_signature"}],
            ),
        ),
    )
    for expected, mutate in mutations:
        candidate = copy.deepcopy(fixture["receipt"])
        mutate(candidate)
        with pytest.raises(ValueError, match=expected):
            sft.validate_production_admission(
                fixture["receipt_snapshot"](candidate),
                fixture["corpus"],
                fixture["tokenizer"],
                fixture["sources"],
                reviewed,
            )

    missing_dependency = copy.deepcopy(fixture["receipt"])
    del missing_dependency["source_manifests"]["generator"]["sources"][
        "pipeline/generate_digitwise_recurrent_v1.py"
    ]
    with pytest.raises(ValueError, match="generator_manifest_source_set"):
        sft.validate_production_admission(
            fixture["receipt_snapshot"](missing_dependency),
            fixture["corpus"],
            fixture["tokenizer"],
            fixture["sources"],
            reviewed,
        )
    with pytest.raises(ValueError, match="packing differs"):
        sft.validate_admitted_packing(
            receipt, {**fixture["packing"], "packing_sha256": "8" * 64}
        )


def test_production_admission_expected_sha_snapshot_defeats_path_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _admission_fixture()
    monkeypatch.setattr(sft, "FACTORIAL_PRODUCTION_TARGET", fixture["target"])
    reviewed = sft.validate_reviewed_source_manifest(
        fixture["manifest"], fixture["commit"], fixture["sources"]
    )
    admitted_bytes = sft.canonical_json_bytes(fixture["receipt"])
    path = tmp_path / "production_admission.json"
    path.write_bytes(admitted_bytes)
    path.chmod(0o444)
    expected = sft._sha256_bytes(admitted_bytes)
    frozen = sft.freeze_file_bytes(path, expected, tmp_path)
    before = path.stat()
    forged = copy.deepcopy(fixture["receipt"])
    forged["production_admission"] = False
    try:
        path.chmod(0o600)
        path.write_bytes(sft.canonical_json_bytes(forged))
        arm, receipt, _ = sft.validate_production_admission(
            frozen,
            fixture["corpus"],
            fixture["tokenizer"],
            fixture["sources"],
            reviewed,
        )
        assert arm == "iid"
        assert receipt["production_admission"] is True
    finally:
        path.write_bytes(admitted_bytes)
        path.chmod(before.st_mode & 0o777)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="hash mismatch"):
        sft.freeze_file_bytes(path, "0" * 64, tmp_path)
    assert frozen.payload == admitted_bytes
    assert path.read_bytes() == admitted_bytes


def test_reviewed_manifest_and_source_bundle_preserve_exact_recoverable_bytes():
    fixture = _admission_fixture()
    reviewed = sft.validate_reviewed_source_manifest(
        fixture["manifest"], fixture["commit"], fixture["sources"]
    )
    assert reviewed == fixture["manifest_value"]
    bundle = sft.scientific_source_bundle(fixture["sources"])
    assert set(bundle["sources"]) == set(sft.CANONICAL_SOURCE_PATHS)
    for relative, record in bundle["sources"].items():
        recovered = base64.b64decode(record["payload_base64"], validate=True)
        assert recovered == fixture["sources"][relative].payload
        assert sft._sha256_bytes(recovered) == record["sha256"]


def test_canonical_json_publication_is_exact_and_no_replace(tmp_path):
    path = tmp_path / "receipt.json"
    value = {"z": 1, "a": [3, 2, 1]}
    artifact = sft.publish_canonical_json(path, value)
    try:
        expected = b'{"a":[3,2,1],"z":1}\n'
        assert path.read_bytes() == expected
        assert artifact.sha256 == sft.sha256_file(path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
        with pytest.raises(FileExistsError):
            sft.publish_canonical_json(path, {"substituted": True})
        assert path.read_bytes() == expected
        assert not list(tmp_path.glob(".receipt.json.tmp.*"))
    finally:
        artifact.close()


def test_immutable_corpus_snapshot_defeats_reviewer_path_substitution(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    corpus = snapshot / "corpus.jsonl"
    _write_rows(corpus, count=12)
    corpus.chmod(0o444)
    admitted_sha = sft.sha256_file(corpus)
    frozen = sft.freeze_file_bytes(corpus, admitted_sha, snapshot)

    replacement = snapshot / "replacement.jsonl"
    _write_rows(replacement, count=2)
    replacement.chmod(0o444)
    os.replace(replacement, corpus)
    _, _, _, stats = sft.build_packed(
        [frozen.open_bytes()],
        _Tokenizer(),
        8,
        ["question"],
        ["response"],
        999,
        return_stats=True,
    )
    assert stats["examples"] == 12
    with corpus.open() as replacement_source:
        assert sum(1 for _ in replacement_source) == 2
    assert frozen.verify_bytes() == admitted_sha
    assert frozen.binding["immutable_byte_snapshot"] is True
    assert frozen.binding["live_descriptor_retained"] is False


def test_init_and_tokenizer_load_from_admitted_immutable_bytes(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    init = snapshot / "init.pt"
    sft.torch.save({"marker": "admitted"}, init)
    init.chmod(0o444)
    frozen_init = sft.freeze_file_bytes(init, sft.sha256_file(init), snapshot)
    replacement_init = snapshot / "replacement.pt"
    sft.torch.save({"marker": "substituted"}, replacement_init)
    replacement_init.chmod(0o444)
    os.replace(replacement_init, init)
    assert sft.torch.load(frozen_init.open_bytes(), map_location="cpu")["marker"] == (
        "admitted"
    )
    assert sft.torch.load(init, map_location="cpu")["marker"] == "substituted"
    assert frozen_init.verify_bytes() == frozen_init.sha256

    tokenizer_path = snapshot / "tokenizer.json"
    admitted_tokenizer = sft.Tokenizer(
        models.WordLevel({"[UNK]": 0, "admitted": 1}, unk_token="[UNK]")
    )
    admitted_tokenizer.save(str(tokenizer_path))
    tokenizer_path.chmod(0o444)
    frozen_tokenizer = sft.freeze_file_bytes(
        tokenizer_path, sft.sha256_file(tokenizer_path), snapshot
    )
    replacement_path = snapshot / "replacement-tokenizer.json"
    replacement_tokenizer = sft.Tokenizer(
        models.WordLevel({"[UNK]": 0, "substituted": 1}, unk_token="[UNK]")
    )
    replacement_tokenizer.save(str(replacement_path))
    replacement_path.chmod(0o444)
    os.replace(replacement_path, tokenizer_path)
    consumed = sft.Tokenizer.from_str(frozen_tokenizer.payload.decode("utf-8"))
    assert consumed.token_to_id("admitted") == 1
    assert consumed.token_to_id("substituted") is None
    assert sft.Tokenizer.from_file(str(tokenizer_path)).token_to_id("substituted") == 1
    assert frozen_tokenizer.verify_bytes() == frozen_tokenizer.sha256


def test_mutate_consume_restore_exploit_cannot_change_frozen_consumption(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    source = snapshot / "corpus.jsonl"
    original = b'{"question":"q x","response":"original"}\n'
    altered = b'{"question":"q x","response":"altered!"}\n'
    assert len(original) == len(altered)
    source.write_bytes(original)
    source.chmod(0o444)
    admitted_sha = sft.sha256_file(source)
    frozen = sft.freeze_file_bytes(source, admitted_sha, snapshot)
    before = source.stat()
    live_descriptor = os.open(source, os.O_RDONLY)
    try:
        source.chmod(0o600)
        with source.open("r+b") as sink:
            sink.write(altered)
            sink.truncate()
            sink.flush()
            os.fsync(sink.fileno())
        os.lseek(live_descriptor, 0, os.SEEK_SET)
        consumed_by_vulnerable_live_fd = os.read(live_descriptor, len(altered))
        assert consumed_by_vulnerable_live_fd == altered

        with source.open("r+b") as sink:
            sink.write(original)
            sink.truncate()
            sink.flush()
            os.fsync(sink.fileno())
        source.chmod(stat.S_IMODE(before.st_mode))
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    finally:
        os.close(live_descriptor)

    after = source.stat()
    assert source.read_bytes() == original
    assert sft._inode_identity(after) == sft._inode_identity(before)
    assert frozen.payload == original
    assert frozen.verify_bytes() == admitted_sha
    _, _, _, frozen_stats = sft.build_packed(
        [frozen.open_bytes()],
        _Tokenizer(),
        4,
        ["question"],
        ["response"],
        999,
        return_stats=True,
    )
    _, _, _, original_stats = sft.build_packed(
        [sft.io.BytesIO(original)],
        _Tokenizer(),
        4,
        ["question"],
        ["response"],
        999,
        return_stats=True,
    )
    assert frozen_stats["packing_sha256"] == original_stats["packing_sha256"]


def test_freeze_file_bytes_rejects_altered_bytes_even_if_metadata_is_restorable(
    tmp_path,
):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    source = snapshot / "input.bin"
    source.write_bytes(b"admitted")
    source.chmod(0o444)
    expected = sft.sha256_file(source)
    before = source.stat()
    source.chmod(0o600)
    source.write_bytes(b"mutated!")
    source.chmod(stat.S_IMODE(before.st_mode))
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(ValueError, match="hash mismatch"):
        sft.freeze_file_bytes(source, expected, snapshot)


def test_scientific_sources_use_the_same_bootstrap_immutable_bytes(tmp_path):
    snapshot = tmp_path / "snapshot"
    expected = {}
    bootstrap_bytes = {}
    repo_root = Path(__file__).resolve().parent.parent
    for relative in sft.CANONICAL_SOURCE_PATHS:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repo_root / relative).read_bytes())
        destination.chmod(0o444)
        expected[relative] = sft.sha256_file(destination)
        bootstrap_bytes[relative] = destination.read_bytes()

    sources = sft.snapshot_scientific_sources(expected, bootstrap_bytes)
    assert set(sources) == set(sft.CANONICAL_SOURCE_PATHS)
    for relative, frozen in sources.items():
        assert frozen.payload is bootstrap_bytes[relative]
        assert frozen.verify_bytes() == expected[relative]
        assert frozen.binding["immutable_byte_snapshot"] is True
        assert frozen.binding["live_descriptor_retained"] is False

    replacement = snapshot / "replacement.py"
    replacement.write_text("raise RuntimeError('substituted')\n")
    replacement.chmod(0o444)
    os.replace(replacement, snapshot / "train" / "model.py")
    assert sources["train/model.py"].payload == bootstrap_bytes["train/model.py"]
    assert sources["train/model.py"].verify_bytes() == expected["train/model.py"]


def test_scientific_source_bootstrap_rejects_missing_or_mutable_payloads(tmp_path):
    del tmp_path
    expected = {relative: "a" * 64 for relative in sft.CANONICAL_SOURCE_PATHS}
    with pytest.raises(ValueError, match="source set mismatch"):
        sft.snapshot_scientific_sources(expected, {})

    payloads = {relative: b"source" for relative in sft.CANONICAL_SOURCE_PATHS}
    payloads["train/model.py"] = bytearray(b"source")
    expected = {
        relative: sft._sha256_bytes(bytes(payload))
        for relative, payload in payloads.items()
    }
    with pytest.raises(ValueError, match="not immutable bytes"):
        sft.snapshot_scientific_sources(expected, payloads)


def test_checkpoint_publication_is_fsynced_read_only_and_no_replace(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    artifact = sft.publish_torch_checkpoint(
        checkpoint, {"model": {"weight": sft.torch.arange(4)}}
    )
    try:
        assert stat.S_IMODE(checkpoint.stat().st_mode) == 0o444
        loaded = sft.torch.load(checkpoint, map_location="cpu")
        assert loaded["model"]["weight"].tolist() == [0, 1, 2, 3]
        original = checkpoint.read_bytes()
        with pytest.raises(FileExistsError):
            sft.publish_torch_checkpoint(checkpoint, {"forged": True})
        assert checkpoint.read_bytes() == original
        artifact.verify_path_identity()
    finally:
        artifact.close()


def _unseal_for_tmp_cleanup(directory):
    directory.chmod(0o700)
    for child in directory.iterdir():
        child.chmod(0o600)


def test_output_seal_is_closed_world_and_read_only(tmp_path):
    out = tmp_path / "out"
    out.mkdir(mode=0o700)
    artifacts = [
        sft.publish_canonical_json(out / "exact_budget_preflight.json", {"phase": 1}),
        sft.publish_torch_checkpoint(out / "sft_ep1.pt", {"model": {}}),
        sft.publish_canonical_json(out / "exact_budget_final.json", {"phase": 2}),
    ]
    try:
        sft.seal_output_directory(out, artifacts)
        assert stat.S_IMODE(out.stat().st_mode) == 0o555
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in out.iterdir())
        for artifact in artifacts:
            artifact.verify_path_identity()
    finally:
        _unseal_for_tmp_cleanup(out)
        for artifact in artifacts:
            artifact.close()


def test_output_seal_rejects_path_replacement_before_sealing(tmp_path):
    out = tmp_path / "out"
    out.mkdir(mode=0o700)
    artifact = sft.publish_canonical_json(
        out / "exact_budget_final.json", {"real": True}
    )
    forged = out / "forged"
    forged.write_bytes(b'{"forged":true}\n')
    forged.chmod(0o444)
    os.replace(forged, artifact.path)
    try:
        with pytest.raises(RuntimeError, match="published path was substituted"):
            sft.seal_output_directory(out, [artifact])
        assert stat.S_IMODE(out.stat().st_mode) == 0o700
    finally:
        artifact.close()


def test_output_seal_rejects_parent_replacement_even_with_same_file_inode(tmp_path):
    out = tmp_path / "out"
    displaced = tmp_path / "displaced"
    out.mkdir(mode=0o700)
    artifact = sft.publish_canonical_json(
        out / "exact_budget_final.json", {"real": True}
    )
    out.rename(displaced)
    out.mkdir(mode=0o700)
    os.link(displaced / artifact.path.name, out / artifact.path.name)
    try:
        with pytest.raises(RuntimeError, match="do not bind the output directory"):
            sft.seal_output_directory(out, [artifact])
    finally:
        artifact.close()


def test_exact_file_binding_rejects_wrong_hash_writable_or_external_input(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    inside = snapshot / "inside.bin"
    inside.write_bytes(b"bound bytes")
    expected = sft.sha256_file(inside)

    inside.chmod(0o444)
    with pytest.raises(ValueError, match="hash mismatch"):
        sft.exact_file_binding(inside, "0" * 64, snapshot)
    inside.chmod(0o644)
    with pytest.raises(ValueError, match="remains writable"):
        sft.exact_file_binding(inside, expected, snapshot)

    inside.chmod(0o444)
    binding = sft.exact_file_binding(inside, expected, snapshot)
    assert binding["path"] == "inside.bin"
    assert binding["sha256"] == expected
    assert binding["immutable_byte_snapshot"] is True
    assert binding["live_descriptor_retained"] is False

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    outside.chmod(0o444)
    with pytest.raises(ValueError, match="escapes private snapshot"):
        sft.exact_file_binding(outside, sft.sha256_file(outside), snapshot)


def test_factorial_sbatch_has_admission_source_and_attempt_custody(tmp_path):
    script_path = Path(__file__).parent / "jobs" / "sft_factorial.sbatch"
    script = script_path.read_text()
    for required in (
        "ADMISSION_SHA256",
        "REVIEWED_SOURCE_MANIFEST_SHA256",
        "REVIEWED_COMMIT",
        "DATA_SHA256",
        "INIT_SHA256",
        "TOKENIZER_SHA256",
        "GENERATOR_PY_SHA256",
        "RECURRENT_PY_SHA256",
        "AUDITOR_PY_SHA256",
        "PROTOCOL_PY_SHA256",
        "SFT_PY_SHA256",
        "MODEL_PY_SHA256",
        "MUON_PY_SHA256",
        "SFT_ENCODING_PY_SHA256",
        "SFT_TEST_PY_SHA256",
        "SBATCH_SHA256",
    ):
        assert required in script
    assert "sft_factorial_snapshot_${JOB_ID}_attempt_${RESTART_COUNT}" in script
    assert "attempt_${RESTART_COUNT}" in script
    assert "--canonical-exact-budget" in script
    assert "--production-admission" in script
    assert "--expected-production-admission-sha256" in script
    assert "--reviewed-source-manifest" in script
    assert "--expected-reviewed-source-manifest-sha256" in script
    assert "--expected-reviewed-commit" in script
    assert "--arm" not in script
    assert "--exact-updates 1560" in script
    assert "--batch-size 16" in script
    assert "--pack-len 2048" in script
    assert script.count("--expected-source-sha256") == len(sft.CANONICAL_SOURCE_PATHS)
    for relative in sft.CANONICAL_SOURCE_PATHS:
        assert f'--expected-source-sha256 "{relative}=' in script
    assert "#SBATCH --no-requeue" in script
    assert "#SBATCH --open-mode=append" in script
    assert "#SBATCH --open-mode=truncate" not in script
    assert "SLURM_RESTART_COUNT" in script
    assert "scontrol show job -o" in script
    assert "Requeue=0" in script
    assert 'git -C "$BASE" diff --quiet' in script
    assert "reviewed source differs from clean commit" in script
    assert "capture_exact_bytes" in script
    assert "types.MappingProxyType" in script
    assert "_CANONICAL_BOOTSTRAP_SOURCE_BYTES" in script
    assert "compile(frozen_sources[relative]" in script
    assert 'compile(frozen_sources["train/sft.py"]' in script
    assert "payload_base64" in script
    assert "not provide owner-proof immutability" in script
    assert '"$PY" sft.py' not in script
    assert "for att in" not in script
    assert "--compile" not in script

    attempt_root = tmp_path / "attempts"
    restarted = subprocess.run(
        ["bash", str(script_path)],
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "FACTORIAL_ATTEMPT_LOG_ROOT": str(attempt_root),
            "SHOHIN_BASE": str(tmp_path / "base"),
            "SLURM_JOB_ID": "12345",
            "SLURM_RESTART_COUNT": "1",
        },
        capture_output=True,
        check=False,
        text=True,
    )
    assert restarted.returncode == 10
    assert "refusing Slurm restart count" in restarted.stdout + restarted.stderr
    attempt = attempt_root / "job_12345" / "attempt_1"
    assert "refusing Slurm restart count" in (attempt / "wrapper.log").read_text()
    exit_evidence = (attempt / "attempt_exit.tsv").read_text()
    assert "status\trefused_requeue" in exit_evidence
    assert "exit_code\t10" in exit_evidence

    prior_log = (attempt / "wrapper.log").read_bytes()
    duplicate = subprocess.run(
        ["bash", str(script_path)],
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "FACTORIAL_ATTEMPT_LOG_ROOT": str(attempt_root),
            "SHOHIN_BASE": str(tmp_path / "base"),
            "SLURM_JOB_ID": "12345",
            "SLURM_RESTART_COUNT": "1",
        },
        capture_output=True,
        check=False,
        text=True,
    )
    assert duplicate.returncode == 11
    assert (attempt / "wrapper.log").read_bytes() == prior_log


def test_exact_budget_trust_boundary_disclaims_remote_attestation():
    boundary = sft.EXACT_BUDGET_TRUST_BOUNDARY
    assert boundary["remote_attestation"] is False
    assert (
        "immutable private-memory snapshots"
        in boundary["input_same_uid_path_replacement"]
    )
    assert "same-inode mutation" in boundary["input_same_uid_path_replacement"]
    assert "Modes 0444 and 0555" in boundary["output_same_uid_boundary"]
    assert "owning UID can chmod" in boundary["output_same_uid_boundary"]
    assert "not provide owner-proof immutability" in boundary["scope"]
    assert "not remote attestation" in boundary["scope"]
