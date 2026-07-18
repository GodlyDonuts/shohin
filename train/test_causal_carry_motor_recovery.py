from __future__ import annotations

import copy
import collections
import hashlib
import inspect
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import causal_carry_motor as upstream
import causal_carry_motor_recovery as recovery


NEW_COMMIT = "b" * 40
EXPECTED_SHARD_RECEIPTS = (
    "4affa12434513ebe9587464ff38656abaaf7e47904d9db6ced252c3adea52a96",
    "4731c1644703e26c1978ca1ec1ba80af7c173c5d9676ae68fbd04368f3b54c2c",
    "e81639e68a838bfa6695be92f7c1333d100b2317c48fb2cf0d995f22a6e50a43",
    "ae86ec1b70dca21d67849fc4be17ffec682472851735c3b9523292836a74e70f",
    "ce5a151f89e20e774c7d37afc446ea026ec14a587c70fa614414f060f10a2144",
    "f02d8221bf3a393566c279e27bf888fcbd1ef9ea17bdd33262472c898950ea83",
    "009b83f0c2a70362654e3e3e4cad27d30f79f93f3bdd32d6ce3064695dd2b9db",
    "8214d356288c56a116a3de753a8948a35f731d52c520fa906f4e31c1b0f14fb4",
)


def _raw_board(rows):
    return {
        "attempts": 100,
        "forbidden_prompt_count": 7,
        "position_counts": {"add|4|0|core|0|1|0": 1},
        "prefix_order_sha256": "1" * 64,
        "prompt_length_histogram": {97: 3, 99: 4, 103: 5, 105: 6},
        "quota": 1,
        "rows": len(rows),
        "seed": 20260717,
        "strata": {"add|4|0|core|0|1": 1},
        "token_length_histogram": {114: 3, 116: 4, 120: 5, 122: 6},
    }


def _normalization_fixture(monkeypatch):
    rows = [{"prefix_sha256": "a" * 64, "target": 1, "target_id": 29}]
    raw = _raw_board(rows)
    raw_with_hash = copy.deepcopy(raw)
    raw_with_hash["rows_sha256"] = upstream.stable_json_sha256(rows)
    sealed = recovery.canonical_json_document(raw_with_hash)
    monkeypatch.setattr(
        recovery, "UPSTREAM_BOARD_ROWS_SHA256", upstream.stable_json_sha256(rows)
    )
    monkeypatch.setattr(
        recovery, "UPSTREAM_CANONICAL_BOARD_SHA256", recovery.stable_json_sha256(sealed)
    )
    return rows, raw, sealed


def _minimal_recovery_plan(monkeypatch, tmp_path):
    upstream_root = tmp_path / "upstream"
    recovery_parent = tmp_path / "recoveries"
    monkeypatch.setattr(recovery, "UPSTREAM_ROOT", upstream_root)
    monkeypatch.setattr(recovery, "RECOVERY_PARENT", recovery_parent)
    root = recovery.recovery_root(NEW_COMMIT)
    document = {
        "audit": recovery.RECOVERY_PLAN_AUDIT,
        "recovery": True,
        "recovery_plan_path": str(root / "recovery_plan.json"),
        "recovery_executor_source_contract": {
            "schema": recovery.RECOVERY_EXECUTOR_SOURCE_SCHEMA,
            "git_commit": NEW_COMMIT,
            "sources": {},
            "manifest_sha256": "1" * 64,
        },
        "executor_runtime_contract": {
            "schema": recovery.RECOVERY_EXECUTOR_RUNTIME_SCHEMA,
            "source_root": "/reviewed/recovery",
        },
        "hostile_review_binding": {
            "path": "/review.json",
            "sha256": "2" * 64,
            "document": {},
        },
        "upstream_protocol": {
            "source_contract": {"git_commit": recovery.UPSTREAM_SOURCE_COMMIT},
            "plan_binding": {"sha256": recovery.UPSTREAM_PLAN_SHA256},
            "shard_receipts": [],
        },
        "normalization_proof": {"mismatch_count": 2},
        "allowed_transformation": recovery.ALLOWED_TRANSFORMATION,
        "fit_contract": {
            "fit_budget": {
                "seed": upstream.FIT_SEED,
                "rank": upstream.RANK,
                "quota": upstream.FIT_QUOTA,
                "updates": upstream.CANONICAL_UPDATES,
                "batch_size": upstream.CANONICAL_BATCH,
                "lr": upstream.CANONICAL_LR,
                "weight_decay": upstream.CANONICAL_WEIGHT_DECAY,
            }
        },
        "output_contract": {
            "root": str(root),
            "fit_artifact": str(root / "fit" / "motor.pt"),
            "development_eval_artifact": str(
                root / "development_eval" / "evaluation.json"
            ),
            "confirmation_eval_artifact": str(
                root / "confirmation_eval" / "evaluation.json"
            ),
            "upstream_root_must_remain_untouched": str(upstream_root),
        },
        "deserialization_contract": recovery.DESERIALIZATION_CONTRACT,
        "claim_boundary": recovery.RECOVERY_PLAN_CLAIM_BOUNDARY,
    }
    return root, document


def _git(repo, *arguments):
    return subprocess.check_output(
        [str(recovery.PINNED_GIT), "-C", str(repo), *arguments], text=True
    ).strip()


def _source_contract_repo(monkeypatch, tmp_path, *, extra_paths=()):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    subprocess.run([str(recovery.PINNED_GIT), "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            str(recovery.PINNED_GIT),
            "-C",
            str(repo),
            "config",
            "user.email",
            "test@example.invalid",
        ],
        check=True,
    )
    subprocess.run(
        [
            str(recovery.PINNED_GIT),
            "-C",
            str(repo),
            "config",
            "user.name",
            "Recovery Test",
        ],
        check=True,
    )
    (repo / "baseline.txt").write_text("sealed upstream\n")
    subprocess.run(
        [str(recovery.PINNED_GIT), "-C", str(repo), "add", "baseline.txt"],
        check=True,
    )
    subprocess.run(
        [str(recovery.PINNED_GIT), "-C", str(repo), "commit", "-qm", "upstream"],
        check=True,
    )
    upstream_commit = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(recovery, "UPSTREAM_SOURCE_COMMIT", upstream_commit)
    for index, name in enumerate((*recovery.RECOVERY_SOURCE_PATHS, *extra_paths)):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"recovery source {index}\n")
    subprocess.run(
        [str(recovery.PINNED_GIT), "-C", str(repo), "add", "--all"], check=True
    )
    subprocess.run(
        [str(recovery.PINNED_GIT), "-C", str(repo), "commit", "-qm", "recovery"],
        check=True,
    )
    recovery_commit = _git(repo, "rev-parse", "HEAD")
    sources = {
        name: recovery.sha256_file(repo / name)
        for name in recovery.RECOVERY_SOURCE_PATHS
    }
    monkeypatch.setattr(
        recovery,
        "validate_loaded_module_paths",
        lambda _root: {
            "recovery": str(repo / "train" / "causal_carry_motor_recovery.py"),
            "upstream": str(repo / "train" / "causal_carry_motor.py"),
            "model": str(repo / "train" / "model.py"),
        },
    )
    return repo, upstream_commit, recovery_commit, recovery.stable_json_sha256(sources)


def test_exact_upstream_receipts_are_complete_and_cross_file_bound():
    root = Path(__file__).resolve().parents[1]
    prereg = (root / "R12_CAUSAL_CARRY_MOTOR_RECOVERY_PREREG.md").read_text()
    wrapper = (root / "train/jobs/causal_carry_motor_recovery.sbatch").read_text()
    assert recovery.UPSTREAM_SOURCE_COMMIT == (
        "a0c258e6709766c643cf127a429a7d6ef4a4211b"
    )
    assert recovery.UPSTREAM_PLAN_SHA256 == (
        "1b845d47f6875df571169efb5adb0716dfbc5d266a2499e4a92451351a262b6d"
    )
    assert recovery.UPSTREAM_SHARD_SHA256 == EXPECTED_SHARD_RECEIPTS
    assert len(set(recovery.UPSTREAM_SHARD_SHA256)) == 8
    for receipt in (
        recovery.UPSTREAM_SOURCE_COMMIT,
        recovery.UPSTREAM_SOURCE_MANIFEST_SHA256,
        recovery.UPSTREAM_PLAN_SHA256,
        recovery.UPSTREAM_CONFIRMATION_COMMITMENT_SHA256,
        recovery.UPSTREAM_BOARD_ROWS_SHA256,
        recovery.UPSTREAM_CANONICAL_BOARD_SHA256,
        recovery.NORMALIZATION_MISMATCH_LEDGER_SHA256,
        *EXPECTED_SHARD_RECEIPTS,
    ):
        assert prereg.count(receipt) >= 1
    assert recovery.UPSTREAM_SOURCE_COMMIT in wrapper
    assert recovery.UPSTREAM_PLAN_SHA256 in wrapper


def test_normalization_proof_has_exactly_two_key_type_differences(monkeypatch):
    rows, raw, sealed = _normalization_fixture(monkeypatch)
    proof, normalized = recovery.build_normalization_proof(raw, sealed, rows)
    assert proof["mismatch_count"] == 2
    assert proof["mismatches"] == list(recovery.EXPECTED_NORMALIZATION_MISMATCHES)
    assert proof["canonical_board_equal"] is True
    assert recovery.type_strict_equal(normalized, sealed)
    assert proof["allowed_transformation"] == recovery.ALLOWED_TRANSFORMATION


def test_normalization_rejects_non_histogram_difference(monkeypatch):
    rows, raw, sealed = _normalization_fixture(monkeypatch)
    sealed["attempts"] += 1
    monkeypatch.setattr(
        recovery, "UPSTREAM_CANONICAL_BOARD_SHA256", recovery.stable_json_sha256(sealed)
    )
    with pytest.raises(ValueError, match="non-histogram board difference"):
        recovery.build_normalization_proof(raw, sealed, rows)


def test_normalization_rejects_histogram_value_or_extra_key_change(monkeypatch):
    rows, raw, sealed = _normalization_fixture(monkeypatch)
    sealed["prompt_length_histogram"]["97"] += 1
    monkeypatch.setattr(
        recovery, "UPSTREAM_CANONICAL_BOARD_SHA256", recovery.stable_json_sha256(sealed)
    )
    with pytest.raises(ValueError, match="differs beyond JSON key typing"):
        recovery.build_normalization_proof(raw, sealed, rows)

    rows, raw, sealed = _normalization_fixture(monkeypatch)
    raw["prompt_length_histogram"]["97"] = raw["prompt_length_histogram"][97]
    with pytest.raises(ValueError, match="generated keys are not all integers"):
        recovery.build_normalization_proof(raw, sealed, rows)


def test_strict_json_and_equality_reject_python_type_aliases():
    assert not recovery.type_strict_equal(True, 1)
    assert not recovery.type_strict_equal(1, 1.0)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        recovery.load_exact_json('{"x":1,"x":2}', "hostile")
    with pytest.raises(ValueError, match="non-finite"):
        recovery.canonical_json_payload({"x": float("nan")})


def test_bound_file_rejects_alias_and_receipt_substitution(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"sealed")
    digest = hashlib.sha256(b"sealed").hexdigest()
    alias = tmp_path / "alias.bin"
    alias.symlink_to(artifact)
    with pytest.raises(ValueError, match="aliases or differs"):
        recovery.BoundFile(str(alias), artifact, digest, "artifact")
    with pytest.raises(ValueError, match="hash mismatch"):
        recovery.BoundFile(str(artifact), artifact, "0" * 64, "artifact")
    bound = recovery.BoundFile(str(artifact), artifact, digest, "artifact")
    try:
        assert bound.bytes() == b"sealed"
    finally:
        bound.close()


def _upstream_custody_fixture(monkeypatch, tmp_path):
    root = tmp_path / "canonical"
    root.mkdir()
    plan_path = root / "plan.json"
    plan_path.write_bytes(b"sealed plan\n")
    os.chmod(plan_path, 0o444)
    for name in ("fit", "development_eval", "confirmation_eval"):
        (root / name).mkdir(mode=0o700)
    shard_hashes = []
    for index in range(upstream.CANONICAL_FEATURE_SHARDS):
        directory = root / f"shard_{index:02d}"
        directory.mkdir()
        artifact = directory / "features.pt"
        artifact.write_bytes(f"sealed shard {index}\n".encode())
        shard_hashes.append(recovery.sha256_file(artifact))
        os.chmod(artifact, 0o444)
        os.chmod(directory, 0o555)
    commitment_dir = tmp_path / "commitment"
    commitment_dir.mkdir()
    commitment_path = commitment_dir / "commitment.json"
    commitment_path.write_bytes(b"sealed commitment\n")
    os.chmod(commitment_path, 0o444)
    os.chmod(commitment_dir, 0o555)
    os.chmod(root, 0o555)
    monkeypatch.setattr(recovery, "UPSTREAM_ROOT", root)
    monkeypatch.setattr(recovery, "UPSTREAM_PLAN_PATH", plan_path)
    monkeypatch.setattr(recovery, "UPSTREAM_CONFIRMATION_PATH", commitment_path)
    monkeypatch.setattr(
        recovery, "UPSTREAM_PLAN_SHA256", recovery.sha256_file(plan_path)
    )
    monkeypatch.setattr(
        recovery,
        "UPSTREAM_CONFIRMATION_COMMITMENT_SHA256",
        recovery.sha256_file(commitment_path),
    )
    monkeypatch.setattr(recovery, "UPSTREAM_SHARD_SHA256", tuple(shard_hashes))
    return root, commitment_path


def test_upstream_custody_snapshot_covers_every_directory_file_and_mode(
    monkeypatch, tmp_path
):
    root, commitment_path = _upstream_custody_fixture(monkeypatch, tmp_path)
    snapshot = recovery.capture_upstream_custody_snapshot()
    entries = {entry["path"]: entry for entry in snapshot["entries"]}
    assert snapshot["schema"] == recovery.UPSTREAM_CUSTODY_SCHEMA
    assert len(entries) == 23
    assert entries[str(root)]["identity"]["mode"] == 0o555
    for name in ("fit", "development_eval", "confirmation_eval"):
        entry = entries[str(root / name)]
        assert entry["identity"]["mode"] == 0o700
        assert entry["children"] == []
    for index in range(upstream.CANONICAL_FEATURE_SHARDS):
        directory = root / f"shard_{index:02d}"
        assert entries[str(directory)]["children"] == ["features.pt"]
        assert entries[str(directory / "features.pt")]["identity"]["mode"] == 0o444
    assert entries[str(commitment_path)]["sha256"] == recovery.sha256_file(
        commitment_path
    )
    assert recovery.assert_upstream_custody_unchanged(snapshot, "during test")


def test_upstream_custody_revalidation_rejects_empty_dir_shard_and_inode_attacks(
    monkeypatch, tmp_path
):
    root, _commitment = _upstream_custody_fixture(monkeypatch, tmp_path)
    snapshot = recovery.capture_upstream_custody_snapshot()
    injected = root / "fit" / "foreign.pt"
    injected.write_bytes(b"attack")
    with pytest.raises(ValueError, match="not closed-world"):
        recovery.assert_upstream_custody_unchanged(snapshot, "after publication")
    injected.unlink()
    snapshot = recovery.capture_upstream_custody_snapshot()

    shard_dir = root / "shard_00"
    os.chmod(shard_dir, 0o755)
    with pytest.raises(ValueError, match="mode mismatch"):
        recovery.assert_upstream_custody_unchanged(snapshot, "after publication")
    os.chmod(shard_dir, 0o555)
    snapshot = recovery.capture_upstream_custody_snapshot()

    artifact = shard_dir / "features.pt"
    original = artifact.read_bytes()
    os.chmod(shard_dir, 0o755)
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(original)
    os.chmod(replacement, 0o444)
    os.replace(replacement, artifact)
    os.chmod(shard_dir, 0o555)
    with pytest.raises(RuntimeError, match="custody changed"):
        recovery.assert_upstream_custody_unchanged(snapshot, "after publication")


def test_fit_revalidates_full_custody_immediately_around_publish_and_seal():
    source = inspect.getsource(recovery._fit)
    before_publish = source.index("immediately before recovery artifact publication")
    publish = source.index("publish_recovery_torch(out, bundle)")
    after_publish = source.index("immediately after recovery artifact publication")
    before_seal = source.index("immediately before recovery artifact sealing")
    seal = source.index("seal_recovery_fit(out)", before_seal)
    after_seal = source.index("immediately after recovery artifact sealing")
    assert before_publish < publish < after_publish
    assert before_seal < seal < after_seal


def test_safe_torch_load_is_weights_only_allowlisted_and_rejects_env(
    monkeypatch, tmp_path
):
    artifact = tmp_path / "payload.pt"
    torch.save({"runtime": torch.torch_version.TorchVersion("2.6.0+cu124")}, artifact)
    digest = recovery.sha256_file(artifact)
    bound = recovery.BoundFile(str(artifact), artifact, digest, "torch payload")
    try:
        loaded = recovery.safe_torch_load(bound)
        assert str(loaded["runtime"]) == "2.6.0+cu124"
        monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        with pytest.raises(RuntimeError, match="override is forbidden"):
            recovery.safe_torch_load(bound)
    finally:
        bound.close()


def test_recovery_runtime_must_equal_frozen_upstream_runtime(monkeypatch):
    monkeypatch.setattr(upstream, "require_canonical_cuda_runtime", lambda: "cuda")
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda _index: "NVIDIA H100 PCIe"
    )
    expected = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": "NVIDIA H100 PCIe",
    }
    assert (
        recovery.require_recovery_cuda_runtime(
            {"runtime_contract": {"artifact_runtime": expected}}
        )
        == "cuda"
    )
    hostile = copy.deepcopy(expected)
    hostile["torch"] = "different"
    with pytest.raises(RuntimeError, match="differs from frozen"):
        recovery.require_recovery_cuda_runtime(
            {"runtime_contract": {"artifact_runtime": hostile}}
        )


def test_board_reconstruction_freezes_row_order_control_schedule_and_init(monkeypatch):
    rows, raw, sealed = _normalization_fixture(monkeypatch)

    class FrozenText:
        def __init__(self, value):
            self.value = value
            self.verified = False

        def text(self):
            return self.value

        def verify(self):
            self.verified = True

    tokenizer_bound = FrozenText("tokenizer")
    episodes_bound = FrozenText("episodes")
    frozen = {"tokenizer": tokenizer_bound, "episodes": episodes_bound}
    tokenizer = object()
    control_labels = torch.tensor([1])
    control = {"permutation_sha256": "4" * 64}
    schedule_sha256 = "5" * 64
    initial_sha256 = "6" * 64
    plan = {
        "board": sealed,
        "extraction_order_sha256": upstream.stable_json_sha256(
            [row["prefix_sha256"] for row in rows]
        ),
        "fit_budget": {
            "control": control,
            "batch_size": 512,
            "updates": 2000,
            "seed": 20260717,
            "schedule_sha256": schedule_sha256,
            "initial_state_sha256": initial_sha256,
        },
        "d_model": 8,
    }
    monkeypatch.setattr(
        recovery.Tokenizer,
        "from_str",
        lambda value: tokenizer if value == "tokenizer" else None,
    )
    monkeypatch.setattr(
        upstream,
        "generate_fit_rows",
        lambda observed_tokenizer, observed_episodes, *_: (
            (
                rows,
                raw,
            )
            if observed_tokenizer is tokenizer and observed_episodes == "episodes"
            else None
        ),
    )
    monkeypatch.setattr(
        upstream,
        "permuted_control_labels",
        lambda _rows: (control_labels, control),
    )
    monkeypatch.setattr(
        upstream,
        "_batch_schedule",
        lambda *_: (None, schedule_sha256),
    )
    monkeypatch.setattr(
        upstream,
        "initial_motor_state",
        lambda _d_model: ({"state": "frozen"}, initial_sha256),
    )

    result = recovery.reconstruct_board(plan, frozen)
    assert result["control_labels"] is control_labels
    assert all(bound.verified for bound in frozen.values())

    hostile = copy.deepcopy(plan)
    hostile["extraction_order_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="row order changed"):
        recovery.reconstruct_board(hostile, frozen)

    hostile = copy.deepcopy(plan)
    hostile["fit_budget"]["control"] = {"permutation_sha256": "0" * 64}
    with pytest.raises(ValueError, match="control changed"):
        recovery.reconstruct_board(hostile, frozen)

    hostile = copy.deepcopy(plan)
    hostile["fit_budget"]["schedule_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="schedule changed"):
        recovery.reconstruct_board(hostile, frozen)

    hostile = copy.deepcopy(plan)
    hostile["fit_budget"]["initial_state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="initial motor state changed"):
        recovery.reconstruct_board(hostile, frozen)


def test_executor_source_contract_is_distinct_clean_and_commit_bound(
    monkeypatch, tmp_path
):
    repo, upstream_commit, commit, manifest = _source_contract_repo(
        monkeypatch, tmp_path
    )
    contract = recovery.build_recovery_executor_source_contract(
        commit, manifest, repo_root=repo
    )
    assert contract["git_commit"] == commit
    assert contract["parent_commit"] == upstream_commit
    assert contract["name_status_diff"] == list(recovery.RECOVERY_NAME_STATUS_DIFF)
    assert contract["manifest_sha256"] == manifest
    (repo / recovery.RECOVERY_SOURCE_PATHS[1]).write_text("dirty attack\n")
    with pytest.raises(ValueError, match="not clean"):
        recovery.build_recovery_executor_source_contract(
            commit, manifest, repo_root=repo
        )
    with pytest.raises(ValueError, match="may not alias"):
        recovery.build_recovery_executor_source_contract(
            upstream_commit,
            manifest,
            repo_root=repo,
        )


def test_executor_source_contract_rejects_extra_file_and_grandchild(
    monkeypatch, tmp_path
):
    extra_repo, _upstream, extra_commit, manifest = _source_contract_repo(
        monkeypatch, tmp_path / "extra", extra_paths=("train/model.py",)
    )
    with pytest.raises(ValueError, match="exactly four added"):
        recovery.build_recovery_executor_source_contract(
            extra_commit,
            manifest,
            repo_root=extra_repo,
        )

    child_repo, _upstream, recovery_commit, manifest = _source_contract_repo(
        monkeypatch, tmp_path / "grandchild"
    )
    subprocess.run(
        [
            str(recovery.PINNED_GIT),
            "-C",
            str(child_repo),
            "commit",
            "--allow-empty",
            "-qm",
            "unreviewed grandchild",
        ],
        check=True,
    )
    grandchild = _git(child_repo, "rev-parse", "HEAD")
    assert grandchild != recovery_commit
    with pytest.raises(ValueError, match="sole direct parent"):
        recovery.build_recovery_executor_source_contract(
            grandchild,
            manifest,
            repo_root=child_repo,
        )


def test_executor_source_contract_rejects_ignored_shadow_file(monkeypatch, tmp_path):
    repo, _upstream, commit, manifest = _source_contract_repo(monkeypatch, tmp_path)
    exclude = repo / ".git" / "info" / "exclude"
    with exclude.open("a") as sink:
        sink.write("train/sitecustomize.py\n")
    shadow = repo / "train" / "sitecustomize.py"
    shadow.write_text("raise RuntimeError('shadow executed')\n")
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""
    with pytest.raises(ValueError, match="not closed-world"):
        recovery.build_recovery_executor_source_contract(
            commit,
            manifest,
            repo_root=repo,
        )


def test_executor_source_contract_rejects_hard_link_alias(monkeypatch, tmp_path):
    repo, _upstream, commit, manifest = _source_contract_repo(monkeypatch, tmp_path)
    source = repo / recovery.RECOVERY_SOURCE_PATHS[0]
    alias = tmp_path / "source-alias.md"
    os.link(source, alias)
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""
    with pytest.raises(ValueError, match="source file identity mismatch"):
        recovery.build_recovery_executor_source_contract(
            commit,
            manifest,
            repo_root=repo,
        )


def test_loaded_module_shadow_is_rejected(monkeypatch, tmp_path):
    repo = Path(__file__).resolve().parents[1]
    recovery.validate_loaded_module_paths(repo)
    shadow = tmp_path / "model.py"
    shadow.write_text("# hostile shadow\n")
    monkeypatch.setattr(recovery.model_module, "__file__", str(shadow))
    with pytest.raises(ValueError, match="shadowed or aliased"):
        recovery.validate_loaded_module_paths(repo)


def test_runtime_contract_uses_fixed_launcher_and_rejects_environment_aliases():
    assert (
        "launcher_path"
        not in inspect.signature(recovery.capture_executor_runtime_contract).parameters
    )
    repo = Path(__file__).resolve().parents[1]
    script = """
import json
import sys
from pathlib import Path
import causal_carry_motor_recovery as recovery
recovery.PINNED_PYTHON_LAUNCHER = Path(sys.executable)
print(json.dumps(recovery.capture_executor_runtime_contract(), sort_keys=True))
"""
    environment = os.environ.copy()
    for name in recovery.FORBIDDEN_EXECUTOR_ENVIRONMENT:
        environment.pop(name, None)
    environment.update(recovery.EXECUTOR_ENVIRONMENT)
    environment["PYTHONPATH"] = str(repo / "train")
    clean = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    contract = json.loads(clean.stdout)
    assert contract["schema"] == recovery.RECOVERY_EXECUTOR_RUNTIME_SCHEMA
    assert contract["source_root"] == str(repo)
    assert contract["python"]["flags"]["no_user_site"] == 1
    assert contract["packages"]["torch"]["entrypoint"]["sha256"]

    hostile = dict(environment)
    hostile["OMP_NUM_THREADS"] = "8"
    rejected = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env=hostile,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "environment mismatch: OMP_NUM_THREADS" in rejected.stderr


def test_recovery_plan_rejects_budget_change_old_root_and_extra_transform(
    monkeypatch, tmp_path
):
    _root, expected = _minimal_recovery_plan(monkeypatch, tmp_path)
    recovery.validate_recovery_plan_document(expected, expected, NEW_COMMIT)

    changed = copy.deepcopy(expected)
    changed["fit_contract"]["fit_budget"]["updates"] += 1
    with pytest.raises(ValueError, match="differs from independently reconstructed"):
        recovery.validate_recovery_plan_document(changed, expected, NEW_COMMIT)

    old_output = copy.deepcopy(expected)
    old_output["output_contract"]["fit_artifact"] = str(
        Path(expected["output_contract"]["upstream_root_must_remain_untouched"])
        / "fit"
        / "motor.pt"
    )
    with pytest.raises(ValueError, match="aliases the old canonical root"):
        recovery.validate_recovery_plan_document(old_output, old_output, NEW_COMMIT)

    expanded = copy.deepcopy(expected)
    expanded["allowed_transformation"] = {
        **expanded["allowed_transformation"],
        "permitted_additional_transformations": 1,
    }
    with pytest.raises(ValueError, match="extra transformations"):
        recovery.validate_recovery_plan_document(expanded, expanded, NEW_COMMIT)

    scalar_alias = copy.deepcopy(expected)
    scalar_alias["allowed_transformation"]["permitted_semantic_changes"] = False
    with pytest.raises(ValueError, match="extra transformations"):
        recovery.validate_recovery_plan_document(scalar_alias, scalar_alias, NEW_COMMIT)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("seed", upstream.FIT_SEED + 1),
        ("rank", upstream.RANK + 1),
        ("quota", upstream.FIT_QUOTA + 1),
        ("updates", upstream.CANONICAL_UPDATES + 1),
        ("batch_size", upstream.CANONICAL_BATCH + 1),
        ("lr", upstream.CANONICAL_LR * 2),
        ("weight_decay", upstream.CANONICAL_WEIGHT_DECAY * 2),
    ),
)
def test_recovery_plan_rejects_every_frozen_fit_scalar(
    monkeypatch, tmp_path, field, replacement
):
    _root, expected = _minimal_recovery_plan(monkeypatch, tmp_path)
    hostile = copy.deepcopy(expected)
    hostile["fit_contract"]["fit_budget"][field] = replacement
    with pytest.raises(ValueError, match="differs from independently reconstructed"):
        recovery.validate_recovery_plan_document(hostile, expected, NEW_COMMIT)


def test_recovery_plan_publication_is_immutable_and_closed_world(monkeypatch, tmp_path):
    root, document = _minimal_recovery_plan(monkeypatch, tmp_path)
    recovery._publish_recovery_plan(root, document)
    recovery.validate_recovery_layout(root, fit_state="empty")
    assert (root / "recovery_plan.json").stat().st_mode & 0o777 == 0o444
    assert root.stat().st_mode & 0o777 == 0o555
    with pytest.raises(FileExistsError, match="must be a new"):
        recovery._publish_recovery_plan(root, document)


def test_descriptor_publication_is_no_replace_one_link_and_crash_recoverable(
    monkeypatch, tmp_path
):
    root, document = _minimal_recovery_plan(monkeypatch, tmp_path)
    recovery._publish_recovery_plan(root, document)
    assert recovery.recovery_fit_state(root) == "empty"
    recovery.validate_recovery_layout(root, fit_state="empty")

    artifact = root / "fit" / "motor.pt"
    payload = {"tensor": torch.tensor([1, 2, 3]), "audit": "v9-test"}
    digest = recovery.publish_recovery_torch(artifact, payload)
    assert digest == recovery.sha256_file(artifact)
    assert recovery.recovery_fit_state(root) == "recoverable"
    recovery.validate_recovery_layout(root, fit_state="recoverable")
    artifact_stat = os.lstat(artifact)
    assert stat.S_IMODE(artifact_stat.st_mode) == 0o444
    assert artifact_stat.st_nlink == 1
    assert [item.name for item in artifact.parent.iterdir()] == ["motor.pt"]
    assert not any("stage" in item.name for item in artifact.parent.iterdir())
    bound = recovery.BoundFile(
        str(artifact),
        artifact,
        digest,
        "recoverable publication",
        required_mode=0o444,
        required_parent_mode=0o700,
    )
    try:
        loaded = recovery.safe_torch_load(bound)
        assert torch.equal(loaded["tensor"], payload["tensor"])
    finally:
        bound.close()

    with pytest.raises(FileExistsError, match="not empty"):
        recovery.publish_recovery_torch(artifact, payload)

    # A crash immediately after publish leaves this exact recoverable state.
    recovery.seal_recovery_fit(artifact)
    assert recovery.recovery_fit_state(root) == "sealed"
    recovery.validate_recovery_layout(root, fit_state="sealed")


def test_interrupted_descriptor_publication_cleans_without_staging_or_two_links(
    monkeypatch, tmp_path
):
    root, document = _minimal_recovery_plan(monkeypatch, tmp_path)
    recovery._publish_recovery_plan(root, document)
    artifact = root / "fit" / "motor.pt"
    original_save = torch.save

    def crash_during_save(_value, sink):
        sink.write(b"partial archive")
        sink.flush()
        raise RuntimeError("injected publication crash")

    monkeypatch.setattr(torch, "save", crash_during_save)
    with pytest.raises(RuntimeError, match="injected publication crash"):
        recovery.publish_recovery_torch(artifact, {"audit": "v9-test"})
    assert recovery.recovery_fit_state(root) == "interrupted"
    artifact_stat = os.lstat(artifact)
    assert stat.S_IMODE(artifact_stat.st_mode) == 0o600
    assert artifact_stat.st_nlink == 1
    assert [item.name for item in artifact.parent.iterdir()] == ["motor.pt"]
    recovery.discard_interrupted_recovery_fit(artifact)
    assert recovery.recovery_fit_state(root) == "empty"
    assert list(artifact.parent.iterdir()) == []

    monkeypatch.setattr(torch, "save", original_save)
    recovery.publish_recovery_torch(artifact, {"audit": "v9-test"})
    assert recovery.recovery_fit_state(root) == "recoverable"


def test_recovery_fit_state_rejects_wrong_mode_extra_child_or_link(
    monkeypatch, tmp_path
):
    root, document = _minimal_recovery_plan(monkeypatch, tmp_path)
    recovery._publish_recovery_plan(root, document)
    artifact = root / "fit" / "motor.pt"
    artifact.write_bytes(b"mutable")
    os.chmod(artifact, 0o640)
    with pytest.raises(ValueError, match="artifact mode mismatch"):
        recovery.recovery_fit_state(root)

    os.chmod(artifact, 0o444)
    sibling = tmp_path / "linked.pt"
    os.link(artifact, sibling)
    with pytest.raises(ValueError, match="artifact identity mismatch"):
        recovery.recovery_fit_state(root)
    sibling.unlink()
    hostile = artifact.parent / "extra.pt"
    hostile.write_bytes(b"substitution")
    with pytest.raises(ValueError, match="not closed-world"):
        recovery.recovery_fit_state(root)


def test_recovery_plan_publication_rejects_symlink_parent(monkeypatch, tmp_path):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    monkeypatch.setattr(recovery, "RECOVERY_PARENT", alias_parent)
    root = recovery.recovery_root(NEW_COMMIT)
    with pytest.raises(ValueError, match="non-symlink"):
        recovery._publish_recovery_plan(root, {"audit": "hostile"})


def test_recovery_plan_publication_never_replaces_raced_target(monkeypatch, tmp_path):
    root, document = _minimal_recovery_plan(monkeypatch, tmp_path)
    root.mkdir(parents=True)
    marker = root / "foreign"
    marker.write_bytes(b"must-survive")
    original_exists = Path.exists

    def hidden_target(path):
        if path == root:
            return False
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", hidden_target)
    with pytest.raises(FileExistsError, match="raced or already exists"):
        recovery._publish_recovery_plan(root, document)
    assert marker.read_bytes() == b"must-survive"


def test_hostile_review_binds_exact_executor_and_normalization(monkeypatch, tmp_path):
    monkeypatch.setattr(recovery, "REVIEW_PARENT", tmp_path / "reviews")
    contract = {
        "schema": recovery.RECOVERY_EXECUTOR_SOURCE_SCHEMA,
        "git_commit": NEW_COMMIT,
        "sources": {"executor.py": "1" * 64},
        "manifest_sha256": "2" * 64,
    }
    runtime = {
        "schema": recovery.RECOVERY_EXECUTOR_RUNTIME_SCHEMA,
        "source_root": "/reviewed/recovery",
    }
    review = {
        "audit": recovery.RECOVERY_REVIEW_AUDIT,
        "decision": "GO",
        "recovery_executor_source_contract": contract,
        "executor_runtime_contract": runtime,
        "upstream_plan_sha256": recovery.UPSTREAM_PLAN_SHA256,
        "normalization_contract": recovery.normalization_contract(),
        "allowed_transformation": recovery.ALLOWED_TRANSFORMATION,
        "claim_boundary": recovery.REVIEW_CLAIM_BOUNDARY,
    }
    path = recovery.recovery_review_path(NEW_COMMIT)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(review, sort_keys=True) + "\n")
    os.chmod(path, 0o444)
    os.chmod(path.parent, 0o555)
    digest = recovery.sha256_file(path)
    bound, loaded = recovery.load_hostile_review(
        NEW_COMMIT, contract, runtime, str(path), digest
    )
    try:
        assert loaded["decision"] == "GO"
    finally:
        bound.close()

    os.chmod(path.parent, 0o755)
    os.chmod(path, 0o644)
    review["decision"] = "NO-GO"
    path.write_text(json.dumps(review, sort_keys=True) + "\n")
    os.chmod(path, 0o444)
    os.chmod(path.parent, 0o555)
    with pytest.raises(ValueError, match="does not authorize"):
        recovery.load_hostile_review(
            NEW_COMMIT,
            contract,
            runtime,
            str(path),
            recovery.sha256_file(path),
        )


def test_confirmation_generator_substitution_is_rejected():
    source_hashes = {
        name: f"{index + 1:064x}"
        for index, name in enumerate(upstream.SCIENTIFIC_SOURCE_PATHS)
    }
    source_contract = {
        "git_commit": recovery.UPSTREAM_SOURCE_COMMIT,
        "manifest_sha256": recovery.UPSTREAM_SOURCE_MANIFEST_SHA256,
    }
    generator_sources = {
        name: source_hashes[name]
        for name in upstream.CANONICAL_CONFIRMATION_GENERATOR_SOURCES
    }
    plan = {
        "confirmation_commitment": {
            "document": {
                "source_contract": source_contract,
                "generator_source_contract": {
                    "schema": upstream.CANONICAL_CONFIRMATION_GENERATOR_SCHEMA,
                    "entrypoint": upstream.CANONICAL_CONFIRMATION_GENERATOR_ENTRYPOINT,
                    "sources": generator_sources,
                    "manifest_sha256": upstream.stable_json_sha256(generator_sources),
                },
            }
        }
    }
    recovery.validate_confirmation_generator_contract(
        plan, source_contract, source_hashes
    )
    hostile = copy.deepcopy(plan)
    hostile["confirmation_commitment"]["document"]["generator_source_contract"][
        "entrypoint"
    ] = "recovery:generate_confirmation_board"
    with pytest.raises(ValueError, match="substitution"):
        recovery.validate_confirmation_generator_contract(
            hostile, source_contract, source_hashes
        )


def _legacy_payload_fixture(monkeypatch):
    fit_budget = {
        "rank": 2,
        "updates": 4,
        "batch_size": 3,
        "lr": 0.003,
        "weight_decay": 0.0001,
        "schedule_sha256": "a" * 64,
        "initial_state_sha256": "b" * 64,
    }
    board = {"rows": 3, "canonical": True}
    control = {"seed": 7, "derangement": True}
    feature_merge = {"rows": 3, "deployment_logit_dtype": "torch.float32"}
    source_hashes = {"train/model.py": "c" * 64}
    source_contract = {
        "git_commit": recovery.UPSTREAM_SOURCE_COMMIT,
        "manifest_sha256": "d" * 64,
    }
    expected_bindings = {
        "base_checkpoint_sha256": "1" * 64,
        "tokenizer_sha256": "2" * 64,
        "episodes_sha256": "3" * 64,
        "cycle_sha256": "4" * 64,
        "confirmation_commitment_sha256": "5" * 64,
    }
    context = {
        "upstream_plan": {
            "fit_budget": fit_budget,
            "checkpoint_step": 280000,
            "d_model": 4,
            "zero_id": 11,
            "one_id": 12,
            "runtime_contract": {"extract_batch": 32},
        },
        "expected_bindings": expected_bindings,
        "upstream_source_hashes": source_hashes,
        "upstream_source_contract": source_contract,
        "feature_merge": feature_merge,
        "features": {"deployment_logit_dtype": "torch.float32"},
        "board_context": {
            "normalized_board": board,
            "control": control,
            "rows": [],
            "control_labels": torch.tensor([], dtype=torch.long),
        },
    }
    state = collections.OrderedDict(
        (
            ("down.weight", torch.zeros(2, 4)),
            ("down.bias", torch.zeros(2)),
            ("up.weight", torch.zeros(2, 2)),
            ("up.bias", torch.zeros(2)),
        )
    )
    fit_report = {
        "updates": 4,
        "batch_size": 3,
        "lr": 0.003,
        "weight_decay": 0.0001,
        "schedule_sha256": "a" * 64,
        "first_loss": 1.25,
        "final_loss": 0.75,
        "min_loss": 0.5,
    }
    linear = {
        "train_rows": 2,
        "test_rows": 1,
        "test_correct": 1,
        "test_accuracy": 1.0,
        "schedule_sha256": "e" * 64,
        "claim_boundary": "diagnostic only",
    }
    evidence = {"accuracy": 1.0, "correct": 3, "nested": {"passed": True}}
    monkeypatch.setattr(recovery, "_expected_teacher_evidence", lambda *_args: evidence)
    payload = {
        "plan_sha256": recovery.UPSTREAM_PLAN_SHA256,
        **expected_bindings,
        "scientific_source_sha256": source_hashes,
        "source_contract": source_contract,
        "checkpoint_step": 280000,
        "d_model": 4,
        "rank": 2,
        "parameter_count": 16,
        "extract_batch": 32,
        "feature_shard_merge": feature_merge,
        "deployment_logit_dtype": "torch.float32",
        "zero_id": 11,
        "one_id": 12,
        "initial_state_sha256": "b" * 64,
        "treatment": copy.deepcopy(state),
        "shuffled": copy.deepcopy(state),
        "treatment_state_sha256": "6" * 64,
        "shuffled_state_sha256": "7" * 64,
        "board": board,
        "control": control,
        "treatment_fit": copy.deepcopy(fit_report),
        "shuffled_fit": copy.deepcopy(fit_report),
        "linear_diagnostic": linear,
        "fit_feature_metrics": evidence,
        "claim_boundary": upstream.CANONICAL_FIT_CLAIM_BOUNDARY,
    }
    assert set(payload) == recovery.LEGACY_PAYLOAD_KEYS
    return payload, context


def test_legacy_payload_validator_is_complete_and_type_strict(monkeypatch):
    payload, context = _legacy_payload_fixture(monkeypatch)
    recovery.validate_legacy_payload_type_strict(payload, context, "cpu")

    attacks = []
    hostile = copy.deepcopy(payload)
    hostile["checkpoint_step"] = 280000.0
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["d_model"] = True
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["rank"] = 2.0
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["zero_id"] = False
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["board"]["rows"] = 3.0
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["control"]["seed"] = 7.0
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["treatment"] = dict(hostile["treatment"])
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["treatment_fit"]["updates"] = 4.0
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["shuffled_fit"]["first_loss"] = 1
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["linear_diagnostic"]["test_correct"] = True
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["linear_diagnostic"]["test_accuracy"] = 1
    attacks.append(hostile)
    hostile = copy.deepcopy(payload)
    hostile["fit_feature_metrics"]["accuracy"] = 1
    attacks.append(hostile)

    for hostile in attacks:
        with pytest.raises(ValueError, match="legacy payload"):
            recovery.validate_legacy_payload_type_strict(hostile, context, "cpu")


def test_v9_output_has_dual_provenance_and_cannot_publish_as_v8(monkeypatch):
    executor = {"git_commit": NEW_COMMIT}
    runtime = {"schema": recovery.RECOVERY_EXECUTOR_RUNTIME_SCHEMA}
    upstream_contract = {"git_commit": recovery.UPSTREAM_SOURCE_COMMIT}
    plan_binding = {"path": "/upstream/plan.json", "sha256": "1" * 64}
    receipts = [{"shard_index": index} for index in range(8)]
    proof = {"mismatch_count": 2}
    recovery_plan = {
        "recovery_executor_source_contract": executor,
        "executor_runtime_contract": runtime,
        "upstream_protocol": {
            "source_contract": upstream_contract,
            "plan_binding": plan_binding,
            "shard_receipts": receipts,
        },
        "normalization_proof": proof,
    }
    fit_payload = {key: None for key in recovery.LEGACY_PAYLOAD_KEYS}
    bundle = {
        "audit": recovery.RECOVERY_FIT_AUDIT,
        "recovery": True,
        "recovery_plan_sha256": "2" * 64,
        "recovery_executor_source_contract": executor,
        "executor_runtime_contract": runtime,
        "upstream_protocol_source_contract": upstream_contract,
        "upstream_plan_binding": plan_binding,
        "upstream_shard_receipts": receipts,
        "normalization_proof": proof,
        "allowed_transformation": recovery.ALLOWED_TRANSFORMATION,
        "deserialization_contract": recovery.DESERIALIZATION_CONTRACT,
        "fit_payload": fit_payload,
        "claim_boundary": recovery.RECOVERY_FIT_CLAIM_BOUNDARY,
    }
    monkeypatch.setattr(
        upstream, "_validate_motor_bundle_against_replayed_features", lambda *_: None
    )
    monkeypatch.setattr(
        recovery, "validate_legacy_payload_type_strict", lambda *_: None
    )
    context = {
        "expected_bindings": {},
        "upstream_source_hashes": {},
        "upstream_source_contract": upstream_contract,
        "upstream_plan": {},
        "features": {},
        "feature_merge": {},
    }
    recovery.validate_recovery_fit_bundle(
        bundle, recovery_plan, "2" * 64, context, "cpu"
    )
    assert bundle["audit"] == recovery.RECOVERY_FIT_AUDIT
    assert "canonical" not in bundle
    hostile = copy.deepcopy(bundle)
    hostile["audit"] = upstream.CANONICAL_FIT_AUDIT
    with pytest.raises(ValueError, match="not a v9"):
        recovery.validate_recovery_fit_bundle(
            hostile, recovery_plan, "2" * 64, context, "cpu"
        )


def test_slurm_wrapper_is_no_requeue_and_never_opens_confirmation_secret():
    wrapper = (
        Path(__file__).resolve().parent / "jobs" / "causal_carry_motor_recovery.sbatch"
    )
    text = wrapper.read_text()
    assert "#SBATCH --no-requeue" in text
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in text
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" in text
    assert "TORCH_FORCE_WEIGHTS_ONLY_LOAD" in text
    assert "readonly PY=$DATA_ROOT/miniforge3/bin/python" in text
    assert "${PY:-" not in text
    assert "readonly GIT=/usr/bin/git" in text
    assert "rev-list --parents -n 1" in text
    assert "diff --name-status --no-renames" in text
    assert "ls-files -z" in text
    assert "source checkout is not closed-world" in text
    assert '"$TREE_MODE" != 100644' in text
    for expected in recovery.RECOVERY_NAME_STATUS_DIFF:
        assert expected.replace("\t", "\\t") in text
    assert "hostile_review" in text
    assert "${CONFIRMATION_SECRET_FILE:?" not in text
    assert 'cat "$CONFIRMATION_SECRET_FILE"' not in text
    assert 'source "$CONFIRMATION_SECRET_FILE"' not in text
    assert '"$PY" train/causal_carry_motor.py' not in text
    assert '"$PY" train/causal_carry_motor_recovery.py fit' in text
    assert "atomic_torch" not in text
    assert "ln " not in text
