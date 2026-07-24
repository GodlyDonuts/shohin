from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

from pipeline.assess_episode_causal_workspace import (
    load_assessor_rows,
    promotion_diagnostics,
    summarize_control,
)
from pipeline.decide_episode_causal_workspace import build_decision
import pipeline.episode_workspace_custody as workspace_custody
from pipeline.episode_workspace_custody import (
    DEFAULT_CUSTODY_BUNDLE,
    committed_source_receipt,
    file_sha256,
    read_json_verified,
    read_jsonl_verified,
    verify_landlock_stage,
    write_json_fsync,
)
from pipeline.publish_episode_workspace_experiment import (
    EXPECTED_STAGE_BUNDLES,
    EXPECTED_STAGE_DIRECTORIES,
    publish_experiment,
)
import pipeline.stage_episode_workspace_sources as source_stager
from compile_episode_causal_workspace import load_worlds
from execute_episode_causal_workspace import load_queries
from fit_episode_causal_workspace import (
    FitConfig,
    LengthBucketScheduler,
    load_train_groups,
    make_fit_batch,
    validate_frozen_arm_input,
)
from causal_bind_select_workspace import (
    CausalWorkspaceConfig,
    CausalWorkspaceGPT,
    WorkspaceState,
)
from model import GPT, GPTConfig
from workspace_checkpoint import (
    ProtectedCheckpointReceipt,
    file_sha256 as checkpoint_file_sha256,
    runtime_source_manifest,
)
from workspace_state_custody import (
    COMPILER_SOURCE_PATHS,
    COMPILER_SOURCE_RECEIPT_SCHEMA,
    WorkspaceStateCustodyError,
    load_compiled_states,
    save_compiled_states,
)


def _landlock_receipt(
    stage: str,
    process_id: int,
    denied_path_name: str,
) -> dict[str, object]:
    denied_path = Path(denied_path_name)
    if not denied_path.is_absolute():
        denied_path = Path("/sealed") / denied_path
    policy = {
        "schema": "shohin_landlock_stage_policy_v1",
        "landlock_abi": 6,
        "stage": stage,
        "handled_access_fs": (1 << 15) - 1,
        "handled_access_fs_names": ["read_file"],
        "rules": [],
    }
    return {
        "schema": "shohin_landlock_stage_receipt_v1",
        "stage": stage,
        "enforced": True,
        "dumpable": False,
        "abi": 6,
        "policy_sha256": workspace_custody.json_sha256(policy),
        "canonical_policy": policy,
        "process_id": process_id,
        "denied_probe_receipt": {
            "schema": "shohin_landlock_denied_probe_receipt_v1",
            "stage": stage,
            "process_id": process_id,
            "operation": "open_read",
            "path": str(denied_path),
            "path_name": denied_path.name,
            "path_sha256": hashlib.sha256(os.fsencode(denied_path)).hexdigest(),
            "denied": True,
            "errno": 13,
        },
    }


def _compiler_source_receipt(process_id: int = 2) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_manifest = {
        path: checkpoint_file_sha256(Path(path)) for path in COMPILER_SOURCE_PATHS
    }
    return {
        "schema": COMPILER_SOURCE_RECEIPT_SCHEMA,
        "primary_path": "train/compile_episode_causal_workspace.py",
        "primary_sha256": source_manifest[
            "train/compile_episode_causal_workspace.py"
        ],
        "repository_commit": commit,
        "local_source_manifest": source_manifest,
        "runtime_source_manifest": runtime_source_manifest(),
        "process_id": process_id,
        "landlock_receipt": _landlock_receipt(
            "compiler",
            process_id,
            "development_queries.jsonl",
        ),
    }


@pytest.fixture(scope="module")
def manifest() -> dict[str, object]:
    return json.loads(
        (DEFAULT_CUSTODY_BUNDLE / "custody_manifest.json").read_text(encoding="utf-8")
    )


def test_custody_bundle_is_physically_disjoint_and_hash_frozen(manifest) -> None:
    assert DEFAULT_CUSTODY_BUNDLE.is_dir()
    assert manifest["pretraining_started"] is False
    assert manifest["continuation_pretraining_authorized"] is False
    assert manifest["counts"] == {
        "train_true_groups": 256,
        "train_shuffled_groups": 256,
        "train_packets_per_arm": 1536,
        "development_worlds": 192,
        "development_queries": 384,
        "development_assessor_rows": 384,
    }
    for name, expected in manifest["files"].items():
        assert file_sha256(DEFAULT_CUSTODY_BUNDLE / name) == expected
    assert manifest["optimizer_visible_files"] == [
        "train_true_groups.jsonl",
        "train_shuffled_groups.jsonl",
    ]
    assert manifest["compiler_visible_files"] == ["development_worlds.jsonl"]
    assert manifest["executor_visible_files"] == ["development_queries.jsonl"]
    assert manifest["assessor_visible_files"] == ["development_assessor.jsonl"]


def test_fit_artifacts_contain_only_six_case_packets_and_targets(manifest) -> None:
    files = manifest["files"]
    true_groups = load_train_groups(
        DEFAULT_CUSTODY_BUNDLE / "train_true_groups.jsonl",
        files["train_true_groups.jsonl"],
    )
    shuffled_groups = load_train_groups(
        DEFAULT_CUSTODY_BUNDLE / "train_shuffled_groups.jsonl",
        files["train_shuffled_groups.jsonl"],
    )
    assert len(true_groups) == len(shuffled_groups) == 256
    true_examples = [example for group in true_groups for example in group.examples]
    shuffled_examples = [
        example for group in shuffled_groups for example in group.examples
    ]
    assert [item.packet_sha256 for item in true_examples] == [
        item.packet_sha256 for item in shuffled_examples
    ]
    assert [item.world_tokens for item in true_examples] == [
        item.world_tokens for item in shuffled_examples
    ]
    assert [item.query_tokens for item in true_examples] == [
        item.query_tokens for item in shuffled_examples
    ]
    assert all(
        true.target_token != shuffled.target_token
        for true, shuffled in zip(
            true_examples,
            shuffled_examples,
            strict=True,
        )
    )


def test_verified_json_inputs_are_hashed_and_parsed_from_one_open(
    tmp_path,
    monkeypatch,
) -> None:
    json_path = tmp_path / "value.json"
    json_path.write_text('{"value":1}\n', encoding="ascii")
    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text('{"row":1}\n{"row":2}\n', encoding="ascii")
    expected = {
        json_path: file_sha256(json_path),
        jsonl_path: file_sha256(jsonl_path),
    }
    open_counts = {json_path: 0, jsonl_path: 0}
    original_open = Path.open

    def tracked_open(path, *args, **kwargs):
        if path in open_counts:
            open_counts[path] += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    assert read_json_verified(json_path, expected[json_path]) == {"value": 1}
    assert read_jsonl_verified(jsonl_path, expected[jsonl_path]) == [
        {"row": 1},
        {"row": 2},
    ]
    assert open_counts == {json_path: 1, jsonl_path: 1}


def test_landlock_receipt_requires_a_real_access_denial(
    tmp_path,
    monkeypatch,
) -> None:
    denied = tmp_path / "development_queries.jsonl"
    denied.write_text("{}\n", encoding="ascii")
    monkeypatch.setenv("SHOHIN_LANDLOCK_ENFORCED", "1")
    monkeypatch.setenv("SHOHIN_LANDLOCK_STAGE", "compiler")
    monkeypatch.setenv("SHOHIN_LANDLOCK_ABI", "6")
    policy = {
        "schema": "shohin_landlock_stage_policy_v1",
        "landlock_abi": 6,
        "stage": "compiler",
        "handled_access_fs": (1 << 15) - 1,
        "handled_access_fs_names": ["read_file"],
        "rules": [],
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(policy, sort_keys=True, separators=(",", ":")),
        encoding="ascii",
    )
    policy_sha256 = file_sha256(policy_path)
    monkeypatch.setenv("SHOHIN_LANDLOCK_POLICY_SHA256", policy_sha256)
    monkeypatch.setenv("SHOHIN_LANDLOCK_POLICY_PATH", str(policy_path))
    monkeypatch.setattr(workspace_custody, "_process_dumpable", lambda: 0)
    original_open = Path.open

    def deny_open(path, *args, **kwargs):
        if path == denied:
            raise PermissionError(13, "permission denied", str(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_open)
    receipt = verify_landlock_stage("compiler", denied)
    assert receipt["enforced"] is True
    assert receipt["canonical_policy"] == policy
    assert receipt["denied_probe_receipt"]["path_name"] == denied.name
    assert receipt["denied_probe_receipt"]["path"] == str(denied)
    monkeypatch.setattr(Path, "open", original_open)
    with pytest.raises(ValueError, match="allowed a forbidden"):
        verify_landlock_stage("compiler", denied)


def test_landlocked_source_receipt_uses_sealed_manifest_without_git(
    tmp_path,
    monkeypatch,
) -> None:
    primary = Path("train/fit_episode_causal_workspace.py").resolve()
    dependency = Path("pipeline/episode_workspace_custody.py").resolve()
    source_manifest = {
        "train/fit_episode_causal_workspace.py": file_sha256(primary),
        "pipeline/episode_workspace_custody.py": file_sha256(dependency),
    }
    sealed = tmp_path / "source_receipt.json"
    write_json_fsync(
        sealed,
        {
            "schema": "shohin_episode_workspace_source_bundle_v1",
            "repository_commit": "a" * 40,
            "source_manifest": source_manifest,
            "pretraining_started": False,
            "continuation_pretraining_authorized": False,
        },
    )
    monkeypatch.setenv("SHOHIN_LANDLOCK_ENFORCED", "1")
    monkeypatch.setenv("SHOHIN_SOURCE_RECEIPT_PATH", str(sealed))
    monkeypatch.setenv("SHOHIN_SOURCE_RECEIPT_SHA256", file_sha256(sealed))

    def reject_git(*args, **kwargs):
        raise AssertionError("sealed source validation must not invoke Git")

    monkeypatch.setattr(subprocess, "run", reject_git)
    receipt = committed_source_receipt(
        primary,
        source_manifest["train/fit_episode_causal_workspace.py"],
        (dependency,),
    )
    assert receipt["repository_commit"] == "a" * 40
    assert receipt["local_source_manifest"] == source_manifest


def test_fit_arm_is_cryptographically_bound_to_frozen_ledger() -> None:
    true = validate_frozen_arm_input(
        "true",
        Path("train_true_groups.jsonl"),
        "80d7e6e503d4aebbda506fcb3d321f1a91db556f7fe1bd2ab8e6ee92d2fbec27",
    )
    assert true["name"] == "train_true_groups.jsonl"
    with pytest.raises(ValueError, match="wrong frozen input"):
        validate_frozen_arm_input(
            "true",
            Path("train_shuffled_groups.jsonl"),
            true["sha256"],
        )
    with pytest.raises(ValueError, match="frozen ledger"):
        validate_frozen_arm_input(
            "true",
            Path("train_true_groups.jsonl"),
            "0" * 64,
        )


def test_whole_experiment_is_published_only_after_complete_validation(
    tmp_path,
) -> None:
    staging = tmp_path / ".experiment.staging"
    output = tmp_path / "experiment"
    source = staging / "source"
    source.mkdir(parents=True)
    source_file = source / "train" / "model.py"
    source_file.parent.mkdir()
    source_file.write_text("MODEL = 1\n", encoding="ascii")
    write_json_fsync(
        source / "source_receipt.json",
        {
            "schema": "shohin_episode_workspace_source_bundle_v1",
            "repository_commit": "a" * 40,
            "source_manifest": {
                "train/model.py": file_sha256(source_file),
            },
            "pretraining_started": False,
            "continuation_pretraining_authorized": False,
        },
    )
    for stage in EXPECTED_STAGE_DIRECTORIES:
        root = staging / stage
        root.mkdir()
        expected_schema, expected_files, report_spec = EXPECTED_STAGE_BUNDLES[stage]
        report_name, report_schema = report_spec
        file_hashes = {}
        for name in expected_files:
            artifact = root / name
            if name == "decision.json":
                write_json_fsync(
                    artifact,
                    {
                        "schema": "episode_causal_workspace_decision_v1",
                        "reasoning_promotion_authorized": False,
                        "continuation_pretraining_authorized": False,
                        "pretraining_started": False,
                    },
                )
            elif name == report_name:
                write_json_fsync(
                    artifact,
                    {
                        "schema": report_schema,
                        "stage": stage,
                        "continuation_pretraining_authorized": False,
                        "pretraining_started": False,
                    },
                )
            elif name.endswith(".json"):
                write_json_fsync(
                    artifact,
                    {
                        "stage": stage,
                        "pretraining_started": False,
                    },
                )
            elif name.endswith(".jsonl"):
                artifact.write_text("{}\n", encoding="ascii")
            else:
                artifact.write_bytes(b"sealed tensor placeholder")
            file_hashes[name] = file_sha256(artifact)
        write_json_fsync(
            root / "bundle_manifest.json",
            {
                "schema": expected_schema,
                "files": file_hashes,
                "pretraining_started": False,
            },
        )
    decision_sha256 = file_sha256(staging / "decision" / "decision.json")
    invalid_staging = tmp_path / ".invalid-experiment.staging"
    invalid_output = tmp_path / "invalid-experiment"
    shutil.copytree(staging, invalid_staging)
    invalid_manifest_path = invalid_staging / "true_fit" / "bundle_manifest.json"
    invalid_manifest = json.loads(invalid_manifest_path.read_text(encoding="utf-8"))
    invalid_manifest["schema"] = "attacker_controlled_bundle_v1"
    invalid_manifest_path.write_text(
        json.dumps(invalid_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stage manifest flags differ"):
        publish_experiment(
            invalid_staging,
            invalid_output,
            expected_commit="a" * 40,
            expected_source_receipt_sha256=file_sha256(
                invalid_staging / "source" / "source_receipt.json"
            ),
            expected_decision_sha256=file_sha256(
                invalid_staging / "decision" / "decision.json"
            ),
            slurm_job_id="test",
            slurm_node="test-node",
        )
    result = publish_experiment(
        staging,
        output,
        expected_commit="a" * 40,
        expected_source_receipt_sha256=file_sha256(
            source / "source_receipt.json"
        ),
        expected_decision_sha256=decision_sha256,
        slurm_job_id="test",
        slurm_node="test-node",
    )
    assert not staging.exists()
    assert output.is_dir() and not output.is_symlink()
    assert result["pretraining_started"] is False
    assert file_sha256(output / "experiment_manifest.json") == result[
        "experiment_manifest_sha256"
    ]


def test_experiment_publisher_rejects_symlinked_staging_root(tmp_path) -> None:
    real_staging = tmp_path / ".real.staging"
    real_staging.mkdir()
    symlink_staging = tmp_path / ".symlink.staging"
    symlink_staging.symlink_to(real_staging, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink directory"):
        publish_experiment(
            symlink_staging,
            tmp_path / "output",
            expected_commit="a" * 40,
            expected_source_receipt_sha256="b" * 64,
            expected_decision_sha256="c" * 64,
            slurm_job_id="test",
            slurm_node="test-node",
        )


def test_source_stager_reads_exact_committed_bytes_into_regular_tree(
    tmp_path,
    monkeypatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "custody@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Custody Test"],
        cwd=repository,
        check=True,
    )
    source = repository / "train" / "unit.py"
    source.parent.mkdir()
    source.write_text("VALUE = 7\n", encoding="ascii")
    subprocess.run(["git", "add", "train/unit.py"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "source"], cwd=repository, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(source_stager, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(source_stager, "SOURCE_PATHS", ("train/unit.py",))
    output = tmp_path / "staged"
    receipt = source_stager.stage_sources(commit, output)
    assert output.is_dir() and not output.is_symlink()
    assert (output / "train/unit.py").read_bytes() == b"VALUE = 7\n"
    assert receipt["repository_commit"] == commit
    assert receipt["source_manifest"]["train/unit.py"] == file_sha256(source)
    assert receipt["pretraining_started"] is False


def test_length_bucket_scheduler_never_mixes_active_lengths(manifest) -> None:
    groups = load_train_groups(
        DEFAULT_CUSTODY_BUNDLE / "train_true_groups.jsonl",
        manifest["files"]["train_true_groups.jsonl"],
    )
    scheduler = LengthBucketScheduler(
        groups,
        groups_per_batch=8,
        seed=2026072347,
    )
    seen: set[str] = set()
    for _ in range(40):
        selected = scheduler.next()
        assert len({group.query_length for group in selected}) == 1
        assert all(len(group.examples) == 6 for group in selected)
        seen.update(
            example.packet_sha256 for group in selected for example in group.examples
        )
    assert len(seen) == 1_536


def test_fit_batch_supervises_exactly_one_answer_position(manifest) -> None:
    groups = load_train_groups(
        DEFAULT_CUSTODY_BUNDLE / "train_true_groups.jsonl",
        manifest["files"]["train_true_groups.jsonl"],
    )
    same_length = tuple(group for group in groups if group.query_length == 7)[:8]
    batch = make_fit_batch(same_length, device=torch.device("cpu"))
    assert batch.world_idx.shape == (48, 145)
    assert batch.query_idx.shape == (48, 7)
    assert batch.answer_index == 5
    assert torch.equal(
        (batch.targets != -1).sum(dim=1),
        torch.ones(48, dtype=torch.long),
    )
    assert torch.all(batch.query_idx[:, batch.answer_index] == 7)
    assert torch.all(batch.query_idx[:, batch.answer_index + 1] == 8)


def test_world_and_query_stage_inputs_are_label_free(manifest) -> None:
    files = manifest["files"]
    worlds = load_worlds(
        DEFAULT_CUSTODY_BUNDLE / "development_worlds.jsonl",
        files["development_worlds.jsonl"],
    )
    queries = load_queries(
        DEFAULT_CUSTODY_BUNDLE / "development_queries.jsonl",
        files["development_queries.jsonl"],
    )
    assert len(worlds) == 192
    assert len(queries) == 384
    assert {len(tokens) for _, tokens in worlds} == {145}
    assert {len(tokens) for _, _, tokens in queries} == {13, 15}
    assert len({world_id for _, world_id, _ in queries}) == 192


def test_compiled_state_round_trip_contains_slots_and_receipts_only(tmp_path) -> None:
    base = GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=24,
            d_ff=48,
            seq_len=256,
            zloss=0.0,
        )
    )
    config = CausalWorkspaceConfig(
        d_model=24,
        slot_width=16,
        num_slots=4,
        num_operators=4,
        operator_rank=4,
        stage_after_block=1,
    )
    model = CausalWorkspaceGPT(base, config)
    parameter_receipt = model.workspace.parameter_receipt(
        protected_base_parameters=base.num_params()
    )
    receipt = ProtectedCheckpointReceipt(
        checkpoint_path="/test/protected.pt",
        checkpoint_bytes=1,
        checkpoint_sha256="a" * 64,
        step=300_000,
        data_seed=777,
        data_stream_generation=1,
        data_stream_seed=1_000_780,
        base_config=asdict(base.cfg),
        config_sha256="b" * 64,
        base_state_sha256="c" * 64,
        state_key_sha256="d" * 64,
        state_key_count=len(base.state_dict()),
        protected_base_parameters=parameter_receipt.protected_base_parameters,
        workspace_parameters=parameter_receipt.workspace_parameters,
        complete_system_parameters=parameter_receipt.complete_system_parameters,
        remaining_under_cap=parameter_receipt.remaining_under_cap,
        strict_state_load=True,
        protected_base_frozen=True,
        optimizer_state_loaded=False,
        pretraining_started=False,
    )
    states = {
        f"{index:064x}": WorkspaceState(
            slots=torch.randn(1, 4, 16),
            token_position=145,
            sealed=True,
        )
        for index in range(192)
    }
    path = tmp_path / "states.pt"
    compiler_receipt = _compiler_source_receipt()
    with pytest.raises(WorkspaceStateCustodyError, match="fields differ"):
        save_compiled_states(
            tmp_path / "invalid.pt",
            states,
            model=model,
            protected_receipt=receipt,
            workspace_delta_sha256="e" * 64,
            world_source_sha256="f" * 64,
            compiler_source_receipt={"sha256": "1" * 64},
        )
    digest = save_compiled_states(
        path,
        states,
        model=model,
        protected_receipt=receipt,
        workspace_delta_sha256="e" * 64,
        world_source_sha256="f" * 64,
        compiler_source_receipt=compiler_receipt,
    )
    assert digest == checkpoint_file_sha256(path)
    loaded, loaded_receipt = load_compiled_states(
        path,
        model=model,
        protected_receipt=receipt,
        expected_sha256=digest,
        expected_workspace_delta_sha256="e" * 64,
        expected_world_source_sha256="f" * 64,
        expected_compiler_source_sha256=compiler_receipt["primary_sha256"],
        expected_repository_commit=compiler_receipt["repository_commit"],
    )
    assert set(loaded) == set(states)
    assert loaded_receipt["source_tokens_serialized"] is False
    assert loaded_receipt["query_tokens_seen"] is False
    assert loaded_receipt["labels_seen"] is False
    assert loaded_receipt["optimizer_state"] is None
    for world_id in loaded:
        assert torch.equal(loaded[world_id].slots, states[world_id].slots)
        assert loaded[world_id].sealed is True
    with pytest.raises(WorkspaceStateCustodyError, match="hash mismatch"):
        load_compiled_states(
            path,
            model=model,
            protected_receipt=receipt,
            expected_sha256="0" * 64,
            expected_workspace_delta_sha256="e" * 64,
            expected_world_source_sha256="f" * 64,
            expected_compiler_source_sha256=compiler_receipt["primary_sha256"],
            expected_repository_commit=compiler_receipt["repository_commit"],
        )


def test_perfect_frozen_predictions_recover_all_factorial_metrics(manifest) -> None:
    rows = load_assessor_rows(
        DEFAULT_CUSTODY_BUNDLE / "development_assessor.jsonl",
        manifest["files"]["development_assessor.jsonl"],
    )
    packet_order = [str(row["packet_sha256"]) for row in rows]
    logits = torch.zeros(len(rows), 32768)
    predictions = []
    for index, row in enumerate(rows):
        target = int(row["target_token"])
        logits[index, target] = 10.0
        predictions.append(
            {
                "control": "treatment",
                "packet_sha256": row["packet_sha256"],
                "predicted_token": target,
            }
        )
    summary, assessed = summarize_control(
        rows,
        predictions,
        {
            "packet_sha256": packet_order,
            "answer_logits": logits,
        },
    )
    assert len(assessed) == 384
    assert summary["packets"]["rate"] == 1.0
    assert summary["complete_six_case_clusters"]["rate"] == 1.0
    assert summary["complete_cyclic_triples"]["rate"] == 1.0
    assert summary["complete_order_pairs"]["rate"] == 1.0
    assert summary["by_depth"]["5"]["rate"] == 1.0
    assert summary["by_depth"]["6"]["rate"] == 1.0


def test_promotion_diagnostics_never_authorize_reasoning_or_pretraining() -> None:
    perfect = {
        "packets": {"rate": 1.0},
        "complete_cyclic_triples": {"rate": 1.0},
        "complete_order_pairs": {"rate": 1.0},
        "by_depth": {"5": {"rate": 1.0}, "6": {"rate": 1.0}},
    }
    zero = {**perfect, "packets": {"rate": 0.0}}
    controls = {
        "treatment": perfect,
        "selected_slot_scramble": zero,
        "discarded_slot_scramble": perfect,
    }
    diagnostics = promotion_diagnostics(controls)
    assert diagnostics["selected_slot_scramble_cost"] == 1.0
    assert diagnostics["discarded_slot_scramble_cost"] == 0.0
    assert diagnostics["all_gates_pass"] is False
    assert diagnostics["reasoning_promotion_authorized"] is False
    assert diagnostics["continuation_pretraining_authorized"] is False


def test_cross_arm_decision_can_advance_only_to_replication() -> None:
    perfect = {
        "packets": {"rate": 1.0},
        "complete_cyclic_triples": {"rate": 1.0},
        "complete_order_pairs": {"rate": 1.0},
        "by_depth": {"5": {"rate": 1.0}, "6": {"rate": 1.0}},
    }
    zero = {**perfect, "packets": {"rate": 0.0}}

    def fit(arm: str, pid: int, delta: str) -> dict[str, object]:
        input_name = (
            "train_true_groups.jsonl"
            if arm == "true"
            else "train_shuffled_groups.jsonl"
        )
        input_sha256 = (
            "80d7e6e503d4aebbda506fcb3d321f1a91db556f7fe1bd2ab8e6ee92d2fbec27"
            if arm == "true"
            else "5917049c910cdc2beae667165465052c71329033be30532a4cec5a04fe419038"
        )
        return {
            "arm": arm,
            "process_id": pid,
            "landlock_receipt": _landlock_receipt(
                "fit",
                pid,
                "development_assessor.jsonl",
            ),
            "optimizer_visible_input": {
                "path": f"/sealed/{input_name}",
                "sha256": input_sha256,
                "frozen_arm_binding": {
                    "name": input_name,
                    "sha256": input_sha256,
                },
                "rows": 256,
                "packets": 1_536,
            },
            "fit_config": {"seed": 7},
            "workspace_initial_state_sha256": "a" * 64,
            "protected_checkpoint": {
                "checkpoint_path": "/repo/train/flagship_out/ckpt_0300000.pt",
                "sha256": "b" * 64,
            },
            "workspace_delta_sha256": delta,
        }

    def assessment(
        *,
        pid: int,
        compiler_pid: int,
        executor_pid: int,
        delta: str,
        treatment,
    ) -> dict[str, object]:
        return {
            "process_id": pid,
            "landlock_receipt": _landlock_receipt(
                "assessor",
                pid,
                "development_worlds.jsonl",
            ),
            "assessor_source": {"open_count": 1},
            "control_summaries": {
                "treatment": treatment,
                "selected_slot_scramble": zero,
                "discarded_slot_scramble": treatment,
            },
            "execution_report": {
                "process_id": executor_pid,
                "landlock_receipt": _landlock_receipt(
                    "executor",
                    executor_pid,
                    "development_worlds.jsonl",
                ),
                "executor_visible_inputs": {
                    "queries": {
                        "path": "/sealed/development_queries.jsonl",
                        "sha256": "e" * 64,
                    }
                },
                "world_tokens_seen": False,
                "targets_seen": False,
                "candidate_sets_seen": False,
                "compiled_state_receipt": {
                    "compiler_source_receipt": {
                        "process_id": compiler_pid,
                        "landlock_receipt": _landlock_receipt(
                            "compiler",
                            compiler_pid,
                            "development_queries.jsonl",
                        ),
                    },
                    "workspace_delta_sha256": delta,
                    "source_tokens_serialized": False,
                    "query_tokens_seen": False,
                    "labels_seen": False,
                },
            },
        }

    true_delta = "c" * 64
    shuffled_delta = "d" * 64
    true_fit = fit("true", 1, true_delta)
    shuffled_fit = fit("shuffled", 5, shuffled_delta)
    true_assessment = assessment(
        pid=4,
        compiler_pid=2,
        executor_pid=3,
        delta=true_delta,
        treatment=perfect,
    )
    shuffled_assessment = assessment(
        pid=8,
        compiler_pid=6,
        executor_pid=7,
        delta=shuffled_delta,
        treatment=zero,
    )
    decision = build_decision(
        true_fit,
        shuffled_fit,
        true_assessment,
        shuffled_assessment,
    )
    assert decision["all_tuning_gates_pass"] is True
    assert decision["next_action"] == "source_poison_and_two_replications"
    assert decision["remaining_confirmation_gates"] == {
        "post_compile_source_poison_bit_identity": False,
        "three_fresh_seed_replication": False,
        "unopened_confirmation_manifest": False,
    }

    compromised_fit = json.loads(json.dumps(true_fit))
    compromised_receipt = compromised_fit["landlock_receipt"]
    compromised_policy = compromised_receipt["canonical_policy"]
    compromised_policy["rules"] = [
        {
            "path": "/sealed",
            "object_type": "directory",
            "st_dev": 1,
            "st_ino": 1,
            "allowed_access_fs": 1 << 2,
            "allowed_access_fs_names": ["read_file"],
        }
    ]
    compromised_receipt["policy_sha256"] = workspace_custody.json_sha256(
        compromised_policy
    )
    rejected = build_decision(
        compromised_fit,
        shuffled_fit,
        true_assessment,
        shuffled_assessment,
    )
    assert rejected["tuning_gates"]["process_level_source_deletion"] is False
    assert rejected["all_tuning_gates_pass"] is False
    assert decision["reasoning_promotion_authorized"] is False
    assert decision["continuation_pretraining_authorized"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"updates": 0},
        {"groups_per_batch": 0},
        {"learning_rate": 0.0},
        {"weight_decay": -1.0},
    ],
)
def test_invalid_fit_configuration_fails_closed(override) -> None:
    defaults = asdict(FitConfig())
    defaults.update(override)
    with pytest.raises(ValueError):
        FitConfig(**defaults)


def test_no_pretraining_job_source_is_present_in_stage_implementation() -> None:
    paths = (
        Path("train/fit_episode_causal_workspace.py"),
        Path("train/compile_episode_causal_workspace.py"),
        Path("train/execute_episode_causal_workspace.py"),
        Path("pipeline/assess_episode_causal_workspace.py"),
    )
    forbidden = ("sbatch", "train.py", "SHARDS", "fresh_optimizer")
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden)
