from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import stat
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import eval_digitwise_factorial_v4 as evaluator  # noqa: E402
from digitwise_protocol import canonical_state, initial_state  # noqa: E402


def _state(operation: str, width: int, position: int, carry: int, result: str) -> str:
    base = initial_state(operation, 12, 3, width)
    base.update({"p": position, "c": carry, "r": result, "z": int(position == width)})
    return canonical_state(base)


def _branch(branch_id: str, split: str = "fit_w4") -> dict:
    width = evaluator.REGIME_WIDTHS[split]
    initial = initial_state("add", 12, 3, width)
    states = []
    result = ["0"] * width
    for position in range(width):
        result[position] = "5" if position == 0 else "1" if position == 1 else "0"
        state = dict(initial)
        state.update(
            {
                "p": position + 1,
                "c": 0,
                "r": "".join(result),
                "z": int(position + 1 == width),
            }
        )
        states.append(canonical_state(state))
    return {
        "id": branch_id,
        "split": split,
        "prompt_style": "heldout",
        "operation": "add",
        "width": width,
        "left": 12,
        "right": 3,
        "initial_state": canonical_state(initial),
        "expected_states": states,
        "expected_answer": 15,
    }


def _generation(text: str) -> evaluator.Generation:
    return evaluator.Generation(text, (1,), (1,), 20, "eos")


def _rollout_result(
    branch_id: str,
    *,
    regime: str,
    operation: str,
    width: int,
    carry_class: str,
    success: bool,
    failure_position: int | None = None,
) -> dict:
    rows = []
    for position in range(width):
        correct = failure_position is None or position < failure_position
        rows.append(
            {
                "position": position,
                "correct": correct,
                "predicted_state": {} if correct else None,
            }
        )
        if not correct:
            break
    prefix = width if failure_position is None else failure_position
    return {
        "id": branch_id,
        "regime": regime,
        "operation": operation,
        "width": width,
        "terminal_carry_class": carry_class,
        "expected_answer": 1,
        "transition_budget": width,
        "transition_calls": len(rows),
        "prefix_exact_length": prefix,
        "fully_parseable": failure_position is None,
        "state_closed_loop_exact": failure_position is None,
        "terminal_transition_exact": failure_position is None,
        "terminal_reached": failure_position is None,
        "final_prompt_issued": failure_position is None,
        "emitted_answer": 1 if success else None,
        "final_answer_correct": success,
        "closed_loop_success": success,
        "first_failure_position": failure_position,
        "first_failure_reason": None if failure_position is None else "malformed_state",
        "emitted_token_count": len(rows),
        "rows": rows,
    }


def _pair(
    index: int, regime: str, success: bool, failure_position: int | None = None
) -> dict:
    width = evaluator.REGIME_WIDTHS[regime]
    normal = _rollout_result(
        f"pair-{index}",
        regime=regime,
        operation="add",
        width=width,
        carry_class="00",
        success=success,
        failure_position=failure_position,
    )
    counterfactual = _rollout_result(
        f"pair-{index}-cf",
        regime=regime,
        operation="add",
        width=width,
        carry_class="01",
        success=success,
        failure_position=failure_position,
    )
    return {
        "id": f"pair-{index}",
        "regime": regime,
        "operation": "add",
        "width": width,
        "normal_terminal_carry_class": "00",
        "counterfactual_terminal_carry_class": "01",
        "expected_answer_changed": True,
        "first_expected_state_divergence_position": 0,
        "first_predicted_state_divergence_position": 0 if success else None,
        "state_intervention_at_expected_position": success,
        "both_state_closed_loop_exact": success,
        "both_final_answers_correct": success,
        "answer_intervention_success": success,
        "both_closed_loop_success": success,
        "normal": normal,
        "counterfactual": counterfactual,
    }


def test_wrong_parseable_state_is_forwarded_without_solver_repair() -> None:
    branch = _branch("fit_w4-00000")
    parseable_wrong = dict(parse_state(branch["expected_states"][0]))
    parseable_wrong["r"] = "4" + parseable_wrong["r"][1:]
    wrong_line = canonical_state(parseable_wrong)
    prompts: list[str] = []

    def ask(prompt: str, _max_new: int, _kind: str) -> evaluator.Generation:
        prompts.append(prompt)
        return _generation(wrong_line if len(prompts) == 1 else "not a state")

    result = evaluator.rollout_branch(branch, ask)
    assert len(prompts) == 2
    assert wrong_line in prompts[1]
    assert result["first_failure_position"] == 0
    assert result["first_failure_reason"] == "state_mismatch"
    assert result["transition_calls"] == 2
    assert result["final_prompt_issued"] is False


def test_multiple_answer_lines_are_malformed_not_last_answer_wins() -> None:
    branch = _branch("fit_w4-00000")
    generations = iter(
        [_generation(state) for state in branch["expected_states"]]
        + [_generation("answer=999\nanswer=15")]
    )

    def ask(_prompt: str, _max_new: int, _kind: str) -> evaluator.Generation:
        return next(generations)

    result = evaluator.rollout_branch(branch, ask)
    assert result["state_closed_loop_exact"] is True
    assert result["emitted_answer"] is None
    assert result["final_answer_correct"] is False
    assert result["closed_loop_success"] is False
    assert result["first_failure_reason"] == "malformed_answer"


def test_intervention_credit_requires_both_states_correct(monkeypatch) -> None:
    normal_branch = _branch("fit_w4-00000")
    counterfactual_branch = _branch("fit_w4-00000-cf")
    counterfactual_branch["expected_states"] = list(
        counterfactual_branch["expected_states"]
    )
    counterfactual_branch["expected_states"][0] += ":counterfactual"
    episode = {**normal_branch, "counterfactual": counterfactual_branch}
    normal = _rollout_result(
        "normal",
        regime="fit_w4",
        operation="add",
        width=4,
        carry_class="00",
        success=False,
        failure_position=0,
    )
    counterfactual = _rollout_result(
        "counterfactual",
        regime="fit_w4",
        operation="add",
        width=4,
        carry_class="01",
        success=False,
        failure_position=0,
    )
    normal["rows"][0].update({"predicted_state": {"c": 0}, "correct": False})
    counterfactual["rows"][0].update({"predicted_state": {"c": 1}, "correct": False})
    results = iter((normal, counterfactual))
    monkeypatch.setattr(
        evaluator, "rollout_branch", lambda _branch, _ask: next(results)
    )
    scored = evaluator.evaluate_pair(episode, lambda *_args: _generation(""))
    assert scored["prediction_diverged_at_expected_position"] is True
    assert scored["state_intervention_at_expected_position"] is False


def test_accounting_includes_all_groups_and_width_8_survival(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluator,
        "REGIME_BUDGETS",
        {"fit_w4": 1, "width_ood_w8": 1},
    )
    monkeypatch.setattr(
        evaluator,
        "REGIME_WIDTHS",
        {"fit_w4": 4, "width_ood_w8": 8},
    )
    monkeypatch.setattr(evaluator, "PAIR_COUNT", 2)
    monkeypatch.setattr(evaluator, "BRANCH_COUNT", 4)
    records = [
        _pair(0, "fit_w4", True),
        _pair(1, "width_ood_w8", False, failure_position=3),
    ]
    metrics = evaluator.build_metrics(records)
    accounting = evaluator.validate_accounting(records, metrics)
    overall = metrics["branches"]["overall"]["counts"]
    assert accounting["pairs"] == 2
    assert accounting["branches"] == 4
    assert accounting["transition_budget"] == 24
    assert overall["closed_loop_success"] == 2
    assert overall["exact_transitions"] == 14
    assert set(metrics["branches"]["by_regime"]) == {"fit_w4", "width_ood_w8"}
    assert set(metrics["branches"]["by_terminal_carry_class"]) == {
        "add:00",
        "add:01",
    }
    survival = metrics["branches"]["width_8_survival"]["positions"]
    assert survival[2]["exact_prefix_survived"] == 2
    assert survival[3]["exact_prefix_survived"] == 0


def test_read_frozen_file_rejects_corruption_and_symlink(tmp_path: Path) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"frozen")
    digest = hashlib.sha256(b"frozen").hexdigest()
    captured = evaluator.read_frozen_file(path, digest)
    assert captured.payload == b"frozen"
    path.write_bytes(b"mutated")
    with pytest.raises(evaluator.ContractError, match="SHA-256 mismatch"):
        evaluator.read_frozen_file(path, digest)
    link = tmp_path / "link.bin"
    link.symlink_to(path)
    with pytest.raises(evaluator.ContractError, match="cannot open"):
        evaluator.read_frozen_file(link, hashlib.sha256(b"mutated").hexdigest())


def test_source_binding_rejects_nonexecuting_root_and_forged_commit(
    tmp_path: Path, monkeypatch
) -> None:
    empty_tree_payload = b""
    empty_tree_id = evaluator.git_object_id("tree", empty_tree_payload, 40)
    commit_payload = (
        b"tree " + empty_tree_id.encode("ascii") + b"\n\nreviewed evaluator\n"
    )
    commit_id = evaluator.git_object_id("commit", commit_payload, 40)
    contract = evaluator.ARM_CONTRACTS["iid"]
    binding = {
        "schema": "shohin-digitwise-factorial-v4-eval-source-binding-v2",
        "eval_commit": commit_id,
        "git_commit_object_b64": base64.b64encode(commit_payload).decode("ascii"),
        "git_tree_objects_b64": {
            empty_tree_id: base64.b64encode(empty_tree_payload).decode("ascii")
        },
        "reviewed_clean_checkout": True,
        "executing_source_root": str(tmp_path.resolve()),
        "sources": {
            path: {
                "sha256": "1" * 64,
                "git_blob_id": "1" * 40,
                "git_mode": "100644",
            }
            for path in evaluator.EVAL_SOURCE_PATHS
        },
        "spooled_wrapper_sha256": "1" * 64,
        "slurm": {"job_id": 1, "array_task_id": 0, "restart_count": 0},
        "scientific_inputs": {
            "arm": contract.arm,
            "checkpoint_origin": contract.checkpoint_origin,
            "checkpoint_sha256": contract.checkpoint_sha256,
            "heldout_sha256": evaluator.HELDOUT_SHA256,
            "tokenizer_sha256": evaluator.TOKENIZER_SHA256,
        },
    }
    binding["sources"]["train/jobs/eval_digitwise_factorial_v4.sbatch"]["sha256"] = (
        binding["spooled_wrapper_sha256"]
    )
    payload = evaluator.canonical_json_bytes(binding)
    frozen = evaluator.FrozenFile(
        path=tmp_path / "binding.json",
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        mode=0o400,
    )
    with pytest.raises(evaluator.ContractError, match="executing source tree"):
        evaluator.validate_source_binding(frozen, tmp_path, contract)
    binding["eval_commit"] = "0" * 40
    payload = evaluator.canonical_json_bytes(binding)
    frozen = evaluator.FrozenFile(
        path=tmp_path / "binding.json",
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        mode=0o400,
    )
    with pytest.raises(evaluator.ContractError, match="commit object"):
        evaluator.validate_source_binding(frozen, tmp_path, contract)
    binding["eval_commit"] = commit_id
    binding["executing_source_root"] = str(tmp_path.resolve())
    payload = evaluator.canonical_json_bytes(binding)
    frozen = evaluator.FrozenFile(
        path=tmp_path / "binding.json",
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        mode=0o400,
    )
    monkeypatch.setattr(evaluator, "ROOT", tmp_path.resolve())
    with pytest.raises(evaluator.ContractError, match="absent from commit tree"):
        evaluator.validate_source_binding(frozen, tmp_path, contract)


def test_source_binding_accepts_exact_commit_tree_blob_chain(
    tmp_path: Path, monkeypatch
) -> None:
    payloads = {
        path: (path + "\n").encode("ascii") for path in evaluator.EVAL_SOURCE_PATHS
    }
    for relative, payload in payloads.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        target.chmod(0o400)

    blob_ids = {
        path: evaluator.git_object_id("blob", payload, 40)
        for path, payload in payloads.items()
    }

    def tree_payload(entries: list[tuple[str, str, str]]) -> bytes:
        return b"".join(
            mode.encode("ascii")
            + b" "
            + name.encode("utf-8")
            + b"\0"
            + bytes.fromhex(object_id)
            for mode, name, object_id in entries
        )

    jobs_payload = tree_payload(
        [
            (
                "100644",
                "eval_digitwise_factorial_v4.sbatch",
                blob_ids["train/jobs/eval_digitwise_factorial_v4.sbatch"],
            )
        ]
    )
    jobs_id = evaluator.git_object_id("tree", jobs_payload, 40)
    train_entries = [
        ("100644", Path(path).name, blob_ids[path])
        for path in evaluator.EVAL_SOURCE_PATHS
        if not path.startswith("train/jobs/")
    ] + [("40000", "jobs", jobs_id)]
    train_payload = tree_payload(sorted(train_entries, key=lambda item: item[1]))
    train_id = evaluator.git_object_id("tree", train_payload, 40)
    root_payload = tree_payload([("40000", "train", train_id)])
    root_id = evaluator.git_object_id("tree", root_payload, 40)
    commit_payload = b"tree " + root_id.encode("ascii") + b"\n\nsynthetic proof\n"
    commit_id = evaluator.git_object_id("commit", commit_payload, 40)
    contract = evaluator.ARM_CONTRACTS["iid"]
    sources = {
        path: {
            "sha256": hashlib.sha256(payloads[path]).hexdigest(),
            "git_blob_id": blob_ids[path],
            "git_mode": "100644",
        }
        for path in evaluator.EVAL_SOURCE_PATHS
    }
    wrapper_sha256 = sources["train/jobs/eval_digitwise_factorial_v4.sbatch"]["sha256"]
    binding = {
        "schema": "shohin-digitwise-factorial-v4-eval-source-binding-v2",
        "eval_commit": commit_id,
        "git_commit_object_b64": base64.b64encode(commit_payload).decode("ascii"),
        "git_tree_objects_b64": {
            object_id: base64.b64encode(payload).decode("ascii")
            for object_id, payload in {
                root_id: root_payload,
                train_id: train_payload,
                jobs_id: jobs_payload,
            }.items()
        },
        "reviewed_clean_checkout": True,
        "executing_source_root": str(tmp_path.resolve()),
        "sources": sources,
        "spooled_wrapper_sha256": wrapper_sha256,
        "slurm": {"job_id": 1, "array_task_id": 0, "restart_count": 0},
        "scientific_inputs": {
            "arm": contract.arm,
            "checkpoint_origin": contract.checkpoint_origin,
            "checkpoint_sha256": contract.checkpoint_sha256,
            "heldout_sha256": evaluator.HELDOUT_SHA256,
            "tokenizer_sha256": evaluator.TOKENIZER_SHA256,
        },
    }
    payload = evaluator.canonical_json_bytes(binding)
    frozen = evaluator.FrozenFile(
        path=tmp_path / "binding.json",
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        mode=0o400,
    )
    monkeypatch.setattr(evaluator, "ROOT", tmp_path.resolve())
    assert evaluator.validate_source_binding(frozen, tmp_path, contract) == binding


def test_checkpoint_metadata_rejects_wrong_arm_and_preflight() -> None:
    contract = evaluator.ARM_CONTRACTS["iid"]
    checkpoint = {
        "cfg": {},
        "model": {},
        "step": "sft_ep1",
        "factorial_arm": "iid",
        "production_admission_sha256": contract.admission_sha256,
        "exact_budget_preflight_sha256": contract.preflight_sha256,
        "exact_budget_updates": evaluator.EXACT_UPDATES,
    }
    evaluator.validate_checkpoint_metadata(checkpoint, contract)
    corrupt = dict(checkpoint, factorial_arm="term")
    with pytest.raises(evaluator.ContractError, match="metadata binding"):
        evaluator.validate_checkpoint_metadata(corrupt, contract)
    corrupt = dict(checkpoint, exact_budget_preflight_sha256="0" * 64)
    with pytest.raises(evaluator.ContractError, match="metadata binding"):
        evaluator.validate_checkpoint_metadata(corrupt, contract)


def test_validate_heldout_rows_rejects_pair_corruption() -> None:
    normal = _branch("fit_w4-00000")
    counterfactual = _branch("fit_w4-00000-cf")
    row = {**normal, "counterfactual": counterfactual}
    validated = evaluator.validate_heldout_rows(
        [row], expected_regime_counts={"fit_w4": 1}
    )
    assert validated[0]["id"] == "fit_w4-00000"
    corrupt = json.loads(json.dumps(row))
    corrupt["counterfactual"]["operation"] = "sub"
    with pytest.raises(evaluator.ContractError):
        evaluator.validate_heldout_rows([corrupt], expected_regime_counts={"fit_w4": 1})
    corrupt = json.loads(json.dumps(row))
    corrupt["expected_states"] = corrupt["expected_states"][:-1]
    with pytest.raises(evaluator.ContractError, match="wrong transition budget"):
        evaluator.validate_heldout_rows([corrupt], expected_regime_counts={"fit_w4": 1})


def test_exact_frozen_heldout_board_passes_full_contract() -> None:
    path = ROOT / "artifacts" / "evals" / "digitwise_recurrent_v2_heldout.jsonl"
    frozen = evaluator.read_frozen_file(path, evaluator.HELDOUT_SHA256)
    rows = evaluator.load_heldout(frozen)
    assert len(rows) == evaluator.PAIR_COUNT == 1_500
    assert {
        regime: sum(row["split"] == regime for row in rows)
        for regime in evaluator.REGIME_BUDGETS
    } == evaluator.REGIME_BUDGETS
    assert all(
        evaluator.terminal_class(row) in {"00", "10", "01", "11"}
        and evaluator.terminal_class(row["counterfactual"]) in {"00", "10", "01", "11"}
        for row in rows
    )


def test_one_purpose_publication_is_canonical_sealed_and_no_replace(
    tmp_path: Path,
) -> None:
    output = tmp_path / "arm"
    target, digest = evaluator.publish_one_report(output, {"z": 1, "a": [2]})
    expected = b'{"a":[2],"z":1}\n'
    assert target.read_bytes() == expected
    assert digest == hashlib.sha256(expected).hexdigest()
    assert stat.S_IMODE(output.stat().st_mode) == 0o500
    assert stat.S_IMODE(target.stat().st_mode) == 0o400
    with pytest.raises(evaluator.ContractError, match="refusing existing"):
        evaluator.publish_one_report(output, {"different": True})


def test_publication_failure_leaves_no_canonical_directory(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "arm"

    def fail_before_publish(_source: Path, _target: Path) -> None:
        raise RuntimeError("simulated publication interruption")

    monkeypatch.setattr(evaluator, "_rename_directory_no_replace", fail_before_publish)
    with pytest.raises(RuntimeError, match="simulated publication interruption"):
        evaluator.publish_one_report(output, {"a": 1})
    assert not output.exists()
    assert list(tmp_path.glob(".arm.staging.*")) == []


# Imported late to keep the helper definitions visually separate from the test contract.
from digitwise_protocol import parse_state  # noqa: E402
