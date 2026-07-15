from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

import score_residual_packet_v1_independent as scorer


class ByteCodec:
    eos_id = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("ascii"))

    def decode(self, token_ids) -> str:
        return bytes(token_ids).decode("ascii")


CODEC = ByteCodec()


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Recorder:
    def __init__(self):
        self.index = 0
        self.records = []
        self.unissued = defaultdict(int)

    def call(self, model, arm, prompt, response, max_new):
        decoded = CODEC.encode(response)
        value = {
            "call_index": self.index,
            "model": model,
            "arm": arm,
            "prompt": prompt,
            "max_new": max_new,
            "response": response,
            "prompt_token_count": len(CODEC.encode(prompt)),
            "sampled_token_ids": decoded + [CODEC.eos_id],
            "sampled_token_count": len(decoded) + 1,
            "decoded_token_ids": decoded,
            "decoded_token_count": len(decoded),
            "stop_reason": "eos",
        }
        self.index += 1
        self.records.append(value)
        return value

    def note(self, model, arm, count):
        self.unissued[(model, arm)] += count

    def ledger(self, resources):
        result = {"by_model": {}}
        for model in scorer.MODEL_NAMES:
            records = [record for record in self.records if record["model"] == model]
            arms = sorted(
                {record["arm"] for record in records}
                | {arm for (name, arm), count in self.unissued.items() if name == model and count >= 0}
            )
            by_arm = {}
            for arm in arms:
                selected = [record for record in records if record["arm"] == arm]
                by_arm[arm] = {
                    "model_calls": len(selected),
                    "prompt_tokens": sum(record["prompt_token_count"] for record in selected),
                    "sampled_tokens": sum(record["sampled_token_count"] for record in selected),
                    "decoded_tokens": sum(record["decoded_token_count"] for record in selected),
                    "supervised_completion_tokens": 0,
                    "packed_forward_token_positions": 0,
                    "calls_not_issued_after_parse_failure": self.unissued[(model, arm)],
                    "retries": 0,
                    "repairs": 0,
                    "searches": 0,
                    "verifier_feedback_calls": 0,
                }
            training = resources.get(model, {})
            by_arm["training"] = {
                "model_calls": 0,
                "prompt_tokens": 0,
                "sampled_tokens": 0,
                "decoded_tokens": 0,
                "supervised_completion_tokens": training.get("supervised_completion_tokens", 0),
                "packed_forward_token_positions": training.get("packed_forward_token_positions", 0),
                "calls_not_issued_after_parse_failure": 0,
                "retries": 0,
                "repairs": 0,
                "searches": 0,
                "verifier_feedback_calls": 0,
            }
            result["by_model"][model] = {"by_arm": by_arm}
        return result


def _ops(*items):
    return [[kind, value] for kind, value in items]


def _trajectory(initial, operations):
    states = [initial]
    for kind, operand in operations:
        if kind == "add":
            states.append(states[-1] + operand)
        elif kind == "multiply":
            states.append(states[-1] * operand)
        else:
            states.append(states[-1] - operand)
    return states


def _board():
    specs = [
        ("mini_000", 10, _ops(("add", 3), ("multiply", 2))),
        ("mini_001", 21, _ops(("subtract", 4), ("add", 10))),
    ]
    rows = []
    for identifier, initial, operations in specs:
        parsed = tuple((kind, value) for kind, value in operations)
        states = _trajectory(initial, parsed)
        rows.append(
            {
                "answer": states[-1],
                "id": identifier,
                "initial_state": initial,
                "operations": operations,
                "packet": scorer.format_packet(initial, parsed),
                "source": f"Mini source {identifier}.",
                "stratum": "mini",
                "template_id": "mini_template",
                "trajectory": states,
            }
        )
    rows_sha = _sha(_canonical(rows))
    return {
        "case_count": 2,
        "per_stratum": 2,
        "protocol_sha256": None,
        "rows": rows,
        "rows_sha256": rows_sha,
        "schema": scorer.BOARD_SCHEMA,
        "seed": 7,
        "stratum_order": ["mini"],
    }


def _row_parts(row):
    operations = tuple((kind, value) for kind, value in row["operations"])
    return row["initial_state"], operations


def _runtime(recorder, model, arm, packet_text):
    parsed = scorer.parse_packet(packet_text)
    if parsed is None:
        return {"termination": "initial_packet_invalid", "steps": []}
    state, operations = parsed
    steps = []
    packet = scorer.format_packet(state, operations)
    for index, operation in enumerate(operations):
        observed = scorer.apply_operation(state, operation)
        executor = recorder.call(
            "raw_260k_executor",
            arm,
            scorer.executor_prompt(state, operation),
            str(observed),
            scorer.MAX_EXECUTOR_TOKENS,
        )
        if index == len(operations) - 1:
            response = scorer.format_answer(observed)
        else:
            response = scorer.format_packet(observed, operations[index + 1 :])
        updater = recorder.call(
            model,
            arm,
            scorer.update_prompt(packet, observed),
            response,
            scorer.MAX_CONTROLLER_TOKENS,
        )
        steps.append({"executor": executor, "updater": updater})
        state = observed
        if index + 1 < len(operations):
            packet = scorer.format_packet(state, operations[index + 1 :])
    return {"termination": "answer", "steps": steps}


def _external(recorder, row):
    state, operations = _row_parts(row)
    calls = []
    for operation in operations:
        observed = scorer.apply_operation(state, operation)
        calls.append(
            recorder.call(
                "raw_260k_executor",
                "external_scheduler",
                scorer.executor_prompt(state, operation),
                str(observed),
                scorer.MAX_EXECUTOR_TOKENS,
            )
        )
        state = observed
    return {"termination": "complete", "steps": calls}


def _teacher_cases(board):
    rows = [
        scorer.BoardRow(
            identifier=row["id"],
            stratum=row["stratum"],
            source=row["source"],
            initial=row["initial_state"],
            operations=tuple((kind, value) for kind, value in row["operations"]),
            packet=row["packet"],
            trajectory=tuple(row["trajectory"]),
            answer=row["answer"],
        )
        for row in board["rows"]
    ]
    return scorer.expected_teacher_cases(rows)


def _transcript(board, hashes, resources):
    recorder = Recorder()
    external = [
        {"id": row["id"], "runtime": _external(recorder, row)} for row in board["rows"]
    ]
    controllers = {}
    teachers = _teacher_cases(board)
    rows = board["rows"]
    for model in ("treatment", "sham"):
        strict = []
        oracle = []
        # Calls are deliberately issued strict/oracle interleaved like acquisition.
        for row in rows:
            initial, operations = _row_parts(row)
            if model == "treatment":
                response = row["packet"]
            else:
                response = "not a packet"
            compiler = recorder.call(
                model,
                "strict_closed_loop",
                scorer.compiler_prompt(row["source"]),
                response,
                scorer.MAX_CONTROLLER_TOKENS,
            )
            runtime = _runtime(recorder, model, "strict_closed_loop", response)
            if model == "sham":
                recorder.note("raw_260k_executor", "strict_closed_loop", len(operations))
                recorder.note(model, "strict_closed_loop", len(operations))
            strict.append({"id": row["id"], "compiler": compiler, "runtime": runtime})
            oracle.append(
                {
                    "id": row["id"],
                    "runtime": _runtime(recorder, model, "oracle_packet_loop", row["packet"]),
                }
            )
        teacher = []
        for row_obj, step_index, packet, observed, expected in teachers:
            call = recorder.call(
                model,
                "teacher_forced_updater",
                scorer.update_prompt(packet, observed),
                expected,
                scorer.MAX_CONTROLLER_TOKENS,
            )
            teacher.append(
                {
                    "id": row_obj.identifier,
                    "step_index": step_index,
                    "packet": packet,
                    "observed": observed,
                    "call": call,
                }
            )
        packet_swaps = []
        for index, original in enumerate(rows):
            donor = rows[(index + 1) % len(rows)]
            compiler = recorder.call(
                model,
                "packet_swap",
                scorer.compiler_prompt(original["source"]),
                original["packet"] if model == "treatment" else "not a packet",
                scorer.MAX_CONTROLLER_TOKENS,
            )
            packet_swaps.append(
                {
                    "original_id": original["id"],
                    "donor_id": donor["id"],
                    "compiler": compiler,
                    "intervened_packet": donor["packet"],
                    "runtime": _runtime(recorder, model, "packet_swap", donor["packet"]),
                }
            )
        controllers[model] = {
            "strict_closed_loop": strict,
            "oracle_packet_loop": oracle,
            "teacher_forced_updater": teacher,
            "packet_swaps": packet_swaps,
        }
    transcript = {
        "schema": scorer.TRANSCRIPT_SCHEMA,
        "seed": 11,
        "protocol_module": "residual_packet_protocol",
        "input_hashes": {
            "board": hashes["board"],
            "tokenizer": hashes["tokenizer"],
            "raw_260k_executor": hashes["executor_checkpoint"],
            "treatment_checkpoint": hashes["treatment_checkpoint"],
            "sham_checkpoint": hashes["sham_checkpoint"],
            "protocol": hashes["protocol"],
            "evaluator": hashes["evaluator"],
            "training_resources": hashes["training_resources"],
        },
        "decode_caps": {"controller": 80, "executor": 48, "maximum_transitions": 5},
        "models": {
            "treatment": {
                "checkpoint_path": "/frozen/treatment.pt",
                "checkpoint_sha256": hashes["treatment_checkpoint"],
                "checkpoint_step": 260000,
            },
            "sham": {
                "checkpoint_path": "/frozen/sham.pt",
                "checkpoint_sha256": hashes["sham_checkpoint"],
                "checkpoint_step": 260000,
            },
            "raw_260k_executor": {
                "checkpoint_path": "/frozen/raw.pt",
                "checkpoint_sha256": hashes["executor_checkpoint"],
                "checkpoint_step": 260000,
            },
        },
        "external_scheduler": external,
        "controllers": controllers,
        "resource_ledger": recorder.ledger(
            {"treatment": resources["treatment"], "sham": resources["sham"]}
        ),
        "call_count": recorder.index,
    }
    return transcript


def _write(path, raw, immutable=False):
    path.write_bytes(raw)
    if immutable:
        path.chmod(0o444)


def build_fixture(tmp_path):
    board = _board()
    raw = {}
    raw["prereg"] = b"frozen prereg\n"
    raw["prerequisite_confirmation"] = _json_bytes(
        {"advance_to_internalization": True}
    )
    raw["protocol"] = b"independent protocol source\n"
    raw["sft_encoding"] = b"independent SFT encoding source\n"
    raw["evaluator"] = b"independent evaluator source\n"
    raw["tokenizer"] = b"toy tokenizer\n"
    raw["executor_checkpoint"] = b"raw checkpoint\n"
    raw["treatment_data"] = b'{"arm":"treatment"}\n'
    raw["sham_data"] = b'{"arm":"sham"}\n'
    raw["treatment_checkpoint"] = b"treatment checkpoint\n"
    raw["sham_checkpoint"] = b"sham checkpoint\n"
    hashes = {label: _sha(value) for label, value in raw.items()}
    board["protocol_sha256"] = hashes["protocol"]
    raw["board"] = _json_bytes(board)
    hashes["board"] = _sha(raw["board"])
    resources = {
        "paired_seed": 11,
        "treatment": {
            "supervised_completion_tokens": 123,
            "packed_forward_token_positions": 456,
        },
        "sham": {
            "supervised_completion_tokens": 123,
            "packed_forward_token_positions": 456,
        },
    }
    raw["training_resources"] = _json_bytes(resources)
    hashes["training_resources"] = _sha(raw["training_resources"])
    manifest = {
        "artifacts": {
            "board_sha256": hashes["board"],
            "board_rows_sha256": board["rows_sha256"],
            "treatment_sha256": hashes["treatment_data"],
            "sham_sha256": hashes["sham_data"],
        },
        "tokenizer_sha256": hashes["tokenizer"],
        "protocol_sha256": hashes["protocol"],
        "sft_encoding_sha256": hashes["sft_encoding"],
    }
    raw["training_manifest"] = _json_bytes(manifest)
    hashes["training_manifest"] = _sha(raw["training_manifest"])
    audit = {
        "admitted": True,
        "failures": [],
        "artifact_sha256": {
            "board": hashes["board"],
            "manifest": hashes["training_manifest"],
            "protocol": hashes["protocol"],
            "sft_encoding": hashes["sft_encoding"],
            "sham": hashes["sham_data"],
            "treatment": hashes["treatment_data"],
            "tokenizer": hashes["tokenizer"],
        },
    }
    raw["audit"] = _json_bytes(audit)
    hashes["audit"] = _sha(raw["audit"])
    controller_bound = [
        hashes[label]
        for label in (
            "treatment_checkpoint",
            "sham_checkpoint",
            "tokenizer",
            "training_resources",
            "board",
            "audit",
            "treatment_data",
            "sham_data",
            "training_manifest",
            "sft_encoding",
        )
    ]
    raw["controller_manifest"] = _json_bytes({"bindings": controller_bound, "seed": 11})
    raw["executor_manifest"] = _json_bytes(
        {"bindings": [hashes["executor_checkpoint"], hashes["tokenizer"]]}
    )
    hashes["controller_manifest"] = _sha(raw["controller_manifest"])
    hashes["executor_manifest"] = _sha(raw["executor_manifest"])
    transcript = _transcript(board, hashes, resources)
    raw["transcript"] = _json_bytes(transcript)
    hashes["transcript"] = _sha(raw["transcript"])
    paths = {}
    immutable = {
        "board",
        "treatment_data",
        "sham_data",
        "training_manifest",
        "audit",
        "training_resources",
        "controller_manifest",
        "executor_manifest",
        "transcript",
    }
    for label in scorer.REQUIRED_ARTIFACTS:
        path = tmp_path / label
        _write(path, raw[label], label in immutable)
        paths[label] = path
    bindings = scorer.FrozenBindings(seed=11, sha256=hashes, board_rows_sha256=board["rows_sha256"])
    contract = scorer.ScoreContract(
        board_seed=7,
        fit_seeds=(11,),
        strata=("mini",),
        case_count=2,
        per_stratum=2,
        swaps_per_stratum=2,
        external_gold_min=2,
        oracle_min=2,
        compile_min=2,
        strict_min=2,
        per_stratum_compile_min=2,
        per_stratum_strict_min=2,
        external_mismatch_max=0,
        swap_min=2,
        update_min=Fraction(1, 1),
        compile_delta_min=Fraction(1, 1),
        strict_delta_min=Fraction(1, 1),
        mcnemar_max=Fraction(1, 1),
    )
    return paths, bindings, contract, board, transcript


def _rewrite_transcript(paths, bindings, transcript):
    path = paths["transcript"]
    path.chmod(0o644)
    raw = _json_bytes(transcript)
    path.write_bytes(raw)
    path.chmod(0o444)
    hashes = dict(bindings.sha256)
    hashes["transcript"] = _sha(raw)
    return replace(bindings, sha256=hashes)


def test_happy_fixture_is_recomputed_without_trusted_scores(tmp_path):
    paths, bindings, contract, _, _ = build_fixture(tmp_path)
    result = scorer.score_artifacts(
        paths, bindings=bindings, contract=contract, codec=CODEC
    )
    assert result["external_scheduler_gold_answer"]["correct"] == 2
    assert result["external_scheduler_gold_trajectory"]["correct"] == 2
    assert result["external_scheduler_atomic_transition"]["correct"] == 4
    assert result["arms"]["treatment"]["compile_exact"]["correct"] == 2
    assert result["arms"]["treatment"]["strict_closed_loop"]["correct"] == 2
    assert result["arms"]["treatment"]["gold_trajectory"]["correct"] == 2
    assert result["arms"]["sham"]["compile_exact"]["correct"] == 0
    assert result["arms"]["sham"]["oracle_packet_loop"]["correct"] == 2
    assert result["arms"]["treatment"]["conditional_update_exact"]["rate"] == 1.0
    assert result["arms"]["treatment"]["packet_swap_follows_donor"]["correct"] == 2
    assert result["paired"]["compile_mcnemar"]["p_value_exact"] == "1/2"
    assert result["gates"]["all_pass"] is True


def test_production_bindings_fail_until_post_gate_hashes_are_frozen():
    binding = scorer.PRODUCTION_BINDINGS[2026071511]
    assert binding.sha256["board"] == "ad6be48f5952a142c0684f304ba6393b66c25b68b2d6c97d8a0b5d80cfedd9e7"
    assert binding.board_rows_sha256 == "fcc2970f9bbd8890a6e3d8cb495ddb45cb7c0825d9adb7318d1b2e0807b9a20e"
    assert binding.sha256["protocol"] == "e011e8389d51188553d9fb0392ec892a5107249cc23c82cdba4df216e6db2ce2"
    assert binding.sha256["sft_encoding"] == "33d07db19afb7155567cef922dbab616916864eafaeba7fa26f12d2203c0d5b8"
    with pytest.raises(scorer.IntegrityError, match="has not been frozen"):
        binding.validate()


def test_production_sources_are_loaded_only_from_exact_paths(tmp_path):
    source_dir = Path(scorer.__file__).resolve().parent
    exact = {
        "protocol": source_dir / "residual_packet_protocol.py",
        "sft_encoding": source_dir / "sft_encoding.py",
        "evaluator": source_dir / "eval_residual_packet_v1.py",
    }
    scorer.validate_production_source_paths(exact)
    wrong = dict(exact)
    wrong["protocol"] = tmp_path / "residual_packet_protocol.py"
    wrong["protocol"].write_text("alternate")
    with pytest.raises(scorer.IntegrityError, match="exact admitted source path"):
        scorer.validate_production_source_paths(wrong)


def test_board_arithmetic_is_replayed_even_under_matching_fixture_hash(tmp_path):
    paths, bindings, contract, board, _ = build_fixture(tmp_path)
    board["rows"][0]["answer"] += 1
    board["rows_sha256"] = _sha(_canonical(board["rows"]))
    tampered = replace(bindings, board_rows_sha256=board["rows_sha256"])
    with pytest.raises(scorer.IntegrityError, match="answer replay mismatch"):
        scorer.validate_board(board, tampered, contract)


def test_transcript_boolean_and_claim_field_are_rejected(tmp_path):
    paths, bindings, contract, _, transcript = build_fixture(tmp_path)
    transcript["controllers"]["treatment"]["strict_closed_loop"][0]["compiler"]["correct"] = True
    bindings = _rewrite_transcript(paths, bindings, transcript)
    with pytest.raises(scorer.IntegrityError, match="contains a boolean"):
        scorer.score_artifacts(paths, bindings=bindings, contract=contract, codec=CODEC)


def test_ledger_tamper_is_rejected(tmp_path):
    paths, bindings, contract, _, transcript = build_fixture(tmp_path)
    transcript["resource_ledger"]["by_model"]["treatment"]["by_arm"]["training"][
        "supervised_completion_tokens"
    ] += 1
    bindings = _rewrite_transcript(paths, bindings, transcript)
    with pytest.raises(scorer.IntegrityError, match="resource ledger reconstruction mismatch"):
        scorer.score_artifacts(paths, bindings=bindings, contract=contract, codec=CODEC)


def test_call_order_tamper_is_rejected(tmp_path):
    paths, bindings, contract, _, transcript = build_fixture(tmp_path)
    first = transcript["external_scheduler"][0]["runtime"]["steps"][0]
    second = transcript["external_scheduler"][0]["runtime"]["steps"][1]
    first["call_index"], second["call_index"] = second["call_index"], first["call_index"]
    bindings = _rewrite_transcript(paths, bindings, transcript)
    with pytest.raises(scorer.IntegrityError, match="call indices do not match issue order"):
        scorer.score_artifacts(paths, bindings=bindings, contract=contract, codec=CODEC)


def test_prompt_source_leak_or_change_is_rejected(tmp_path):
    paths, bindings, contract, _, transcript = build_fixture(tmp_path)
    updater = transcript["controllers"]["treatment"]["strict_closed_loop"][0]["runtime"][
        "steps"
    ][0]["updater"]
    updater["prompt"] += "\nOriginal source: leaked"
    updater["prompt_token_count"] = len(CODEC.encode(updater["prompt"]))
    bindings = _rewrite_transcript(paths, bindings, transcript)
    with pytest.raises(scorer.IntegrityError, match="prompt mismatch"):
        scorer.score_artifacts(paths, bindings=bindings, contract=contract, codec=CODEC)


def test_trusted_termination_is_recomputed(tmp_path):
    paths, bindings, contract, _, transcript = build_fixture(tmp_path)
    transcript["controllers"]["treatment"]["strict_closed_loop"][0]["runtime"][
        "termination"
    ] = "updater_output_invalid"
    bindings = _rewrite_transcript(paths, bindings, transcript)
    with pytest.raises(scorer.IntegrityError, match="trusted termination mismatch"):
        scorer.score_artifacts(paths, bindings=bindings, contract=contract, codec=CODEC)


def test_exact_parsers_reject_forgiving_forms():
    assert scorer.parse_packet("State: 01\nPlan: add 2") is None
    assert scorer.parse_packet("State: 1\nPlan: plus 2") is None
    assert scorer.parse_packet("State: 1\nPlan: add 2\nComment: x") is None
    assert scorer.parse_answer("Answer: +3") is None
    assert scorer.parse_answer("Answer: 3 extra") is None
    assert scorer.parse_executor_result("work 1,234") == 1234
    assert scorer.parse_executor_result("3.14") is None


def test_mcnemar_is_exact_two_sided():
    result = scorer.exact_mcnemar([True] * 10, [False] * 10)
    assert result["treatment_only"] == 10
    assert result["sham_only"] == 0
    assert result["p_value_exact"] == "1/512"
    tied = scorer.exact_mcnemar([True, False], [False, True])
    assert tied["p_value_exact"] == "1/1"


def test_hash_binding_rejects_self_rehashed_substitute(tmp_path):
    paths, bindings, contract, _, _ = build_fixture(tmp_path)
    paths["transcript"].chmod(0o644)
    paths["transcript"].write_bytes(paths["transcript"].read_bytes() + b" ")
    paths["transcript"].chmod(0o444)
    with pytest.raises(scorer.IntegrityError, match="transcript hash mismatch"):
        scorer.score_artifacts(paths, bindings=bindings, contract=contract, codec=CODEC)


def test_exclusive_output_refuses_overwrite_and_is_read_only(tmp_path):
    output = tmp_path / "score.json"
    digest = scorer.write_exclusive_immutable(output, {"schema": "test"})
    assert digest == _sha(output.read_bytes())
    assert output.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        scorer.write_exclusive_immutable(output, {"schema": "replacement"})
