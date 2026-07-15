#!/usr/bin/env python3
"""Executable custody tests for SFT accounting and RSP-C1 job wrappers."""
from __future__ import annotations

import ast
import copy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "train" / "sft.py"
SFT_JOB = ROOT / "train" / "jobs" / "sft_residual_packet_v1.sbatch"
EVAL_JOB = ROOT / "train" / "jobs" / "eval_residual_packet_v1.sbatch"


def _marked_python(path: Path, marker: str) -> str:
    text = path.read_text()
    begin = f"# {marker}_BEGIN\n"
    end = f"# {marker}_END"
    assert text.count(begin) == 1 and text.count(end) == 1
    return text.split(begin, 1)[1].split(end, 1)[0]


def _load_marked_python(path: Path, marker: str) -> SimpleNamespace:
    source = _marked_python(path, marker)
    namespace = {"__name__": f"test_{marker.lower()}"}
    exec(compile(source, f"{path}:{marker}", "exec"), namespace)
    return SimpleNamespace(**namespace)


def _is_a_seed(node):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "seed"
        and isinstance(node.value, ast.Name)
        and node.value.id == "a"
    )


def _calls(tree, owner, method):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != method:
            continue
        value = node.func.value
        if isinstance(value, ast.Name) and value.id == owner:
            yield node
        elif (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == owner
        ):
            yield node


def test_seed_and_floor_step_schedule_are_live_contracts():
    tree = ast.parse(SFT.read_text())
    seed_arguments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--seed"
    ]
    assert len(seed_arguments) == 1
    keywords = {keyword.arg: keyword.value for keyword in seed_arguments[0].keywords}
    assert isinstance(keywords["default"], ast.Constant) and keywords["default"].value == 1337
    assert isinstance(keywords["type"], ast.Name) and keywords["type"].id == "int"
    torch_seeds = list(_calls(tree, "torch", "manual_seed"))
    numpy_seeds = list(_calls(tree, "np", "default_rng"))
    assert len(torch_seeds) == 1 and _is_a_seed(torch_seeds[0].args[0])
    assert len(numpy_seeds) == 1 and _is_a_seed(numpy_seeds[0].args[0])

    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    steps = assignments["steps_per_epoch"]
    assert isinstance(steps, ast.BinOp) and isinstance(steps.op, ast.FloorDiv)
    assert isinstance(steps.left, ast.Name) and steps.left.id == "N"
    assert isinstance(steps.right, ast.Attribute) and steps.right.attr == "batch_size"
    total = assignments["total_steps"]
    assert isinstance(total, ast.BinOp) and isinstance(total.op, ast.Mult)
    assert {getattr(total.left, "id", None), getattr(total.right, "id", None)} == {None, "steps_per_epoch"}
    assert "math.ceil(N / a.batch_size)" not in SFT.read_text()


def _base_pair(module):
    h = {
        "treatment": "1" * 64,
        "sham": "2" * 64,
        "treatment_receipt": "3" * 64,
        "sham_receipt": "4" * 64,
        "tokenizer": "5" * 64,
        "preregistration": "6" * 64,
        "training_manifest": "7" * 64,
        "admission_audit": "8" * 64,
        "training_resources": "9" * 64,
        "executor": "a" * 64,
        "model": "b" * 64,
        "protocol": "c" * 64,
    }
    source_hashes = {
        "job_sha256": "d" * 64,
        "trainer_sha256": "e" * 64,
        "encoding_sha256": "f" * 64,
        "model_sha256": h["model"],
        "muon_sha256": "0" * 64,
        "protocol_sha256": h["protocol"],
    }
    common_artifacts = {
        "metadata_sha256": "1" * 64,
        "pair_equality_sha256": "2" * 64,
        "init_sha256": "3" * 64,
        "tokenizer_sha256": h["tokenizer"],
        "treatment_data_sha256": "4" * 64,
        "sham_data_sha256": "5" * 64,
        "training_manifest_sha256": h["training_manifest"],
        "admission_audit_sha256": h["admission_audit"],
        "preregistration_sha256": h["preregistration"],
    }
    accounting = {
        "full_tokens_before_packing": 100_000,
        "packed_sequences": 780,
        "discarded_tail_tokens": 160,
        "steps_per_epoch": 12,
        "optimizer_updates": 120,
        "packed_forward_token_positions": 983_040,
        "supervised_completion_tokens": 123_456,
    }
    runtime = {
        "python_executable": "/python",
        "python_executable_sha256": "8" * 64,
        "python_version": "3.test",
        "platform": "test-platform",
        "numpy": "test",
        "torch": "test",
        "tokenizers": "test",
        "cuda_runtime": "test",
        "cudnn": 1,
        "gpu": [{"index": 0, "name": "H100", "capability": [9, 0], "total_memory": 80}],
        "nvidia_smi": ["GPU-uuid, H100, driver"],
        "deterministic_algorithms_requested": True,
        "cublas_workspace_config": ":4096:8",
        "pythonhashseed": str(module.FIT_SEEDS[0]),
    }

    def receipt(arm, checkpoint, data):
        return {
            "schema": module.FIT_RECEIPT_SCHEMA,
            "arm": arm,
            "paired_seed": module.FIT_SEEDS[0],
            "run_manifest_sha256": "6" * 64,
            "checkpoint_step": "sft_ep10",
            "artifacts": {
                **common_artifacts,
                "checkpoint_sha256": checkpoint,
                "data_sha256": data,
            },
            "sources": source_hashes,
            "fit": module.FIT,
            "accounting": accounting,
            "pair_geometry_sha256": "7" * 64,
            "runtime": runtime,
            "slurm_job_id": "123",
        }

    treatment = receipt("treatment", h["treatment"], common_artifacts["treatment_data_sha256"])
    sham = receipt("sham", h["sham"], common_artifacts["sham_data_sha256"])
    common = {
        "init_sha256": common_artifacts["init_sha256"],
        "tokenizer_sha256": h["tokenizer"],
        "preregistration_sha256": h["preregistration"],
        "training_manifest_sha256": h["training_manifest"],
        "admission_audit_sha256": h["admission_audit"],
        "sft_run_manifest_sha256": treatment["run_manifest_sha256"],
        "fit": module.FIT,
        "sources": source_hashes,
        "pair_geometry_sha256": treatment["pair_geometry_sha256"],
        "runtime_contract_sha256": module.canonical_digest(
            module.stable_runtime_contract(runtime, module.FIT_SEEDS[0])
        ),
    }
    controller = {
        "schema": module.CONTROLLER_SCHEMA,
        "paired_seed": module.FIT_SEEDS[0],
        "common": common,
        "arms": {
            "treatment": {
                "checkpoint_sha256": h["treatment"],
                "fit_receipt_sha256": h["treatment_receipt"],
                "data_sha256": common_artifacts["treatment_data_sha256"],
            },
            "sham": {
                "checkpoint_sha256": h["sham"],
                "fit_receipt_sha256": h["sham_receipt"],
                "data_sha256": common_artifacts["sham_data_sha256"],
            },
        },
        "training_resources_sha256": h["training_resources"],
    }
    resources = {
        "schema": module.RESOURCE_SCHEMA,
        "paired_seed": module.FIT_SEEDS[0],
        "treatment": {
            "checkpoint_sha256": h["treatment"],
            "fit_receipt_sha256": h["treatment_receipt"],
            "supervised_completion_tokens": accounting["supervised_completion_tokens"],
            "packed_forward_token_positions": accounting["packed_forward_token_positions"],
            "optimizer_updates": accounting["optimizer_updates"],
        },
        "sham": {
            "checkpoint_sha256": h["sham"],
            "fit_receipt_sha256": h["sham_receipt"],
            "supervised_completion_tokens": accounting["supervised_completion_tokens"],
            "packed_forward_token_positions": accounting["packed_forward_token_positions"],
            "optimizer_updates": accounting["optimizer_updates"],
        },
    }
    executor = {
        "schema": module.EXECUTOR_SCHEMA,
        "checkpoint_sha256": h["executor"],
        "checkpoint_step": 260000,
        "tokenizer_sha256": h["tokenizer"],
        "model_sha256": h["model"],
        "protocol_sha256": h["protocol"],
    }
    return h, controller, executor, resources, treatment, sham


def test_exact_pair_contract_accepts_one_valid_pair():
    module = _load_marked_python(EVAL_JOB, "EVAL_CUSTODY_PY")
    args = _base_pair(module)
    module.validate_pair_contract(args[1], args[2], args[3], args[4], args[5], args[0], module.FIT_SEEDS[0])


def test_exact_pair_contract_rejects_swapped_arms():
    module = _load_marked_python(EVAL_JOB, "EVAL_CUSTODY_PY")
    h, controller, executor, resources, treatment, sham = _base_pair(module)
    with pytest.raises(ValueError, match="identity|swapped"):
        module.validate_pair_contract(controller, executor, resources, sham, treatment, h, module.FIT_SEEDS[0])


def test_exact_pair_contract_rejects_cross_seed_receipt():
    module = _load_marked_python(EVAL_JOB, "EVAL_CUSTODY_PY")
    h, controller, executor, resources, treatment, sham = _base_pair(module)
    sham = copy.deepcopy(sham)
    sham["paired_seed"] = module.FIT_SEEDS[1]
    with pytest.raises(ValueError, match="identity"):
        module.validate_pair_contract(controller, executor, resources, treatment, sham, h, module.FIT_SEEDS[0])


def test_exact_pair_contract_rejects_hash_in_irrelevant_named_field():
    module = _load_marked_python(EVAL_JOB, "EVAL_CUSTODY_PY")
    h, controller, executor, resources, treatment, sham = _base_pair(module)
    controller = copy.deepcopy(controller)
    controller["arms"]["treatment"]["checkpoint_sha256"] = "0" * 64
    controller["arms"]["treatment"]["data_sha256"] = h["treatment"]
    with pytest.raises(ValueError, match="named bindings"):
        module.validate_pair_contract(controller, executor, resources, treatment, sham, h, module.FIT_SEEDS[0])


def test_sft_pair_geometry_rejects_rowwise_mask_mismatch_despite_response_multiset_parity():
    module = _load_marked_python(SFT_JOB, "SFT_CUSTODY_PY")

    class Tokenizer:
        pass

    def encode(_tokenizer, prompt, completion, eos_id):
        prompt_ids = [1] * len(prompt)
        completion_ids = [2] * len(completion)
        full = prompt_ids + completion_ids + [eos_id]
        return prompt_ids, full, [0] * len(prompt_ids) + [1] * (len(completion_ids) + 1)

    treatment = []
    for index in range(128):
        response = "x" * (10 if index % 2 == 0 else 20)
        treatment.append({
            "completion_prompt": f"Prompt {index}: " + "p" * 80,
            "id": f"row_{index}",
            "kind": "compiler",
            "program_id": f"program_{index}",
            "question": f"Prompt {index}: " + "p" * 80,
            "response": response,
            "training_group": "residual_packet",
        })
    sham = copy.deepcopy(treatment)
    sham[0]["response"], sham[1]["response"] = sham[1]["response"], sham[0]["response"]
    with pytest.raises(ValueError, match="per-row token/mask geometry"):
        module.compute_pair_geometry(treatment, sham, Tokenizer(), 0, encode, module.FIT_SEEDS[0])


def test_sft_pair_geometry_rejects_different_discarded_tail_with_equal_geometry_and_multisets():
    module = _load_marked_python(SFT_JOB, "SFT_CUSTODY_PY")

    class Tokenizer:
        pass

    def encode(_tokenizer, prompt, completion, eos_id):
        prompt_ids = [1] * len(prompt)
        completion_ids = [2 + (ord(char) % 17) for char in completion]
        full = prompt_ids + completion_ids + [eos_id]
        return prompt_ids, full, [0] * len(prompt_ids) + [1] * (len(completion_ids) + 1)

    treatment = []
    for index in range(128):
        prompt = f"Prompt {index}: " + "p" * 80
        treatment.append({
            "completion_prompt": prompt,
            "id": f"row_{index}",
            "kind": "compiler",
            "program_id": f"program_{index}",
            "question": prompt,
            "response": ("a" if index % 2 == 0 else "b") * 10,
            "training_group": "residual_packet",
        })
    sham = copy.deepcopy(treatment)
    sham[-2]["response"], sham[-1]["response"] = sham[-1]["response"], sham[-2]["response"]
    with pytest.raises(ValueError, match="discarded token/mask tails"):
        module.compute_pair_geometry(treatment, sham, Tokenizer(), 0, encode, module.FIT_SEEDS[0])


def test_wrappers_bind_snapshots_receipts_and_final_rehashes():
    sft = SFT_JOB.read_text()
    evaluation = EVAL_JOB.read_text()
    for required in (
        "rsp_c1_sft_run_manifest_v1",
        "rsp_c1_fit_receipt_v1",
        "copy_verified",
        "full_tokens_before_packing",
        "full token-ID multisets differ between arms",
        "discarded token/mask tails differ between arms",
        "pair_geometry_sha256",
        "final input rehash failed",
        "sealed output postcondition failed",
        "torch.use_deterministic_algorithms(True)",
        "CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONNOUSERSITE=1",
    ):
        assert required in sft
    for required in (
        "rsp_c1_eval_run_manifest_v1",
        "rsp_c1_controller_pair_manifest_v1",
        "rsp_c1_executor_manifest_v1",
        "rsp_c1_transcript_receipt_v1",
        "copy_verified",
        "final input rehash failed",
        "transcript changed while its receipt was sealed",
        "transcript input hashes differ",
        "transcript schema/seed mismatch",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONNOUSERSITE=1",
    ):
        assert required in evaluation
    assert "in serialized" not in sft
    assert "in json.dumps" not in evaluation


def test_job_shell_and_embedded_python_syntax():
    for job in (SFT_JOB, EVAL_JOB):
        subprocess.run(["bash", "-n", str(job)], check=True)
        blocks = job.read_text().split("<<'PY'\n")[1:]
        assert blocks
        for block in blocks:
            source, marker, _ = block.partition("\nPY\n")
            assert marker, f"unterminated Python heredoc in {job.name}"
            ast.parse(source)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
