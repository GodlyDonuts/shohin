#!/usr/bin/env python3
"""Evaluate the sole S9.2 global-anchor development board."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from s7_learned_cayley_generator import LearnedCayleyGenerator, PRIMARY_MODULI
from s8_nil_linked_graph_compiler import (
    ROLE_INDEX,
    compile_row as compile_s8_row,
    recode_operation_ids,
    semantic_graph_key,
)
from s8_nil_linked_law_graph import (
    derange_cards,
    execute_graph,
    linked_path,
    one_witness_unit_completion,
    rewire_path,
)
from s9_1_alpha_closed_compiler import structured_spans_from_logits
from s9_2_global_anchor_compiler import (
    ANCHOR_ROLES,
    global_anchor_assignment,
    structured_spans_from_assignment,
)
from s9_occurrence_quotient import (
    compile_quotient,
    permute_relation_storage,
    quotient_from_emitted_spans,
    reindex_classes,
)
from s9_occurrence_quotient_compiler import (
    OccurrenceQuotientCompiler,
    SpanCandidate,
    all_candidates,
    compile_row as compile_s9_row,
    emitted_spans_from_logits,
    load_adapter_state,
    load_examples,
    pad_batch,
    sha256_file,
)
from s9_occurrence_quotient_falsifier import expected_graph, semantic_key
from train_s9_2_global_anchor import verify_runtime_source


ARM_NAMES = ("treatment", "positive_only", "no_class", "shuffled", "layout")
CAUSAL_ARMS = (
    "reversed_links",
    "deranged_cards",
    "one_witness",
    "state_reset",
    "early_nil",
)


def verify_evaluation_bindings(checkpoint, report, base_path, tokenizer_path):
    """Bind evaluation to the exact bytes used by training and board creation."""

    base_sha256 = sha256_file(base_path)
    tokenizer_sha256 = sha256_file(tokenizer_path)
    if base_sha256 != checkpoint.get("base_sha256"):
        raise ValueError("S9.2 evaluation base hash mismatch")
    if tokenizer_sha256 != checkpoint.get("tokenizer_sha256"):
        raise ValueError("S9.2 evaluation tokenizer/checkpoint hash mismatch")
    if tokenizer_sha256 != report.get("tokenizer_sha256"):
        raise ValueError("S9.2 evaluation tokenizer/board hash mismatch")
    if checkpoint.get("source_commit") != report.get("source_commit"):
        raise ValueError("S9.2 evaluation source commit mismatch")
    return base_sha256, tokenizer_sha256


def development_access_ledger_path(repo_root: Path, board_sha256: str) -> Path:
    return (
        repo_root
        / "artifacts"
        / "r12"
        / "access_ledgers"
        / f"s9_2_{board_sha256}.development.json"
    )


def claim_development_access(path: Path, payload: dict[str, object]) -> None:
    """Atomically consume the sole development read before opening its bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = (json.dumps(payload, sort_keys=True) + "\n").encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    os.chmod(path, 0o400)


def _instantiate(base, checkpoint, state_key):
    model = GPT(GPTConfig(**base["cfg"])).to("cuda").eval()
    model.load_state_dict(base["model"])
    architecture = checkpoint["architecture"]
    compiler = OccurrenceQuotientCompiler(
        model,
        layer=architecture["layer"],
        width=architecture["width"],
        heads=architecture["heads"],
        encoder_layers=architecture["encoder_layers"],
        ff=architecture["ff"],
    ).to("cuda")
    load_adapter_state(compiler, checkpoint[state_key])
    return compiler


def _target_stripped(
    candidates: Sequence[SpanCandidate],
) -> tuple[SpanCandidate, ...]:
    return tuple(
        replace(candidate, target=ROLE_INDEX["none"]) for candidate in candidates
    )


def _root_payload(assignment, candidates):
    selected = []
    for role, index in zip(
        assignment.roles,
        assignment.candidate_indices,
        strict=True,
    ):
        candidate = candidates[index]
        selected.append(
            {
                "role": role,
                "start": int(candidate.char_start),
                "end": int(candidate.char_end),
            }
        )
    return {
        "modulus": int(assignment.modulus),
        "card_count": int(assignment.card_count),
        "event_count": int(assignment.event_count),
        "score": float(assignment.score),
        "selected": selected,
    }


def _finish_global(example, candidates, logits):
    """Decode once while retaining root diagnostics after later failures."""

    stripped = _target_stripped(candidates)
    try:
        assignment = global_anchor_assignment(stripped, logits)
    except (ValueError, IndexError) as error:
        return {"error_stage": "root", "error": str(error)}
    result = {"root": _root_payload(assignment, stripped)}
    try:
        spans = structured_spans_from_assignment(
            example,
            stripped,
            logits,
            assignment,
        )
        result["spans"] = spans
    except (ValueError, IndexError) as error:
        result.update(error_stage="child", error=str(error))
        return result
    try:
        quotient = quotient_from_emitted_spans(str(example.row["question"]), spans)
        result["quotient"] = quotient
    except (ValueError, IndexError) as error:
        result.update(error_stage="quotient", error=str(error))
        return result
    try:
        result["graph"] = compile_quotient(quotient)
    except (ValueError, IndexError) as error:
        result.update(error_stage="compile", error=str(error))
    return result


def _finish_ablation(example, candidates, logits, *, local_root):
    try:
        spans = (
            structured_spans_from_logits(example, candidates, logits)
            if local_root
            else emitted_spans_from_logits(example, candidates, logits)
        )
        quotient = quotient_from_emitted_spans(str(example.row["question"]), spans)
        return {"graph": compile_quotient(quotient)}
    except (ValueError, IndexError) as error:
        return {"error": str(error)}


def _mask_layout_gold(ids, selected):
    masked = ids.clone()
    for row_index, example in enumerate(selected):
        for start, end, _ in example.gold:
            masked[row_index, start : end + 1] = 0
    return masked


def _decode_all(
    compiler,
    examples,
    batch_size,
    class_messages,
    *,
    include_ablations=False,
    source_free=False,
    layout_only=False,
):
    if source_free and layout_only:
        raise ValueError("S9.2 source-free and layout-only modes are exclusive")
    global_result = []
    local_result = []
    unconstrained_result = []
    compiler.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            indices = list(range(start, min(len(examples), start + batch_size)))
            selected, candidates, ids, valid, tensors = pad_batch(
                examples,
                indices,
                "cuda",
                negative_limit=None,
                seed=0,
            )
            if source_free:
                ids = torch.zeros_like(ids)
            elif layout_only:
                ids = _mask_layout_gold(ids, selected)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = compiler(
                    ids,
                    valid,
                    tensors,
                    class_messages=class_messages,
                )
            cursor = 0
            for example, row_candidates in zip(selected, candidates, strict=True):
                width = len(row_candidates)
                logits = outputs["role_logits"][cursor : cursor + width]
                cursor += width
                global_result.append(_finish_global(example, row_candidates, logits))
                if include_ablations:
                    local_result.append(
                        _finish_ablation(
                            example,
                            row_candidates,
                            logits,
                            local_root=True,
                        )
                    )
                    unconstrained_result.append(
                        _finish_ablation(
                            example,
                            row_candidates,
                            logits,
                            local_root=False,
                        )
                    )
    return global_result, local_result, unconstrained_result


def _uniform_decode(examples):
    result = []
    for example in examples:
        candidates = all_candidates(example)
        logits = torch.zeros((len(candidates), 14))
        result.append(_finish_global(example, candidates, logits))
    return result


def _new_score():
    return {"state": 0, "answer": 0, "total": 0, "depth": {}}


def _add(score, depth, output, expected_state, expected_answer):
    state_ok = output is not None and output[0] == expected_state
    answer_ok = output is not None and output[1] == expected_answer
    score["state"] += int(state_ok)
    score["answer"] += int(answer_ok)
    score["total"] += 1
    cell = score["depth"].setdefault(str(depth), {"correct": 0, "total": 0})
    cell["correct"] += int(state_ok)
    cell["total"] += 1


def _finish_score(score):
    return {
        **score,
        "state_accuracy": score["state"] / score["total"],
        "answer_accuracy": score["answer"] / score["total"],
        "depth": {
            key: {**value, "accuracy": value["correct"] / value["total"]}
            for key, value in sorted(score["depth"].items())
        },
    }


def _coarse(label: str) -> str:
    if label.startswith("entity.roster."):
        return "entity.roster"
    if label.startswith("position.roster."):
        return "position.roster"
    if label.startswith("state.entity."):
        return "state.entity"
    if label.startswith("card."):
        return "card." + label.rsplit(".", 1)[-1]
    if label.startswith("event."):
        return "event." + label.rsplit(".", 1)[-1]
    return label


def _span_set(spans):
    return {
        (int(value["start"]), int(value["end"]), _coarse(str(label)))
        for label, value in spans.items()
    }


def _gold_root_set(row):
    return {
        (int(value["start"]), int(value["end"]), _coarse(str(label)))
        for label, value in row["spans"].items()
        if _coarse(str(label)) in ANCHOR_ROLES
    }


def _predicted_root_set(decoded):
    if "root" not in decoded:
        return None
    return {
        (int(value["start"]), int(value["end"]), str(value["role"]))
        for value in decoded["root"]["selected"]
    }


def _root_transport_key(decoded, row):
    """Map selected roots to stable semantic occurrence labels post hoc."""

    if "root" not in decoded:
        return None
    labels = {
        (int(value["start"]), int(value["end"])): str(label)
        for label, value in row["spans"].items()
        if _coarse(str(label)) in ANCHOR_ROLES
    }
    transported = []
    for value in decoded["root"]["selected"]:
        span = (int(value["start"]), int(value["end"]))
        label = labels.get(span)
        if label is None:
            return None
        transported.append((str(value["role"]), label))
    return tuple(sorted(transported))


def _new_root_score():
    return {
        "eligible": 0,
        "span_exact": 0,
        "count_exact": 0,
        "modulus_exact": 0,
        "card_count_exact": 0,
        "event_count_exact": 0,
        "total": 0,
        "failure_stage": {},
    }


def _add_root(score, decoded, row):
    score["total"] += 1
    if "root" not in decoded:
        stage = str(decoded.get("error_stage", "root"))
        score["failure_stage"][stage] = score["failure_stage"].get(stage, 0) + 1
        return
    score["eligible"] += 1
    root = decoded["root"]
    score["span_exact"] += int(_predicted_root_set(decoded) == _gold_root_set(row))
    modulus_ok = int(root["modulus"]) == int(row["modulus"])
    card_ok = int(root["card_count"]) == len(row["cards"])
    event_ok = int(root["event_count"]) == int(row["depth"])
    score["modulus_exact"] += int(modulus_ok)
    score["card_count_exact"] += int(card_ok)
    score["event_count_exact"] += int(event_ok)
    score["count_exact"] += int(modulus_ok and card_ok and event_ok)
    if "error_stage" in decoded:
        stage = str(decoded["error_stage"])
        score["failure_stage"][stage] = score["failure_stage"].get(stage, 0) + 1


def _finish_root(score):
    total = score["total"]
    return {
        **score,
        "eligible_accuracy": score["eligible"] / total,
        "span_exact_accuracy": score["span_exact"] / total,
        "count_exact_accuracy": score["count_exact"] / total,
        "modulus_exact_accuracy": score["modulus_exact"] / total,
        "card_count_exact_accuracy": score["card_count_exact"] / total,
        "event_count_exact_accuracy": score["event_count_exact"] / total,
    }


def _state_hash(values):
    payload = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _execute(decoded, successors, zeros, modulus):
    if "graph" not in decoded:
        return None
    try:
        linked_path(decoded["graph"])
        return execute_graph(decoded["graph"], successors[modulus], zeros[modulus])
    except (ValueError, IndexError):
        return None


def _fit_payload(checkpoint):
    fit = checkpoint.get("fit")
    if not isinstance(fit, dict) or any(name not in fit for name in ARM_NAMES):
        raise SystemExit("S9.2 checkpoint is missing the frozen five-arm fit payload")
    return {name: fit[name] for name in ARM_NAMES}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing S9.2 evaluation: {args.out}")
    if not torch.cuda.is_available():
        raise SystemExit("S9.2 evaluation requires CUDA")
    torch.set_float32_matmul_precision("high")

    report_path = args.data_dir / "report.json"
    report = json.loads(report_path.read_text())
    audit = report["audit"]
    if audit["development_accesses"] or audit["confirmation_accesses"]:
        raise SystemExit("S9.2 board access counters are not zero")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "r12_s9_2_global_anchor_checkpoint_v1":
        raise SystemExit("unexpected S9.2 checkpoint schema")
    if checkpoint["board_report_sha256"] != sha256_file(report_path):
        raise SystemExit("S9.2 checkpoint/board mismatch")
    try:
        verify_evaluation_bindings(
            checkpoint,
            report,
            args.base,
            args.tokenizer,
        )
        verify_runtime_source(
            Path(__file__).resolve().parents[1],
            checkpoint["source_commit"],
        )
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    fit = _fit_payload(checkpoint)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    development_path = args.data_dir / "development.jsonl"
    access_ledger = development_access_ledger_path(
        Path(__file__).resolve().parents[1],
        checkpoint["board_report_sha256"],
    )
    try:
        claim_development_access(
            access_ledger,
            {
                "schema": "r12_s9_2_development_access_v1",
                "board_report_sha256": checkpoint["board_report_sha256"],
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "source_commit": checkpoint["source_commit"],
                "development_accesses": 1,
                "confirmation_accesses": 0,
            },
        )
    except FileExistsError as error:
        raise SystemExit("S9.2 development board was already consumed") from error
    if sha256_file(development_path) != report["files"]["development.jsonl"]["sha256"]:
        raise SystemExit("S9.2 development hash mismatch")
    examples = load_examples(
        development_path,
        tokenizer,
        "s8_nil_graph_development",
        base["cfg"]["seq_len"],
    )
    if len(examples) != 2048:
        raise SystemExit("S9.2 requires 2,048 development rows")

    treatment_model = _instantiate(
        base,
        checkpoint,
        "treatment_adapter_state",
    )
    treatment, local_root, unconstrained = _decode_all(
        treatment_model,
        examples,
        args.batch_size,
        True,
        include_ablations=True,
    )
    recoded_examples = [
        compile_s9_row(
            recode_operation_ids(
                compile_s8_row(example.row, tokenizer),
                tokenizer,
            ).row,
            tokenizer,
        )
        for example in examples
    ]
    recoded, _, _ = _decode_all(
        treatment_model,
        recoded_examples,
        args.batch_size,
        True,
    )
    source_free, _, _ = _decode_all(
        treatment_model,
        examples,
        args.batch_size,
        True,
        source_free=True,
    )
    del treatment_model
    torch.cuda.empty_cache()

    decoded_arms = {
        "treatment": treatment,
        "positive_only": None,
        "no_class": None,
        "shuffled": None,
        "layout": None,
    }
    arm_specs = (
        ("positive_only", "positive_only_adapter_state", True, False),
        ("no_class", "no_class_adapter_state", False, False),
        ("shuffled", "shuffled_adapter_state", True, False),
        ("layout", "layout_adapter_state", False, True),
    )
    for name, state_key, class_messages, layout_only in arm_specs:
        model = _instantiate(base, checkpoint, state_key)
        decoded, _, _ = _decode_all(
            model,
            examples,
            args.batch_size,
            class_messages,
            layout_only=layout_only,
        )
        decoded_arms[name] = decoded
        del model
        torch.cuda.empty_cache()
    uniform = _uniform_decode(examples)

    generator = LearnedCayleyGenerator().to("cuda")
    generator.load_state_dict(checkpoint["generator_state"])
    successors = {
        modulus: generator.discrete_successor(modulus) for modulus in PRIMARY_MODULI
    }
    zeros = {modulus: generator.discrete_zero(modulus) for modulus in PRIMARY_MODULI}
    score_names = (
        "gold_graph",
        "treatment",
        "positive_only",
        "no_class_message",
        "shuffled",
        "layout",
        *CAUSAL_ARMS,
    )
    arms = {name: _new_score() for name in score_names}
    graph = {
        "valid": 0,
        "exact": 0,
        "positive_only_exact": 0,
        "no_class_exact": 0,
        "shuffled_exact": 0,
        "layout_exact": 0,
        "local_root_exact": 0,
        "unconstrained_exact": 0,
        "source_free_exact": 0,
        "uniform_exact": 0,
    }
    spans = {
        "tp": 0,
        "predicted": 0,
        "gold": 0,
        "row_exact": 0,
        "class_exact": 0,
    }
    root_scores = {
        name: _new_root_score()
        for name in (*ARM_NAMES, "source_free", "uniform", "recoded_treatment")
    }
    invariance = {
        "eligible": 0,
        "class_reindex": 0,
        "relation_storage_reindex": 0,
        "nonce_eligible": 0,
        "nonce_graph_identical": 0,
        "nonce_state_identical": 0,
        "nonce_answer_identical": 0,
        "nonce_root_eligible": 0,
        "nonce_root_identical": 0,
        "nonce_count_identical": 0,
    }
    sample = []
    treatment_states = []
    for index, example in enumerate(examples):
        row = example.row
        modulus, depth = int(row["modulus"]), int(row["depth"])
        expected_state = tuple(int(value) for value in row["final_state"])
        expected_answer = int(row["answer"])
        expected = expected_graph(row)
        gold_output = execute_graph(expected, successors[modulus], zeros[modulus])
        _add(arms["gold_graph"], depth, gold_output, expected_state, expected_answer)

        for name in ARM_NAMES:
            _add_root(root_scores[name], decoded_arms[name][index], row)
        _add_root(root_scores["source_free"], source_free[index], row)
        _add_root(root_scores["uniform"], uniform[index], row)
        _add_root(
            root_scores["recoded_treatment"],
            recoded[index],
            recoded_examples[index].row,
        )

        decoded = treatment[index]
        output = None
        controls_scored = False
        if "graph" in decoded:
            predicted = decoded["graph"]
            try:
                path = tuple(linked_path(predicted))
                graph["valid"] += 1
                graph["exact"] += int(semantic_key(predicted) == semantic_key(expected))
                output = execute_graph(predicted, successors[modulus], zeros[modulus])
                controls = {
                    "reversed_links": rewire_path(predicted, tuple(reversed(path))),
                    "deranged_cards": derange_cards(predicted),
                    "one_witness": one_witness_unit_completion(
                        predicted,
                        successors[modulus],
                        zeros[modulus],
                    ),
                }
                for name, control in controls.items():
                    _add(
                        arms[name],
                        depth,
                        execute_graph(control, successors[modulus], zeros[modulus]),
                        expected_state,
                        expected_answer,
                    )
                _add(
                    arms["state_reset"],
                    depth,
                    execute_graph(
                        predicted,
                        successors[modulus],
                        zeros[modulus],
                        reset_state=True,
                    ),
                    expected_state,
                    expected_answer,
                )
                _add(
                    arms["early_nil"],
                    depth,
                    execute_graph(
                        predicted,
                        successors[modulus],
                        zeros[modulus],
                        halt_after=1,
                    ),
                    expected_state,
                    expected_answer,
                )
                controls_scored = True
                quotient = decoded["quotient"]
                class_perm = tuple(reversed(range(len(quotient.classes))))
                relation_perm = tuple(reversed(range(len(quotient.relations))))
                invariance["eligible"] += 1
                invariance["class_reindex"] += int(
                    semantic_key(
                        compile_quotient(reindex_classes(quotient, class_perm))
                    )
                    == semantic_key(predicted)
                )
                invariance["relation_storage_reindex"] += int(
                    semantic_key(
                        compile_quotient(
                            permute_relation_storage(quotient, relation_perm)
                        )
                    )
                    == semantic_key(predicted)
                )
            except (ValueError, IndexError):
                pass
        if "spans" in decoded:
            predicted_spans = _span_set(decoded["spans"])
            gold_spans = _span_set(row["spans"])
            spans["tp"] += len(predicted_spans & gold_spans)
            spans["predicted"] += len(predicted_spans)
            spans["gold"] += len(gold_spans)
            spans["row_exact"] += int(predicted_spans == gold_spans)
            spans["class_exact"] += int(
                {value[:2] for value in predicted_spans}
                == {value[:2] for value in gold_spans}
            )
        else:
            spans["gold"] += len(_span_set(row["spans"]))
        _add(arms["treatment"], depth, output, expected_state, expected_answer)
        treatment_states.append(list(output[0]) if output is not None else None)
        if not controls_scored:
            for name in CAUSAL_ARMS:
                _add(arms[name], depth, None, expected_state, expected_answer)

        for name, score_name, graph_key in (
            ("positive_only", "positive_only", "positive_only_exact"),
            ("no_class", "no_class_message", "no_class_exact"),
            ("shuffled", "shuffled", "shuffled_exact"),
            ("layout", "layout", "layout_exact"),
        ):
            arm_decoded = decoded_arms[name][index]
            if "graph" in arm_decoded:
                graph[graph_key] += int(
                    semantic_key(arm_decoded["graph"]) == semantic_key(expected)
                )
            arm_output = _execute(arm_decoded, successors, zeros, modulus)
            _add(arms[score_name], depth, arm_output, expected_state, expected_answer)

        for values, name in (
            (local_root, "local_root_exact"),
            (unconstrained, "unconstrained_exact"),
            (source_free, "source_free_exact"),
            (uniform, "uniform_exact"),
        ):
            if "graph" in values[index]:
                graph[name] += int(
                    semantic_key(values[index]["graph"]) == semantic_key(expected)
                )

        if output is not None:
            recoded_output = _execute(
                recoded[index],
                successors,
                zeros,
                modulus,
            )
            if recoded_output is not None:
                invariance["nonce_eligible"] += 1
                invariance["nonce_graph_identical"] += int(
                    semantic_graph_key(recoded[index]["graph"])
                    == semantic_graph_key(decoded["graph"])
                )
                invariance["nonce_state_identical"] += int(
                    recoded_output[0] == output[0]
                )
                invariance["nonce_answer_identical"] += int(
                    recoded_output[1] == output[1]
                )
            original_root = _root_transport_key(decoded, row)
            recoded_root = _root_transport_key(
                recoded[index],
                recoded_examples[index].row,
            )
            if original_root is not None and recoded_root is not None:
                invariance["nonce_root_eligible"] += 1
                invariance["nonce_root_identical"] += int(original_root == recoded_root)
                original_counts = (
                    decoded["root"]["modulus"],
                    decoded["root"]["card_count"],
                    decoded["root"]["event_count"],
                )
                recoded_counts = (
                    recoded[index]["root"]["modulus"],
                    recoded[index]["root"]["card_count"],
                    recoded[index]["root"]["event_count"],
                )
                invariance["nonce_count_identical"] += int(
                    original_counts == recoded_counts
                )

        if len(sample) < 16:
            sample.append(
                {
                    "id": row["id"],
                    "renderer": row["renderer"],
                    "depth": depth,
                    "root": decoded.get("root"),
                    "error_stage": decoded.get("error_stage"),
                    "error": decoded.get("error"),
                    "valid": "graph" in decoded,
                    "exact": (
                        "graph" in decoded
                        and semantic_key(decoded["graph"]) == semantic_key(expected)
                    ),
                    "recoded_error_stage": recoded[index].get("error_stage"),
                    "recoded_error": recoded[index].get("error"),
                }
            )

    finished = {name: _finish_score(value) for name, value in arms.items()}
    precision = spans["tp"] / max(1, spans["predicted"])
    recall = spans["tp"] / max(1, spans["gold"])
    spans.update(
        {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
            "row_exact_accuracy": spans["row_exact"] / len(examples),
            "class_exact_accuracy": spans["class_exact"] / len(examples),
        }
    )
    graph.update(
        {key + "_accuracy": value / len(examples) for key, value in graph.items()}
    )
    roots = {name: _finish_root(value) for name, value in root_scores.items()}
    evaluation = {
        "schema": "r12_s9_2_global_anchor_development_evaluation_v1",
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "board_report_sha256": sha256_file(report_path),
        "development_sha256": sha256_file(development_path),
        "parameters": checkpoint["parameters"],
        "architecture": checkpoint["architecture"],
        "source_commit": checkpoint["source_commit"],
        "base_sha256": checkpoint["base_sha256"],
        "tokenizer_sha256": checkpoint["tokenizer_sha256"],
        "access_ledger_sha256": sha256_file(access_ledger),
        "training_contract": checkpoint["training_contract"],
        "fit": fit,
        "generator_fit": checkpoint["generator_fit"],
        "rows": len(examples),
        "span": spans,
        "root": roots,
        "graph": graph,
        "arms": finished,
        "invariance": invariance,
        "predicted_state_sha256": _state_hash(treatment_states),
        "samples": sample,
        "development_accesses": 1,
        "confirmation_accesses": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "graph": graph,
                "span": spans,
                "root": roots["treatment"],
                "treatment": finished["treatment"],
                "invariance": invariance,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
