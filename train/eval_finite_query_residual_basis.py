#!/usr/bin/env python3
"""Read-only group-causal evaluation for Finite-Query Residual Basis (FQRB).

Each FQRB group shares one source-free ``donor + edited - base`` tape across
five incompatible finite consumers.  The evaluator encodes that triple once,
then measures normal, paraphrase, counterfactual, zero, whole-group shuffle,
and wrong-query controls.  A group counts as strict only when every consumer
passes every required control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from counterfactual_residual_algebra import compose_counterfactual_tape, compose_two_edit_counterfactual_tape
from eval_counterfactual_residual_algebra import decode_tape, encode_tape
from model import GPT, GPTConfig


QUERY_KINDS = ("ones", "tens", "sign", "parity", "relation")
ONE_EDIT_SHARED_SOURCE_FIELDS = (
    "base_source", "edited_source", "donor_source",
    "paraphrase_base_source", "paraphrase_edited_source", "paraphrase_donor_source",
)
TWO_EDIT_SHARED_SOURCE_FIELDS = (
    "base_source", "primary_edited_source", "secondary_edited_source", "donor_source",
    "paraphrase_base_source", "paraphrase_primary_edited_source", "paraphrase_secondary_edited_source",
    "paraphrase_donor_source",
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_rows(rows: list[dict], split: str, max_groups: int) -> list[tuple[str, list[dict]]]:
    """Validate and preserve finite-query groups in data-file order."""
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for row in rows:
        if row.get("schema") != "counterfactual_residual_algebra_v1" or row.get("split") != split:
            continue
        if row.get("basis_mode") not in ("multi_consumer", "multi_consumer_two_edit", "multi_consumer_ephemeral_codebook") or not isinstance(row.get("basis_id"), str):
            raise ValueError("row is not a finite-query residual-basis example")
        groups.setdefault(row["basis_id"], []).append(row)
    result = []
    for basis_id, group in groups.items():
        kinds = {row.get("query_kind") for row in group}
        if len(group) != len(QUERY_KINDS) or kinds != set(QUERY_KINDS):
            raise ValueError("basis {} does not contain exactly one row per query kind".format(basis_id))
        first = group[0]
        mode = first.get("mode", "one_edit")
        if any(row.get("mode", "one_edit") != mode for row in group):
            raise ValueError("basis {} mixes composition modes".format(basis_id))
        shared_fields = TWO_EDIT_SHARED_SOURCE_FIELDS if mode == "two_edit" else ONE_EDIT_SHARED_SOURCE_FIELDS
        if mode not in ("one_edit", "two_edit"):
            raise ValueError("basis {} has an unknown composition mode".format(basis_id))
        if any(any(row.get(field) != first.get(field) for row in group[1:]) for field in shared_fields):
            raise ValueError("basis {} does not share an identical source triple".format(basis_id))
        if first.get("basis_mode") == "multi_consumer_ephemeral_codebook":
            if not isinstance(first.get("codebook"), dict) or any(row.get("codebook") != first["codebook"] for row in group[1:]):
                raise ValueError("ephemeral-codebook basis {} does not share one binding table".format(basis_id))
        if any(row.get("response") == row.get("counterfactual_response") for row in group):
            raise ValueError("basis {} has an answer-invariant counterfactual".format(basis_id))
        indexed = {row["query_kind"]: row for row in group}
        result.append((basis_id, [indexed[kind] for kind in QUERY_KINDS]))
    if max_groups:
        result = result[:max_groups]
    if not result:
        raise ValueError("no FQRB groups for requested split")
    return result


def compose_group_tapes(model, tokenizer, group: list[dict], layer: int, tape_len: int, source_window: int = 0):
    """Encode a shared FQRB group once, with optional unseen two-edit composition."""
    first = group[0]
    base = encode_tape(model, tokenizer, first["base_source"], layer, tape_len, source_window)
    donor = encode_tape(model, tokenizer, first["donor_source"], layer, tape_len, source_window)
    para_base = encode_tape(model, tokenizer, first["paraphrase_base_source"], layer, tape_len, source_window)
    para_donor = encode_tape(model, tokenizer, first["paraphrase_donor_source"], layer, tape_len, source_window)
    if first.get("mode", "one_edit") == "two_edit":
        primary = encode_tape(model, tokenizer, first["primary_edited_source"], layer, tape_len, source_window)
        secondary = encode_tape(model, tokenizer, first["secondary_edited_source"], layer, tape_len, source_window)
        para_primary = encode_tape(model, tokenizer, first["paraphrase_primary_edited_source"], layer, tape_len, source_window)
        para_secondary = encode_tape(model, tokenizer, first["paraphrase_secondary_edited_source"], layer, tape_len, source_window)
        return (
            compose_two_edit_counterfactual_tape(base, primary, secondary, donor),
            compose_two_edit_counterfactual_tape(para_base, para_primary, para_secondary, para_donor),
            base, donor, secondary,
        )
    edited = encode_tape(model, tokenizer, first["edited_source"], layer, tape_len, source_window)
    para_edited = encode_tape(model, tokenizer, first["paraphrase_edited_source"], layer, tape_len, source_window)
    return compose_counterfactual_tape(base, edited, donor), compose_counterfactual_tape(para_base, para_edited, para_donor), base, donor, None


def compose_counter_tape(model, tokenizer, row: dict, base, donor, secondary, layer: int, tape_len: int, source_window: int = 0):
    if row.get("mode", "one_edit") == "two_edit":
        counter = encode_tape(model, tokenizer, row["counterfactual_primary_edited_source"], layer, tape_len, source_window)
        return compose_two_edit_counterfactual_tape(base, counter, secondary, donor)
    counter = encode_tape(model, tokenizer, row["counterfactual_edited_source"], layer, tape_len, source_window)
    return compose_counterfactual_tape(base, counter, donor)


def shifted_group_keys(group_keys: list[str]) -> dict[str, str]:
    """Map every group to a different complete source group for shuffle control."""
    if len(group_keys) < 2:
        raise ValueError("whole-group shuffle requires at least two groups")
    return {key: group_keys[(index + 1) % len(group_keys)] for index, key in enumerate(group_keys)}


def score_result(result: dict) -> dict:
    if result["expected"] == result["counterfactual_expected"]:
        raise ValueError("counterfactual target must differ from normal target")
    result["normal_correct"] = result["normal"] == result["expected"]
    result["paraphrase_correct"] = result["paraphrase"] == result["expected"]
    result["counterfactual_correct"] = result["counterfactual"] == result["counterfactual_expected"]
    result["zero_recreates_normal"] = result["zero"] == result["expected"]
    result["shuffle_recreates_normal"] = result["shuffled"] == result["expected"]
    result["wrong_query_recreates_normal"] = result["wrong_query"] == result["expected"]
    has_codebook_control = "codebook_swap" in result
    if has_codebook_control:
        if result.get("codebook_swap_expected") == result["expected"]:
            raise ValueError("codebook intervention must alter the expected code")
        result["codebook_swap_correct"] = result["codebook_swap"] == result["codebook_swap_expected"]
        result["codebook_swap_recreates_normal"] = result["codebook_swap"] == result["expected"]
    result["strict_causal"] = bool(
        result["normal_correct"] and result["paraphrase_correct"] and result["counterfactual_correct"]
        and not result["zero_recreates_normal"] and not result["shuffle_recreates_normal"]
        and not result["wrong_query_recreates_normal"]
        and (not has_codebook_control or (result["codebook_swap_correct"] and not result["codebook_swap_recreates_normal"]))
    )
    return result


def summarize_groups(results: list[dict], group_keys: list[str]) -> dict:
    by_group: dict[str, list[dict]] = {key: [] for key in group_keys}
    for result in results:
        by_group[result["basis_id"]].append(result)
    required = ["normal_correct", "paraphrase_correct", "counterfactual_correct", "strict_causal"]
    if results and all("codebook_swap_correct" in row for row in results):
        required.append("codebook_swap_correct")
    summary = {"groups": len(group_keys)}
    for field in required:
        summary["joint_" + field.removesuffix("_correct").removesuffix("_causal")] = sum(
            all(bool(row[field]) for row in by_group[key]) for key in group_keys
        )
    controls = ["zero_recreates_normal", "shuffle_recreates_normal", "wrong_query_recreates_normal"]
    if results and all("codebook_swap_recreates_normal" in row for row in results):
        controls.append("codebook_swap_recreates_normal")
    for field in controls:
        summary["any_" + field] = sum(any(bool(row[field]) for row in by_group[key]) for key in group_keys)
    if any(len(by_group[key]) != len(QUERY_KINDS) for key in group_keys):
        raise ValueError("result set lost a FQRB consumer")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--max-new", type=int, default=12)
    parser.add_argument("--allow-raw", action="store_true")
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--tape-len", type=int, default=0)
    parser.add_argument("--source-anchor", default="\nEnd state record:")
    args = parser.parse_args()
    if not torch.cuda.is_available() or Path(args.out).exists():
        raise SystemExit("CUDA required and output path must be fresh")
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to("cuda").eval()
    model.load_state_dict(checkpoint["model"])
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise SystemExit("tokenizer EOS missing")
    metadata = checkpoint.get("counterfactual_residual_algebra")
    if isinstance(metadata, dict):
        if metadata.get("source_present_at_suffix") is not False:
            raise SystemExit("checkpoint does not certify source-free residual algebra")
        if metadata.get("extra_trainable_parameters") != 0 or metadata.get("composition") != "donor + edited - base":
            raise SystemExit("checkpoint does not certify native residual composition")
        layer, tape_len = int(metadata["layer"]), int(metadata["tape_len"])
        source_window = int(metadata.get("source_window", 0))
    elif args.allow_raw:
        anchor_ids = tokenizer.encode(args.source_anchor).ids
        tape_len = args.tape_len or len(anchor_ids)
        if not anchor_ids or tape_len != len(anchor_ids):
            raise SystemExit("raw FQRB baseline requires the exact source anchor tape")
        layer = int(args.layer)
        source_window = 0
        metadata = {
            "raw_baseline": True, "layer": layer, "tape_len": tape_len,
            "source_anchor": args.source_anchor, "source_present_at_suffix": False,
            "extra_trainable_parameters": 0, "composition": "donor + edited - base",
        }
    else:
        raise SystemExit("checkpoint does not certify FQRB-compatible residual algebra")
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    groups = group_rows(rows, args.split, args.max_groups)
    group_keys = [basis_id for basis_id, _ in groups]
    shuffled_from = shifted_group_keys(group_keys)
    tapes: dict[str, torch.Tensor] = {}
    paraphrase_tapes: dict[str, torch.Tensor] = {}
    results: list[dict] = []
    with torch.no_grad():
        for index, (basis_id, group) in enumerate(groups, 1):
            tape, paraphrase_tape, base, donor, secondary = compose_group_tapes(
                model, tokenizer, group, layer, tape_len, source_window,
            )
            tapes[basis_id], paraphrase_tapes[basis_id] = tape, paraphrase_tape
            cosine = float(torch.nn.functional.cosine_similarity(
                tape.float().reshape(1, -1), paraphrase_tape.float().reshape(1, -1), dim=-1,
            ).item())
            for query_index, row in enumerate(group):
                counter_tape = compose_counter_tape(
                    model, tokenizer, row, base, donor, secondary, layer, tape_len, source_window,
                )
                wrong_row = group[(query_index + 1) % len(group)]
                tape_start_pos = source_window - tape_len if source_window else 0
                result = {
                    "basis_id": basis_id, "episode_id": row["episode_id"], "query_kind": row["query_kind"],
                    "normal": decode_tape(model, tokenizer, tape, row["suffix_prompt"], layer, tape_len, eos_id, args.max_new, tape_start_pos),
                    "paraphrase": decode_tape(model, tokenizer, paraphrase_tape, row["suffix_prompt"], layer, tape_len, eos_id, args.max_new, tape_start_pos),
                    "counterfactual": decode_tape(model, tokenizer, counter_tape, row["suffix_prompt"], layer, tape_len, eos_id, args.max_new, tape_start_pos),
                    "zero": decode_tape(model, tokenizer, torch.zeros_like(tape), row["suffix_prompt"], layer, tape_len, eos_id, args.max_new, tape_start_pos),
                    "wrong_query": decode_tape(model, tokenizer, tape, wrong_row["suffix_prompt"], layer, tape_len, eos_id, args.max_new, tape_start_pos),
                    "expected": row["response"], "counterfactual_expected": row["counterfactual_response"],
                    "same_tape_cosine": cosine,
                }
                if "codebook_swap_suffix_prompt" in row:
                    swap_expected = row.get("codebook_swap_response")
                    if not isinstance(swap_expected, str):
                        raise ValueError("codebook row lacks a swap response")
                    result["codebook_swap"] = decode_tape(
                        model, tokenizer, tape, row["codebook_swap_suffix_prompt"], layer, tape_len, eos_id, args.max_new, tape_start_pos,
                    )
                    result["codebook_swap_expected"] = swap_expected
                results.append(result)
            print("[fqrb-eval] encoded group {}/{}".format(index, len(groups)), flush=True)
        for result in results:
            result["shuffled"] = decode_tape(
                model, tokenizer, tapes[shuffled_from[result["basis_id"]]],
                next(row for basis_id, group in groups if basis_id == result["basis_id"] for row in group if row["query_kind"] == result["query_kind"])["suffix_prompt"],
                layer, tape_len, eos_id, args.max_new, source_window - tape_len if source_window else 0,
            )
    for result in results:
        score_result(result)
    fields = [
        "normal_correct", "paraphrase_correct", "counterfactual_correct", "zero_recreates_normal",
        "shuffle_recreates_normal", "wrong_query_recreates_normal", "strict_causal",
    ]
    if results and all("codebook_swap_correct" in row for row in results):
        fields.extend(("codebook_swap_correct", "codebook_swap_recreates_normal"))
    consumer_summary = {
        kind: {field: sum(bool(row[field]) for row in results if row["query_kind"] == kind) for field in fields}
        for kind in QUERY_KINDS
    }
    report = {
        "audit": "finite_query_residual_basis_v1", "checkpoint": args.ckpt, "step": checkpoint.get("step"),
        "checkpoint_metadata": metadata, "data": args.data, "data_sha256": sha256_file(args.data),
        "split": args.split, "groups": len(groups), "rows": len(results),
        "summary": {field: sum(bool(row[field]) for row in results) for field in fields},
        "consumer_summary": consumer_summary, "basis_summary": summarize_groups(results, group_keys),
        "mean_same_tape_cosine": sum(row["same_tape_cosine"] for row in results) / len(results),
        "results": results,
        "claim_boundary": "A strict finite-query group pass establishes only a bounded source-free latent basis; an ephemeral-codebook pass additionally tests late-bound query use, not general reasoning.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("[fqrb-eval] summary=" + json.dumps(report["summary"], sort_keys=True), flush=True)
    print("[fqrb-eval] basis=" + json.dumps(report["basis_summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
