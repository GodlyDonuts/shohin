#!/usr/bin/env python3
"""Train the five frozen, equal-budget S9.2 global-anchor arms."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time
from typing import Literal, Sequence

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from s7_learned_cayley_generator import LearnedCayleyGenerator
from s8_nil_linked_graph_compiler import (
    compile_row as compile_s8_row,
    recode_operation_ids,
)
from s9_occurrence_quotient_compiler import (
    MAX_SPAN_WIDTH,
    OccurrenceQuotientCompiler,
    S9Example,
    SpanCandidate,
    adapter_hash,
    adapter_state,
    compile_row as compile_s9_row,
    compiler_loss,
    load_adapter_state,
    load_examples,
    make_batches,
    pad_batch,
    sha256_file,
)
from s9_1_alpha_closed_compiler import (
    aligned_positive_logits,
    orbit_consistency_loss,
)
from s9_2_global_anchor_compiler import alpha_orbit_consistency_loss
from train_s8_nil_linked_graph import _fit_generator


PAIRED_SOURCE_ROWS = 24_000
CHARGED_VIEWS = 48_000
BATCH_SIZE = 64
PAIR_BATCH_SIZE = BATCH_SIZE // 2
UPDATES = 750
NEGATIVE_CANDIDATES = 128
HARD_NEGATIVE_TOP_K = 8
ORBIT_WEIGHT = 0.25
COMPLETE_SYSTEM_PARAMETERS = 134_580_264
CHECKPOINT_SCHEMA = "r12_s9_2_global_anchor_checkpoint_v1"
FROZEN_LAYER = 19
FROZEN_WIDTH = 384
FROZEN_HEADS = 8
FROZEN_ENCODER_LAYERS = 5
FROZEN_FF = 1408
FROZEN_LR = 1e-3
FROZEN_WARMUP = 50
FROZEN_CLIP = 1.0
FROZEN_SOURCE_PATHS = (
    "pipeline/build_s8_nil_linked_law_graph_board.py",
    "pipeline/s8_nil_linked_law_graph.py",
    "pipeline/s9_occurrence_quotient.py",
    "pipeline/s9_occurrence_quotient_falsifier.py",
    "train/model.py",
    "train/s7_learned_cayley_generator.py",
    "train/s8_nil_linked_graph_compiler.py",
    "train/s9_occurrence_quotient_compiler.py",
    "train/s9_1_alpha_closed_compiler.py",
    "train/s9_2_global_anchor_compiler.py",
    "train/train_s8_nil_linked_graph.py",
    "train/train_s9_2_global_anchor.py",
    "train/eval_s9_2_global_anchor.py",
    "train/assess_s9_2_global_anchor.py",
    "train/jobs/run_s9_2_global_anchor_development.sbatch",
)

OrbitMode = Literal["full", "positive"]


@dataclass(frozen=True)
class ArmSpec:
    name: str
    orbit_mode: OrbitMode
    class_messages: bool
    mask_gold_tokens: bool = False


ARM_SPECS = (
    ArmSpec("treatment", "full", True),
    ArmSpec("positive_orbit_only", "positive", True),
    ArmSpec("no_class", "full", False),
    ArmSpec("shuffled", "full", True),
    ArmSpec("layout_only", "full", False, mask_gold_tokens=True),
)


@dataclass(frozen=True)
class PairedOrbitSelection:
    """Aligned original/recoded views after variable candidate sampling."""

    original_examples: tuple[S9Example, ...]
    recoded_examples: tuple[S9Example, ...]
    original_candidate_rows: tuple[tuple[SpanCandidate, ...], ...]
    recoded_candidate_rows: tuple[tuple[SpanCandidate, ...], ...]
    original_logits: torch.Tensor
    recoded_logits: torch.Tensor


def lr_scale(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def verify_runtime_source(
    repo_root: Path,
    expected_commit: str,
    paths: Sequence[str] = FROZEN_SOURCE_PATHS,
) -> str:
    """Verify that every scientific runtime byte still equals the frozen commit."""

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("git", *arguments),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )

    resolved = git("rev-parse", "--verify", f"{expected_commit}^{{commit}}")
    if resolved.returncode:
        raise RuntimeError("S9.2 source commit is not locally available")
    resolved_sha = resolved.stdout.strip()
    if git("merge-base", "--is-ancestor", resolved_sha, "HEAD").returncode:
        raise RuntimeError("S9.2 source commit is not an ancestor of HEAD")
    for path in paths:
        if git("cat-file", "-e", f"{resolved_sha}:{path}").returncode:
            raise RuntimeError(f"S9.2 source commit omits frozen path: {path}")
    if git("diff", "--quiet", resolved_sha, "--", *paths).returncode:
        raise RuntimeError("S9.2 scientific runtime bytes differ from source commit")
    return resolved_sha


def paired_source_indices(
    row_count: int,
    seed: int,
    *,
    unique_sources: int = PAIRED_SOURCE_ROWS,
) -> tuple[int, ...]:
    """Select the frozen unique-source side of the paired training board."""

    if row_count < unique_sources or unique_sources <= 0:
        raise ValueError("S9.2 paired source budget is not available")
    indices = list(range(row_count))
    random.Random(seed).shuffle(indices)
    selected = tuple(indices[:unique_sources])
    if len(set(selected)) != unique_sources:
        raise RuntimeError("S9.2 paired source selection contains duplicates")
    return selected


def build_paired_source_split(
    examples: Sequence[S9Example],
    tokenizer: Tokenizer,
    seed: int,
) -> tuple[list[S9Example], list[S9Example], str]:
    """Select originals and construct exactly one operation recode per source."""

    indices = paired_source_indices(len(examples), seed)
    original = [examples[index] for index in indices]
    recoded = [
        compile_s9_row(
            recode_operation_ids(compile_s8_row(example.row, tokenizer), tokenizer).row,
            tokenizer,
        )
        for example in original
    ]
    for first, second in zip(original, recoded, strict=True):
        first_targets = tuple(target for _, _, target in first.gold)
        second_targets = tuple(target for _, _, target in second.gold)
        if first_targets != second_targets:
            raise RuntimeError("S9.2 recoding changed relation occurrence order")
        if max(end - start + 1 for start, end, _ in second.gold) > MAX_SPAN_WIDTH:
            raise RuntimeError("S9.2 recoding exceeded the proposal cap")
    digest = hashlib.sha256(
        json.dumps(indices, separators=(",", ":")).encode()
    ).hexdigest()
    return original, recoded, digest


def paired_consistent_shuffle(
    original: Sequence[S9Example],
    recoded: Sequence[S9Example],
    seed: int,
) -> tuple[list[S9Example], list[S9Example]]:
    """Apply one role permutation per source identically to both orbit views."""

    if len(original) != len(recoded):
        raise ValueError("S9.2 shuffled pair lengths differ")
    rng = random.Random(seed)
    first_result = []
    second_result = []
    for first, second in zip(original, recoded, strict=True):
        first_labels = [target for _, _, target in first.gold]
        second_labels = [target for _, _, target in second.gold]
        if first_labels != second_labels:
            raise ValueError("S9.2 shuffled pair labels are not initially aligned")
        permutation = list(range(len(first_labels)))
        rng.shuffle(permutation)
        labels = [first_labels[index] for index in permutation]
        first_result.append(
            replace(
                first,
                gold=tuple(
                    (start, end, labels[index])
                    for index, (start, end, _) in enumerate(first.gold)
                ),
            )
        )
        second_result.append(
            replace(
                second,
                gold=tuple(
                    (start, end, labels[index])
                    for index, (start, end, _) in enumerate(second.gold)
                ),
            )
        )
    return first_result, second_result


def mask_gold_span_tokens(
    ids: torch.Tensor,
    examples: Sequence[S9Example],
    *,
    mask_token_id: int = 0,
) -> torch.Tensor:
    """Return a copy with every supervised span token replaced by token zero."""

    if ids.ndim != 2 or ids.shape[0] != len(examples):
        raise ValueError("S9.2 layout mask batch shape mismatch")
    if mask_token_id != 0:
        raise ValueError("S9.2 frozen layout mask token must be zero")
    masked = ids.clone()
    for row, example in enumerate(examples):
        for start, end, _ in example.gold:
            if start < 0 or end < start or end >= len(example.ids):
                raise ValueError("S9.2 gold span is outside its token sequence")
            masked[row, start : end + 1] = mask_token_id
    return masked


def split_paired_orbit_selection(
    examples: Sequence[S9Example],
    candidate_rows: Sequence[Sequence[SpanCandidate]],
    logits: torch.Tensor,
    pair_count: int,
) -> PairedOrbitSelection:
    """Split paired model outputs without assuming equal candidate counts."""

    if pair_count <= 0 or len(examples) != 2 * pair_count:
        raise ValueError("S9.2 paired example split is invalid")
    if len(candidate_rows) != len(examples):
        raise ValueError("S9.2 paired candidate split is invalid")
    original_width = sum(len(row) for row in candidate_rows[:pair_count])
    total_width = sum(len(row) for row in candidate_rows)
    if logits.ndim != 2 or logits.shape[0] != total_width:
        raise ValueError("S9.2 paired logits do not match candidate rows")
    return PairedOrbitSelection(
        original_examples=tuple(examples[:pair_count]),
        recoded_examples=tuple(examples[pair_count:]),
        original_candidate_rows=tuple(
            tuple(row) for row in candidate_rows[:pair_count]
        ),
        recoded_candidate_rows=tuple(tuple(row) for row in candidate_rows[pair_count:]),
        original_logits=logits[:original_width],
        recoded_logits=logits[original_width:],
    )


def select_orbit_loss(
    selection: PairedOrbitSelection,
    mode: OrbitMode,
    *,
    top_k: int = HARD_NEGATIVE_TOP_K,
) -> torch.Tensor:
    """Select the frozen S9.2 full or S9.1 positive-only orbit objective."""

    if mode == "full":
        return alpha_orbit_consistency_loss(
            selection.original_examples,
            selection.recoded_examples,
            selection.original_candidate_rows,
            selection.recoded_candidate_rows,
            selection.original_logits,
            selection.recoded_logits,
            top_k=top_k,
        )
    if mode != "positive":
        raise ValueError(f"unknown S9.2 orbit mode: {mode}")
    original_positive, original_targets = aligned_positive_logits(
        selection.original_examples,
        selection.original_candidate_rows,
        selection.original_logits,
    )
    recoded_positive, recoded_targets = aligned_positive_logits(
        selection.recoded_examples,
        selection.recoded_candidate_rows,
        selection.recoded_logits,
    )
    return orbit_consistency_loss(
        original_positive,
        recoded_positive,
        original_targets,
        recoded_targets,
    )


def _require_report_hash(
    path: Path,
    report: dict[str, object],
    name: str,
) -> None:
    files = report.get("files")
    if not isinstance(files, dict) or name not in files:
        raise SystemExit(f"S9.2 board report omits {name}")
    if sha256_file(path) != files[name]["sha256"]:
        raise SystemExit(f"S9.2 {name} hash mismatch")


def _rows(path: Path, report: dict[str, object], name: str) -> list[dict]:
    _require_report_hash(path, report, name)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _new_compiler(cfg, base_state, initializer, args):
    model = GPT(cfg).to("cuda").eval()
    model.load_state_dict(base_state)
    compiler = OccurrenceQuotientCompiler(
        model,
        layer=args.layer,
        width=args.width,
        heads=args.heads,
        encoder_layers=args.encoder_layers,
        ff=args.ff,
    ).to("cuda")
    loaded = compiler.initialize_memory_encoder(initializer["treatment_adapter_state"])
    return compiler, loaded


def _fit(
    compiler,
    original,
    recoded,
    args,
    seed,
    spec: ArmSpec,
):
    trainable = list(compiler.adapter_parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    batches = make_batches(original, PAIR_BATCH_SIZE, seed)
    if len(batches) != UPDATES or any(
        len(batch) != PAIR_BATCH_SIZE for batch in batches
    ):
        raise RuntimeError("S9.2 frozen pair/update budget changed")
    started = time.time()
    final = {}
    compiler.train()
    compiler.model.eval()
    for step, indices in enumerate(batches):
        first = [original[index] for index in indices]
        second = [recoded[index] for index in indices]
        combined = first + second
        selected, candidate_rows, ids, valid, candidates = pad_batch(
            combined,
            list(range(len(combined))),
            "cuda",
            negative_limit=args.negative_candidates,
            seed=seed ^ step,
        )
        if spec.mask_gold_tokens:
            ids = mask_gold_span_tokens(ids, selected)
        optimizer.param_groups[0]["lr"] = args.lr * lr_scale(
            step, len(batches), args.warmup
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = compiler(
                ids,
                valid,
                candidates,
                class_messages=spec.class_messages,
            )
            supervised = compiler_loss(outputs, candidates["target"])
            selection = split_paired_orbit_selection(
                selected,
                candidate_rows,
                outputs["role_logits"],
                len(first),
            )
            orbit = select_orbit_loss(
                selection,
                spec.orbit_mode,
                top_k=args.hard_negative_top_k,
            )
            loss = supervised + args.orbit_weight * orbit
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite S9.2 {spec.name} loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite S9.2 {spec.name} gradient")
        optimizer.step()
        predictions = outputs["role_logits"].argmax(-1)
        positive = candidates["target"] != 0
        final = {
            "loss": float(loss.item()),
            "supervised_loss": float(supervised.item()),
            "orbit_loss": float(orbit.item()),
            "grad_norm": float(grad_norm.item()),
            "candidate_accuracy": float(
                (predictions == candidates["target"]).float().mean().item()
            ),
            "positive_accuracy": float(
                (predictions[positive] == candidates["target"][positive])
                .float()
                .mean()
                .item()
            ),
        }
        if step % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "arm": spec.name,
                        "update": step,
                        **final,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return {
        "updates": len(batches),
        "unique_sources": len(original),
        "charged_views": 2 * len(original),
        "batch_size": args.batch_size,
        "elapsed_seconds": time.time() - started,
        "class_messages": spec.class_messages,
        "mask_gold_tokens": spec.mask_gold_tokens,
        "mask_token_id": 0 if spec.mask_gold_tokens else None,
        "orbit_mode": spec.orbit_mode,
        "orbit_weight": args.orbit_weight,
        "learning_rate": args.lr,
        "warmup_updates": args.warmup,
        "gradient_clip": args.clip,
        "hard_negative_top_k": (
            args.hard_negative_top_k if spec.orbit_mode == "full" else None
        ),
        "negative_candidates_per_view": args.negative_candidates,
        "final": final,
        "adapter_sha256": adapter_hash(compiler),
    }


def _validate_frozen_args(args) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("S9.2 training requires CUDA")
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    if args.out.exists() or partial.exists():
        raise SystemExit(f"refusing existing S9.2 checkpoint custody: {args.out}")
    if (
        args.batch_size != BATCH_SIZE
        or args.negative_candidates != NEGATIVE_CANDIDATES
        or args.hard_negative_top_k != HARD_NEGATIVE_TOP_K
        or args.orbit_weight != ORBIT_WEIGHT
        or args.layer != FROZEN_LAYER
        or args.width != FROZEN_WIDTH
        or args.heads != FROZEN_HEADS
        or args.encoder_layers != FROZEN_ENCODER_LAYERS
        or args.ff != FROZEN_FF
        or args.lr != FROZEN_LR
        or args.warmup != FROZEN_WARMUP
        or args.clip != FROZEN_CLIP
    ):
        raise SystemExit("S9.2 frozen architecture or optimization contract changed")


def checkpoint_arm_payload(arm_results: dict[str, dict]) -> dict[str, object]:
    """Materialize the frozen evaluator-facing five-arm checkpoint schema."""

    required = {spec.name for spec in ARM_SPECS}
    if set(arm_results) != required:
        raise ValueError("S9.2 checkpoint arm set is incomplete")
    aliases = {
        "treatment": "treatment",
        "positive_only": "positive_orbit_only",
        "no_class": "no_class",
        "shuffled": "shuffled",
        "layout": "layout_only",
    }
    payload: dict[str, object] = {
        "fit": {
            alias: arm_results[source]["fit"]
            for alias, source in aliases.items()
        }
    }
    for alias, source in aliases.items():
        payload[f"{alias}_adapter_state"] = arm_results[source]["adapter_state"]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--layer", type=int, default=FROZEN_LAYER)
    parser.add_argument("--width", type=int, default=FROZEN_WIDTH)
    parser.add_argument("--heads", type=int, default=FROZEN_HEADS)
    parser.add_argument("--encoder-layers", type=int, default=FROZEN_ENCODER_LAYERS)
    parser.add_argument("--ff", type=int, default=FROZEN_FF)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--negative-candidates", type=int, default=NEGATIVE_CANDIDATES)
    parser.add_argument("--hard-negative-top-k", type=int, default=HARD_NEGATIVE_TOP_K)
    parser.add_argument("--orbit-weight", type=float, default=ORBIT_WEIGHT)
    parser.add_argument("--lr", type=float, default=FROZEN_LR)
    parser.add_argument("--warmup", type=int, default=FROZEN_WARMUP)
    parser.add_argument("--clip", type=float, default=FROZEN_CLIP)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    _validate_frozen_args(args)
    source_commit = verify_runtime_source(
        Path(__file__).resolve().parents[1],
        args.source_commit,
    )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    report_path = args.data_dir / "report.json"
    report = json.loads(report_path.read_text())
    if report.get("schema") != "r12_s8_nil_linked_law_graph_board_report_v1":
        raise SystemExit("unexpected S9.2 board schema")
    if report.get("decision") != "admit_s8_nil_linked_law_graph_board":
        raise SystemExit("S9.2 board is not admitted")
    if report.get("source_commit") != source_commit:
        raise SystemExit("S9.2 board/source commit mismatch")
    audit = report.get("audit", {})
    if audit.get("development_accesses") or audit.get("confirmation_accesses"):
        raise SystemExit("S9.2 score board was already accessed")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**base["cfg"])
    initializer = torch.load(args.initializer, map_location="cpu", weights_only=False)
    if initializer.get("schema") != "r12_s8_nil_linked_law_graph_checkpoint_v1":
        raise SystemExit("S9.2 initializer is not the closed S8.1 checkpoint")
    if initializer.get("base_sha256") != sha256_file(args.base):
        raise SystemExit("S9.2 initializer/base mismatch")

    train_path = args.data_dir / "train.jsonl"
    _require_report_hash(train_path, report, "train.jsonl")
    examples = load_examples(
        train_path,
        tokenizer,
        "s8_nil_graph_train",
        cfg.seq_len,
    )
    if len(examples) != CHARGED_VIEWS:
        raise SystemExit("S9.2 requires the admitted 48,000-row source pool")
    original, recoded, subset_sha256 = build_paired_source_split(
        examples, tokenizer, args.seed ^ 0xA2FA
    )
    shuffled_seed = args.seed ^ 0x592A
    shuffled_original, shuffled_recoded = paired_consistent_shuffle(
        original, recoded, shuffled_seed
    )

    generator_rows = _rows(
        args.data_dir / "generator_train.jsonl",
        report,
        "generator_train.jsonl",
    )
    if len(generator_rows) != 23:
        raise SystemExit("S9.2 generator row count mismatch")
    generator = LearnedCayleyGenerator().to("cuda")
    generator_fit = _fit_generator(generator, generator_rows, "next_symbol")
    if (
        generator_fit["successor_accuracy"] != 1.0
        or generator_fit["zero_accuracy"] != 1.0
    ):
        raise RuntimeError("S9.2 generator failed exact fit")

    probe, loaded_reference = _new_compiler(cfg, base["model"], initializer, args)
    base_parameters = sum(value.numel() for value in probe.model.parameters())
    compiler_parameters = probe.adapter_num_params()
    initial_state = adapter_state(probe)
    initial_sha256 = adapter_hash(probe)
    del probe
    torch.cuda.empty_cache()

    parameters = {
        "base": base_parameters,
        "compiler": compiler_parameters,
        "generator": sum(value.numel() for value in generator.parameters()),
    }
    parameters["complete_system"] = sum(parameters.values())
    if parameters["complete_system"] != COMPLETE_SYSTEM_PARAMETERS:
        raise RuntimeError(
            "S9.2 architecture changed: expected exactly "
            f"{COMPLETE_SYSTEM_PARAMETERS}, got {parameters['complete_system']}"
        )

    arm_results = {}
    loaded_controls = {}
    for spec in ARM_SPECS:
        compiler, loaded = _new_compiler(cfg, base["model"], initializer, args)
        load_adapter_state(compiler, initial_state)
        if adapter_hash(compiler) != initial_sha256:
            raise RuntimeError(f"S9.2 {spec.name} initialization diverged")
        loaded_controls[spec.name] = loaded
        arm_original = shuffled_original if spec.name == "shuffled" else original
        arm_recoded = shuffled_recoded if spec.name == "shuffled" else recoded
        fit = _fit(
            compiler,
            arm_original,
            arm_recoded,
            args,
            args.seed,
            spec,
        )
        arm_results[spec.name] = {
            "spec": {
                "orbit_mode": spec.orbit_mode,
                "class_messages": spec.class_messages,
                "mask_gold_tokens": spec.mask_gold_tokens,
            },
            "fit": fit,
            "adapter_state": adapter_state(compiler),
        }
        del compiler
        torch.cuda.empty_cache()

    if any(value != loaded_reference for value in loaded_controls.values()):
        raise RuntimeError("S9.2 initializer loading differed across arms")
    if any(
        result["fit"]["updates"] != UPDATES
        or result["fit"]["unique_sources"] != PAIRED_SOURCE_ROWS
        or result["fit"]["charged_views"] != CHARGED_VIEWS
        for result in arm_results.values()
    ):
        raise RuntimeError("S9.2 arm budget equality check failed")

    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "source_commit": source_commit,
        "seed": args.seed,
        "base_sha256": sha256_file(args.base),
        "initializer_sha256": sha256_file(args.initializer),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "board_report_sha256": sha256_file(report_path),
        "architecture": {
            "compiler_class": "OccurrenceQuotientCompiler",
            "layer": args.layer,
            "width": args.width,
            "heads": args.heads,
            "encoder_layers": args.encoder_layers,
            "ff": args.ff,
            "max_span_width": MAX_SPAN_WIDTH,
            "negative_candidates_per_view": args.negative_candidates,
            "hard_negative_top_k": args.hard_negative_top_k,
            "added_trainable_parameters": 0,
        },
        "parameters": parameters,
        "training_contract": {
            "unique_sources_per_arm": PAIRED_SOURCE_ROWS,
            "charged_views_per_arm": CHARGED_VIEWS,
            "batch_size": args.batch_size,
            "pair_batch_size": PAIR_BATCH_SIZE,
            "updates_per_arm": UPDATES,
            "negative_candidates_per_view": args.negative_candidates,
            "hard_negative_top_k": args.hard_negative_top_k,
            "orbit_weight": args.orbit_weight,
            "learning_rate": args.lr,
            "warmup_updates": args.warmup,
            "gradient_clip": args.clip,
            "supervision": "graph fields only",
            "forbidden_supervision": (
                "final state, answer, recurrence, development laws, confirmation laws"
            ),
        },
        "paired_source_indices_sha256": subset_sha256,
        "shuffled_seed": shuffled_seed,
        "initial_adapter_sha256": initial_sha256,
        "initializer_loaded": loaded_reference,
        "initializer_control_equal": True,
        "generator_fit": generator_fit,
        "generator_state": generator.state_dict(),
        "arm_order": [spec.name for spec in ARM_SPECS],
        **checkpoint_arm_payload(arm_results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    torch.save(checkpoint, partial)
    if args.out.exists():
        raise RuntimeError("S9.2 checkpoint appeared during atomic write")
    partial.replace(args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "schema": CHECKPOINT_SCHEMA,
                "parameters": parameters,
                "paired_source_indices_sha256": subset_sha256,
                "arms": {name: value["fit"] for name, value in arm_results.items()},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
