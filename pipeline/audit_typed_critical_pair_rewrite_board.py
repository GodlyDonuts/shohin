"""Emit an independently replayable receipt for TCRR CPU mechanics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from typed_critical_pair_rewrite_board import (
    IndependentNestedReferenceOracle,
    ProductionRewriteStateOracle,
    apply_reduction,
    build_mechanics_board,
    canonical_graph_serialization,
    legal_reductions,
    reindex_graph,
)


SOURCE_PATHS = (
    "pipeline/audit_typed_critical_pair_rewrite_board.py",
    "pipeline/test_audit_typed_critical_pair_rewrite_board.py",
    "pipeline/test_typed_critical_pair_rewrite_board.py",
    "pipeline/typed_critical_pair_rewrite_board.py",
)
REQUIRED_CLASSES = frozenset(
    (
        "capacity_unblocking_order",
        "confluent_diamond",
        "counterfactual_rhs_pointer",
        "destructive_cancellation",
        "heterogeneous_valid_typing",
        "independent_redexes",
        "mixed_cyclic_terminating",
        "nested_redex_creation_removal",
        "nonconfluent_fork",
        "repeated_rhs_pointer_sharing",
        "repeated_variable_binding",
        "root_deletion",
        "shared_occurrence_redexes",
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _assert_source_clean(root: Path) -> None:
    for arguments in (
        ("git", "diff", "--quiet", "--", *SOURCE_PATHS),
        ("git", "diff", "--cached", "--quiet", "--", *SOURCE_PATHS),
    ):
        result = subprocess.run(arguments, cwd=root, check=False)
        if result.returncode != 0:
            raise RuntimeError("TCRR mechanics source is not clean")


def _reachable_conservation(episode: Any) -> dict[str, int | bool]:
    frontier = [episode.initial_graph]
    states: dict[str, Any] = {}
    transitions = 0
    all_conserved = True
    while frontier:
        graph = frontier.pop()
        key = canonical_graph_serialization(graph)
        if key in states:
            continue
        states[key] = graph
        receipt = graph.conservation_receipt()
        all_conserved = all_conserved and bool(receipt["conserved"])
        for reduction in legal_reductions(episode.system, graph):
            successor = apply_reduction(
                episode.system,
                graph,
                reduction,
            )
            all_conserved = (
                all_conserved
                and successor.capacity == graph.capacity
                and bool(successor.conservation_receipt()["conserved"])
            )
            transitions += 1
            frontier.append(successor)
    return {
        "states": len(states),
        "transitions_with_multiplicity": transitions,
        "all_conserved": all_conserved,
    }


def build_audit(
    *,
    root: Path,
    seed: int,
    require_clean: bool = True,
) -> dict[str, Any]:
    if require_clean:
        _assert_source_clean(root)
    source_commit = _git_head(root)
    sources = {
        relative: _sha256(root / relative)
        for relative in SOURCE_PATHS
    }
    episodes = build_mechanics_board(seed)
    rows = []
    production_state_total = 0
    production_transition_total = 0
    normal_form_total = 0
    cyclic_component_total = 0
    all_agree = True
    all_reindexed = True
    all_conserved = True
    for episode in episodes:
        production = ProductionRewriteStateOracle().enumerate(
            episode.system,
            episode.initial_graph,
        )
        reference = IndependentNestedReferenceOracle().enumerate(
            episode.system,
            episode.initial_graph,
        )
        agreement = (
            production.normal_forms == reference.normal_forms
            and production.transitions == reference.transitions
            and production.cyclic_sccs == reference.cyclic_sccs
            and production.cyclic_states == reference.cyclic_states
            and production.states_explored == reference.states_explored
            and production.transitions_explored
            == reference.transitions_explored
        )
        permutation = tuple(
            reversed(range(episode.initial_graph.capacity))
        )
        reindexed = ProductionRewriteStateOracle().enumerate(
            episode.system,
            reindex_graph(episode.initial_graph, permutation),
        )
        reindex_invariant = (
            production.normal_forms == reindexed.normal_forms
            and production.cyclic_sccs == reindexed.cyclic_sccs
        )
        conservation = _reachable_conservation(episode)
        all_agree = all_agree and agreement
        all_reindexed = all_reindexed and reindex_invariant
        all_conserved = all_conserved and bool(
            conservation["all_conserved"]
        )
        production_state_total += production.states_explored
        production_transition_total += production.transitions_explored
        normal_form_total += len(production.normal_forms)
        cyclic_component_total += len(production.cyclic_sccs)
        rows.append(
            {
                "name": episode.name,
                "episode_class": episode.episode_class,
                "capacity": episode.initial_graph.capacity,
                "production_reference_agreement": agreement,
                "storage_reindex_invariant": reindex_invariant,
                "normal_forms": len(production.normal_forms),
                "states": production.states_explored,
                "transitions": production.transitions_explored,
                "cyclic_components": len(production.cyclic_sccs),
                "conservation": conservation,
            }
        )
    classes = frozenset(row["episode_class"] for row in rows)
    decision = (
        "admit_cpu_mechanics_only"
        if (
            len(rows) == 14
            and classes == REQUIRED_CLASSES
            and all_agree
            and all_reindexed
            and all_conserved
        )
        else "reject_cpu_mechanics"
    )
    if {
        relative: _sha256(root / relative)
        for relative in SOURCE_PATHS
    } != sources:
        raise RuntimeError("TCRR mechanics source drifted during audit")
    return {
        "schema": "typed_critical_pair_rewrite_mechanics_audit_v1",
        "claim_boundary": (
            "This report admits deterministic CPU rewrite mechanics only. "
            "It contains no neural model, source-deleted tensorizer, learned "
            "reasoning result, Shohin integration, or general-reasoning claim."
        ),
        "decision": decision,
        "seed": seed,
        "source_commit": source_commit,
        "source_sha256": sources,
        "episode_count": len(rows),
        "episode_classes": sorted(classes),
        "production_reference_agreement": all_agree,
        "storage_reindex_invariant": all_reindexed,
        "all_reachable_states_conserve_capacity": all_conserved,
        "states": production_state_total,
        "transitions": production_transition_total,
        "normal_forms": normal_form_total,
        "cyclic_components": cyclic_component_total,
        "episodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = build_audit(root=root, seed=arguments.seed)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "episodes": report["episode_count"],
                "states": report["states"],
                "transitions": report["transitions"],
                "output": str(arguments.output),
                "sha256": _sha256(arguments.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
