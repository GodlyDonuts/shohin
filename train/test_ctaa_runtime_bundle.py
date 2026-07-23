from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

import ctaa_runtime_bundle as bundle
from ctaa_run_contract import ARMS, DATASETS, RUN_CONTRACT_SCHEMA, canonical_sha256


def digest(label: object) -> str:
    import hashlib

    return hashlib.sha256(str(label).encode()).hexdigest()


def contract() -> dict[str, object]:
    seeds = [101, 202, 303, 404, 505]
    runs = []
    for seed in seeds:
        for arm in ARMS:
            for dataset in DATASETS:
                core_kind = (
                    "outer_product_control"
                    if arm == "oprc_closure"
                    else "closure_feature"
                )
                runs.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "dataset": dataset,
                        "compiler_sha256": digest((seed, arm, "compiler")),
                        "raw_evidence_receipt_sha256": digest(
                            (seed, arm, dataset, "receipt")
                        ),
                        "core_training": {
                            "core_sha256": digest((seed, arm, "core")),
                            "core_kind": core_kind,
                        },
                    }
                )
    value: dict[str, object] = {
        "schema": RUN_CONTRACT_SCHEMA,
        "partition": "development",
        "manifest_sha256": digest("manifest"),
        "board_sha256": digest("board"),
        "run_plan_sha256": digest("run-plan"),
        "bootstrap_seed_receipt_sha256": digest("bootstrap"),
        "bootstrap_seed": 77,
        "training_seeds": seeds,
        "arms": list(ARMS),
        "datasets": list(DATASETS),
        "run_count": 40,
        "oracle_files": {},
        "runs": runs,
    }
    value["run_contract_sha256"] = canonical_sha256(value)
    return value


def plan(seed: int, run_contract: dict[str, object]) -> SimpleNamespace:
    row = next(
        item
        for item in run_contract["runs"]  # type: ignore[union-attr]
        if item["seed"] == seed
        and item["arm"] == "ctaa_closure"
        and item["dataset"] == "base"
    )
    training = row["core_training"]
    bindings = SimpleNamespace(
        board_manifest_sha256=run_contract["manifest_sha256"],
        board_tree_sha256=run_contract["board_sha256"],
        run_contract_sha256=run_contract["run_contract_sha256"],
        selection_seed=909,
        selection_seed_receipt_sha256=digest("selection-receipt"),
        arm_id="ctaa_closure",
        tokenizer_sha256=digest("tokenizer"),
        base_checkpoint_sha256=digest("checkpoint"),
        runtime_implementation_sha256=digest("runtime"),
        batch_order_sha256=digest("batch-order"),
        training_seed=seed,
        compiler_sha256=row["compiler_sha256"],
        core_sha256=training["core_sha256"],
        core_kind=training["core_kind"],
        base_raw_evidence_receipt_sha256=row["raw_evidence_receipt_sha256"],
        partition=SimpleNamespace(value="development"),
    )
    return SimpleNamespace(
        bindings=bindings,
        anchor_panel_sha256=digest("anchor-panel"),
        donor_registry_sha256=digest("donor-registry"),
        plan_sha256=digest((seed, "plan")),
    )


def artifacts(run_contract: dict[str, object]) -> list[tuple[object, ...]]:
    result = []
    for seed in run_contract["training_seeds"]:  # type: ignore[union-attr]
        frozen = plan(seed, run_contract)
        evidence = {
            "training_seed": seed,
            "compiler_sha256": frozen.bindings.compiler_sha256,
            "core_sha256": frozen.bindings.core_sha256,
            "core_kind": frozen.bindings.core_kind,
            "base_raw_evidence_receipt_sha256": (
                frozen.bindings.base_raw_evidence_receipt_sha256
            ),
            "evidence_sha256": digest((seed, "evidence")),
        }
        result.append(
            (
                frozen,
                evidence,
                f"runtime-plan-{seed}.json",
                digest((seed, "plan-file")),
                f"runtime-evidence-{seed}.json",
                digest((seed, "evidence-file")),
            )
        )
    return result


@pytest.fixture(autouse=True)
def bypass_upstream_validators(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bundle, "validate_runtime_intervention_plan", lambda value: value
    )
    monkeypatch.setattr(
        bundle, "validate_runtime_evidence", lambda value, _plan: dict(value)
    )


def test_bundle_is_exactly_five_treatment_seeds_and_not_scored() -> None:
    run_contract = contract()
    value = bundle.make_runtime_bundle(
        run_contract=run_contract, artifacts=artifacts(run_contract)
    )
    assert [row["training_seed"] for row in value["entries"]] == [
        101,
        202,
        303,
        404,
        505,
    ]
    assert value["seed_count"] == 5
    assert value["runtime_panel_size_per_seed"] == 864
    assert value["operation_count"] == 29
    assert value["attempt_count_per_seed"] == 25_056
    assert value["scored_row_count"] == 40_608
    assert value["runtime_attempts_affect_scored_denominator"] is False
    assert value["oracle_access"] == 0


def test_missing_or_duplicate_seed_is_rejected() -> None:
    run_contract = contract()
    values = artifacts(run_contract)
    with pytest.raises(bundle.RuntimeBundleError, match="seed coverage"):
        bundle.make_runtime_bundle(run_contract=run_contract, artifacts=values[:-1])
    with pytest.raises(bundle.RuntimeBundleError):
        bundle.make_runtime_bundle(
            run_contract=run_contract, artifacts=[*values[:-1], values[0]]
        )


def test_core_receipt_or_compiler_substitution_is_rejected() -> None:
    run_contract = contract()
    values = artifacts(run_contract)
    for attribute in (
        "core_sha256",
        "compiler_sha256",
        "base_raw_evidence_receipt_sha256",
    ):
        changed = list(values)
        original = changed[0][0]
        binding = SimpleNamespace(**vars(original.bindings))
        setattr(binding, attribute, digest((attribute, "forged")))
        changed[0] = (
            SimpleNamespace(**{**vars(original), "bindings": binding}),
            *changed[0][1:],
        )
        with pytest.raises(bundle.RuntimeBundleError, match="treatment run"):
            bundle.make_runtime_bundle(run_contract=run_contract, artifacts=changed)


def test_cross_seed_panel_or_implementation_substitution_is_rejected() -> None:
    run_contract = contract()
    values = artifacts(run_contract)
    for attribute in ("batch_order_sha256", "runtime_implementation_sha256"):
        changed = list(values)
        original = changed[-1][0]
        binding = SimpleNamespace(**vars(original.bindings))
        setattr(binding, attribute, digest((attribute, "forged")))
        changed[-1] = (
            SimpleNamespace(**{**vars(original), "bindings": binding}),
            *changed[-1][1:],
        )
        with pytest.raises(bundle.RuntimeBundleError, match="five-seed panel"):
            bundle.make_runtime_bundle(run_contract=run_contract, artifacts=changed)


def test_unsafe_or_repeated_member_names_are_rejected() -> None:
    run_contract = contract()
    values = artifacts(run_contract)
    changed = list(values)
    changed[0] = (changed[0][0], changed[0][1], "../plan.json", *changed[0][3:])
    with pytest.raises(bundle.RuntimeBundleError, match="unsafe"):
        bundle.make_runtime_bundle(run_contract=run_contract, artifacts=changed)
    changed = list(values)
    changed[1] = (
        changed[1][0],
        changed[1][1],
        changed[0][2],
        *changed[1][3:],
    )
    with pytest.raises(bundle.RuntimeBundleError, match="filename repeats"):
        bundle.make_runtime_bundle(run_contract=run_contract, artifacts=changed)


def test_entry_and_bundle_hash_mutations_fail_closed() -> None:
    run_contract = contract()
    value = bundle.make_runtime_bundle(
        run_contract=run_contract, artifacts=artifacts(run_contract)
    )
    changed = deepcopy(value)
    changed["entries"][0]["runtime_evidence_sha256"] = digest("forged")
    with pytest.raises(bundle.RuntimeBundleError, match="entry commitment"):
        bundle.validate_runtime_bundle(changed, run_contract=run_contract)
    changed = deepcopy(value)
    changed["bundle_sha256"] = digest("forged")
    with pytest.raises(bundle.RuntimeBundleError, match="bundle commitment"):
        bundle.validate_runtime_bundle(changed, run_contract=run_contract)


def test_entry_reorder_and_unknown_field_fail_closed() -> None:
    run_contract = contract()
    value = bundle.make_runtime_bundle(
        run_contract=run_contract, artifacts=artifacts(run_contract)
    )
    reordered = deepcopy(value)
    reordered["entries"][0], reordered["entries"][1] = (
        reordered["entries"][1],
        reordered["entries"][0],
    )
    with pytest.raises(bundle.RuntimeBundleError, match="order/coverage"):
        bundle.validate_runtime_bundle(reordered, run_contract=run_contract)
    extra = deepcopy(value)
    extra["entries"][0]["scientific_pass"] = True
    with pytest.raises(bundle.RuntimeBundleError, match="entry schema"):
        bundle.validate_runtime_bundle(extra, run_contract=run_contract)
