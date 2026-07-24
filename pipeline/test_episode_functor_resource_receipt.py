from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import json

import pytest

import pipeline.episode_functor_resource_receipt as receipt_module
from pipeline.episode_functor_resource_receipt import (
    ArtifactBindings,
    PERSISTENT_BYTE_LIMIT,
    QualificationResourceReceipt,
    ResourceObservation,
    ResourceReceiptError,
    ResourceVector,
    TOTAL_PARAMETER_LIMIT_EXCLUSIVE,
    create_resource_receipt,
    load_resource_receipt,
    sha256_bytes,
)


def _observation(
    value: int,
    basis: str = "forecast",
) -> ResourceObservation:
    return ResourceObservation(value=value, basis=basis)


def _bindings() -> ArtifactBindings:
    return ArtifactBindings(
        board_sha256=sha256_bytes(b"qualification-board"),
        source_sha256=sha256_bytes(b"raw-source-corpus"),
        config_sha256=sha256_bytes(b"compiler-config"),
    )


def _resources(
    *,
    basis: str = "forecast",
    compiler_flops: ResourceObservation | None = None,
    compiler_time_ns: ResourceObservation | None = None,
) -> ResourceVector:
    if compiler_flops is None and compiler_time_ns is None:
        compiler_flops = _observation(9_000_000_000, basis)
    return ResourceVector(
        examples=_observation(40_000, basis),
        target_bits=_observation(12_800_000, basis),
        source_bytes=_observation(96_000_000, basis),
        oracle_calls=_observation(0, basis),
        updates=_observation(10_000, basis),
        trainable_parameters=_observation(24_000_000, basis),
        total_parameters=_observation(149_000_000, basis),
        optimizer_bytes=_observation(192_000_000, basis),
        compiler_flops=compiler_flops,
        compiler_time_ns=compiler_time_ns,
        persistent_bytes=_observation(PERSISTENT_BYTE_LIMIT, basis),
        executor_flops_per_query=_observation(24_576, basis),
    )


def _mapping_copy(
    receipt: QualificationResourceReceipt,
) -> dict[str, object]:
    return json.loads(receipt.to_json_bytes())


def test_forecast_receipt_freezes_full_prereg_vector_and_bindings() -> None:
    bindings = _bindings()
    receipt = create_resource_receipt(
        bindings=bindings,
        resources=_resources(),
    )
    mapping = receipt.to_mapping()

    assert receipt.receipt_kind == "forecast"
    assert mapping["bindings"] == bindings.to_mapping()
    assert mapping["limits"] == {
        "persistent_bytes_max": 1_536,
        "total_parameters_exclusive": 200_000_000,
    }
    assert set(mapping["resources"]) == {
        "compiler_flops",
        "compiler_time_ns",
        "examples",
        "executor_flops_per_query",
        "optimizer_bytes",
        "oracle_calls",
        "persistent_bytes",
        "source_bytes",
        "target_bits",
        "total_parameters",
        "trainable_parameters",
        "updates",
    }
    assert mapping["resources"]["compiler_time_ns"] is None
    assert mapping["resources"]["compiler_flops"] == {
        "basis": "forecast",
        "value": 9_000_000_000,
    }

    restored = load_resource_receipt(receipt.to_json_bytes())
    assert restored == receipt
    restored.assert_bindings(bindings)


def test_explicitly_measured_compiler_time_makes_measured_receipt() -> None:
    resources = _resources(
        basis="measured",
        compiler_flops=None,
        compiler_time_ns=_observation(12_500_000, "measured"),
    )
    receipt = create_resource_receipt(
        bindings=_bindings(),
        resources=resources,
    )

    assert receipt.receipt_kind == "measured"
    assert receipt.to_mapping()["resources"]["compiler_flops"] is None
    assert receipt.to_mapping()["resources"]["compiler_time_ns"] == {
        "basis": "measured",
        "value": 12_500_000,
    }


def test_mixed_provenance_is_reported_not_silently_coerced() -> None:
    resources = replace(
        _resources(),
        examples=_observation(40_000, "measured"),
    )
    receipt = create_resource_receipt(
        bindings=_bindings(),
        resources=resources,
    )
    assert receipt.receipt_kind == "mixed"


@pytest.mark.parametrize(
    ("compiler_flops", "compiler_time_ns"),
    [
        (None, None),
        (
            _observation(10, "forecast"),
            _observation(10, "measured"),
        ),
    ],
)
def test_compiler_cost_requires_exactly_one_representation(
    compiler_flops: ResourceObservation | None,
    compiler_time_ns: ResourceObservation | None,
) -> None:
    resources = _resources()
    with pytest.raises(ResourceReceiptError, match="exactly one"):
        replace(
            resources,
            compiler_flops=compiler_flops,
            compiler_time_ns=compiler_time_ns,
        )


def test_compiler_time_must_be_explicitly_measured() -> None:
    with pytest.raises(ResourceReceiptError, match="explicitly measured"):
        _resources(
            compiler_flops=None,
            compiler_time_ns=_observation(1_000, "forecast"),
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("examples", 0, "examples must be positive"),
        ("target_bits", 0, "target_bits must be positive"),
        ("source_bytes", 0, "source_bytes must be positive"),
        (
            "total_parameters",
            TOTAL_PARAMETER_LIMIT_EXCLUSIVE,
            "total parameters leave",
        ),
        (
            "persistent_bytes",
            PERSISTENT_BYTE_LIMIT + 1,
            "persistent state leaves",
        ),
        (
            "executor_flops_per_query",
            0,
            "executor_flops_per_query must be positive",
        ),
    ],
)
def test_resource_bounds_fail_closed(
    field_name: str,
    value: int,
    message: str,
) -> None:
    resources = _resources()
    with pytest.raises(ResourceReceiptError, match=message):
        replace(resources, **{field_name: _observation(value)})


def test_parameter_and_optimizer_inconsistencies_fail_closed() -> None:
    resources = _resources()
    with pytest.raises(ResourceReceiptError, match="exceed total"):
        replace(
            resources,
            trainable_parameters=_observation(150_000_000),
        )
    with pytest.raises(ResourceReceiptError, match="updates require"):
        replace(
            resources,
            trainable_parameters=_observation(0),
            updates=_observation(1),
            optimizer_bytes=_observation(0),
        )
    with pytest.raises(ResourceReceiptError, match="optimizer state"):
        replace(
            resources,
            trainable_parameters=_observation(0),
            updates=_observation(0),
            optimizer_bytes=_observation(1),
        )


def test_invalid_or_mismatched_bindings_fail_closed() -> None:
    with pytest.raises(ResourceReceiptError, match="digest differs"):
        ArtifactBindings(
            board_sha256="A" * 64,
            source_sha256="b" * 64,
            config_sha256="c" * 64,
        )

    receipt = create_resource_receipt(
        bindings=_bindings(),
        resources=_resources(),
    )
    other = replace(
        _bindings(),
        config_sha256=sha256_bytes(b"different-config"),
    )
    with pytest.raises(ResourceReceiptError, match="custody binding"):
        receipt.assert_bindings(other)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.update({"unknown": 1}),
            "resource receipt schema differs",
        ),
        (
            lambda value: value["bindings"].update({"unknown": "x"}),
            "artifact binding schema differs",
        ),
        (
            lambda value: value["resources"].update({"unknown": 1}),
            "resource vector schema differs",
        ),
        (
            lambda value: value["resources"]["examples"].update({"unknown": 1}),
            "examples observation schema differs",
        ),
    ],
)
def test_unknown_fields_fail_closed(mutation, message: str) -> None:
    receipt = create_resource_receipt(
        bindings=_bindings(),
        resources=_resources(),
    )
    mapping = _mapping_copy(receipt)
    mutation(mapping)
    with pytest.raises(ResourceReceiptError, match=message):
        QualificationResourceReceipt.from_mapping(mapping)


def test_limit_kind_and_digest_tampering_fail_closed() -> None:
    receipt = create_resource_receipt(
        bindings=_bindings(),
        resources=_resources(),
    )

    mapping = _mapping_copy(receipt)
    mapping["limits"]["persistent_bytes_max"] = 1_535
    with pytest.raises(ResourceReceiptError, match="limit differs"):
        QualificationResourceReceipt.from_mapping(mapping)

    mapping = _mapping_copy(receipt)
    mapping["receipt_kind"] = "measured"
    with pytest.raises(ResourceReceiptError, match="kind differs"):
        QualificationResourceReceipt.from_mapping(mapping)

    mapping = _mapping_copy(receipt)
    mapping["receipt_sha256"] = "0" * 64
    with pytest.raises(ResourceReceiptError, match="hash differs"):
        QualificationResourceReceipt.from_mapping(mapping)


def test_noncanonical_and_duplicate_json_fail_closed() -> None:
    receipt = create_resource_receipt(
        bindings=_bindings(),
        resources=_resources(),
    )
    pretty = json.dumps(receipt.to_mapping(), indent=2).encode("ascii")
    with pytest.raises(ResourceReceiptError, match="not canonical"):
        load_resource_receipt(pretty)

    duplicate = (
        receipt.to_json_bytes()[:-2]
        + b',"schema":"efc-qualification-resource-custody/v1"}\n'
    )
    with pytest.raises(ResourceReceiptError, match="duplicate key"):
        load_resource_receipt(duplicate)


def test_receipt_module_has_no_execution_or_environment_surface() -> None:
    source = inspect.getsource(receipt_module)
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    accessed_attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert imported_roots.isdisjoint({"dotenv", "os", "subprocess", "torch"})
    assert called_names.isdisjoint({"exec", "Popen", "run"})
    assert accessed_attributes.isdisjoint({"environ", "getenv", "Popen", "system"})
    assert '".env"' not in source
