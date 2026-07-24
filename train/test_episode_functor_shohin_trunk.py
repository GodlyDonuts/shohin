from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import inspect
from pathlib import Path
import sys
import builtins

import pytest
import torch


TRAIN = Path(__file__).resolve().parent
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from episode_functor_shohin_trunk import (  # noqa: E402
    ByteAlignedResidualFeatures,
    FrozenShohinTrunk,
    FrozenShohinTrunkError,
    ShohinTrunkBatch,
)
import episode_functor_shohin_trunk as trunk_module  # noqa: E402
import model as model_module  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402


def _tiny_checkpoint(path: Path) -> tuple[GPTConfig, int, str]:
    torch.manual_seed(73)
    config = GPTConfig(
        vocab_size=64,
        n_layer=3,
        n_head=4,
        n_kv_head=2,
        d_model=16,
        d_ff=32,
        seq_len=16,
        zloss=0.0,
    )
    model = GPT(config)
    torch.save(
        {
            "cfg": asdict(config),
            "model": model.state_dict(),
        },
        path,
    )
    return config, model.num_params(), sha256(path.read_bytes()).hexdigest()


def _valid_inputs() -> tuple[
    tuple[bytes, ...],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    payloads = (b"abcd", b"xyz")
    token_ids = torch.tensor(
        (
            (3, 5, 0),
            (7, 11, 13),
        ),
        dtype=torch.long,
    )
    token_valid = torch.tensor(
        (
            (True, True, False),
            (True, True, True),
        ),
        dtype=torch.bool,
    )
    token_byte_bounds = torch.tensor(
        (
            ((0, 1), (1, 4), (0, 0)),
            ((0, 1), (1, 2), (2, 3)),
        ),
        dtype=torch.int32,
    )
    return payloads, token_ids, token_valid, token_byte_bounds


def test_hash_bound_load_freezes_parent_and_reports_exact_counts(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    config, expected_parameters, expected_sha256 = _tiny_checkpoint(checkpoint)
    before = sha256(checkpoint.read_bytes()).hexdigest()

    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    receipt = trunk.parameter_receipt()

    assert trunk.parent.cfg == config
    assert not trunk.parent.training
    assert all(not parameter.requires_grad for parameter in trunk.parent.parameters())
    assert receipt.checkpoint_sha256 == expected_sha256
    assert receipt.checkpoint_verified
    assert receipt.parent_unique_parameters == expected_parameters
    assert receipt.adapter_unique_parameters == 0
    assert receipt.integrated_unique_parameters == expected_parameters
    assert receipt.trainable_unique_parameters == 0
    assert sha256(checkpoint.read_bytes()).hexdigest() == before

    trunk.train()
    assert trunk.training
    assert not trunk.parent.training


def test_checkpoint_verification_is_not_a_replayable_constructor_token(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    config, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    random_parent = GPT(config)
    trunk = FrozenShohinTrunk(
        random_parent,
        checkpoint_sha256=expected_sha256,
        block_indices=(0, 2),
    )

    assert "_verification_capability" not in inspect.signature(
        FrozenShohinTrunk
    ).parameters
    assert not hasattr(
        trunk_module,
        "_VERIFIED_CHECKPOINT_CONSTRUCTOR",
    )
    assert not trunk.parameter_receipt().checkpoint_verified


def test_verified_receipt_fails_if_loaded_parent_changes(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    assert trunk.parameter_receipt().checkpoint_verified

    with torch.no_grad():
        next(trunk.parent.parameters()).flatten()[0].add_(1.0)
    assert not trunk.parameter_receipt().checkpoint_verified


def test_verified_receipt_fails_on_runtime_semantic_hook(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    assert trunk.parameter_receipt().checkpoint_verified

    handle = trunk.parent.blocks[0].register_forward_hook(
        lambda _module, _inputs, output: (output[0] + 1.0, output[1])
    )
    try:
        assert not trunk.parameter_receipt().checkpoint_verified
    finally:
        handle.remove()
    assert trunk.parameter_receipt().checkpoint_verified


def test_verified_receipt_fails_on_runtime_class_method_replacement(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    assert trunk.parameter_receipt().checkpoint_verified

    block_type = type(trunk.parent.blocks[0])
    original = block_type.forward

    def changed_forward(self, *args, **kwargs):
        output, cache = original(self, *args, **kwargs)
        return output + 1.0, cache

    block_type.forward = changed_forward
    try:
        assert not trunk.parameter_receipt().checkpoint_verified
    finally:
        block_type.forward = original
    assert trunk.parameter_receipt().checkpoint_verified


def test_verified_receipt_fails_on_runtime_method_default_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    assert trunk.parameter_receipt().checkpoint_verified

    forward = type(trunk.parent.blocks[0]).forward
    original = forward.__defaults__
    assert original is not None
    forward.__defaults__ = (*original[:-1], int(original[-1]) + 1)
    try:
        assert not trunk.parameter_receipt().checkpoint_verified
    finally:
        forward.__defaults__ = original
    assert trunk.parameter_receipt().checkpoint_verified


def test_verified_receipt_fails_on_referenced_module_callable_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    assert trunk.parameter_receipt().checkpoint_verified

    original = model_module.F.silu

    def changed_silu(value):
        return original(value) + 1.0

    model_module.F.silu = changed_silu
    try:
        assert not trunk.parameter_receipt().checkpoint_verified
    finally:
        model_module.F.silu = original
    assert trunk.parameter_receipt().checkpoint_verified


def test_verified_receipt_fails_on_transitive_external_callable_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    assert trunk.parameter_receipt().checkpoint_verified

    torch_module = model_module.F.silu.__globals__["torch"]
    original = torch_module._C._nn.silu

    def changed_silu(value):
        return original(value) + 1.0

    torch_module._C._nn.silu = changed_silu
    try:
        assert not trunk.parameter_receipt().checkpoint_verified
    finally:
        torch_module._C._nn.silu = original
    assert trunk.parameter_receipt().checkpoint_verified


def test_verified_receipt_fails_on_external_class_method_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    assert trunk.parameter_receipt().checkpoint_verified

    original = model_module.nn.Linear.forward

    def changed_forward(self, value):
        return original(self, value) + 1.0

    model_module.nn.Linear.forward = changed_forward
    try:
        assert not trunk.parameter_receipt().checkpoint_verified
    finally:
        model_module.nn.Linear.forward = original
    assert trunk.parameter_receipt().checkpoint_verified


def test_verified_receipt_fails_on_external_container_iteration_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    assert trunk.parameter_receipt().checkpoint_verified

    original = model_module.nn.ModuleList.__iter__

    def reversed_iteration(self):
        return iter(tuple(original(self))[::-1])

    model_module.nn.ModuleList.__iter__ = reversed_iteration
    try:
        assert not trunk.parameter_receipt().checkpoint_verified
    finally:
        model_module.nn.ModuleList.__iter__ = original
    assert trunk.parameter_receipt().checkpoint_verified


def test_verified_receipt_fails_on_external_attribute_dispatch_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    assert trunk.parameter_receipt().checkpoint_verified

    parent = trunk.parent
    original = model_module.nn.Module.__getattr__

    def changed_getattr(self, name):
        value = original(self, name)
        if self is parent and name == "blocks":
            return model_module.nn.ModuleList(reversed(tuple(value)))
        return value

    model_module.nn.Module.__getattr__ = changed_getattr
    try:
        assert not trunk.parameter_receipt().checkpoint_verified
    finally:
        model_module.nn.Module.__getattr__ = original
    assert trunk.parameter_receipt().checkpoint_verified


def test_verified_receipt_fails_on_referenced_builtin_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    assert trunk.parameter_receipt().checkpoint_verified

    original = builtins.enumerate

    def changed_enumerate(iterable, start=0):
        if isinstance(iterable, model_module.nn.ModuleList):
            iterable = reversed(tuple(iterable))
        return original(iterable, start)

    builtins.enumerate = changed_enumerate
    try:
        assert not trunk.parameter_receipt().checkpoint_verified
    finally:
        builtins.enumerate = original
    assert trunk.parameter_receipt().checkpoint_verified


def test_verified_receipt_fails_on_ordered_child_topology_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    assert trunk.parameter_receipt().checkpoint_verified

    original = trunk.parent.blocks._modules
    trunk.parent.blocks._modules = dict(reversed(tuple(original.items())))
    try:
        assert not trunk.parameter_receipt().checkpoint_verified
    finally:
        trunk.parent.blocks._modules = original
    assert trunk.parameter_receipt().checkpoint_verified


def test_verified_receipt_fails_on_parent_data_descriptor_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    assert trunk.parameter_receipt().checkpoint_verified

    def changed_blocks(self):
        blocks = self._modules["blocks"]
        return model_module.nn.ModuleList(reversed(tuple(blocks)))

    model_module.GPT.blocks = property(changed_blocks)
    try:
        assert not trunk.parameter_receipt().checkpoint_verified
    finally:
        del model_module.GPT.blocks
    assert trunk.parameter_receipt().checkpoint_verified


def test_verified_receipt_fails_on_output_constructor_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    assert trunk.parameter_receipt().checkpoint_verified

    original = ByteAlignedResidualFeatures.__init__

    def changed_init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        object.__setattr__(
            self,
            "token_features",
            self.token_features + 1.0,
        )

    ByteAlignedResidualFeatures.__init__ = changed_init
    try:
        assert not trunk.parameter_receipt().checkpoint_verified
    finally:
        ByteAlignedResidualFeatures.__init__ = original
    assert trunk.parameter_receipt().checkpoint_verified


def test_verified_receipt_fails_on_feature_width_descriptor_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    assert trunk.parameter_receipt().checkpoint_verified

    original = FrozenShohinTrunk.feature_width
    FrozenShohinTrunk.feature_width = property(
        lambda self: original.fget(self) + 1
    )
    try:
        assert not trunk.parameter_receipt().checkpoint_verified
    finally:
        FrozenShohinTrunk.feature_width = original
    assert trunk.parameter_receipt().checkpoint_verified


@pytest.mark.parametrize(
    "transport_type",
    (ByteAlignedResidualFeatures, ShohinTrunkBatch),
)
def test_verified_receipt_fails_on_transport_attribute_dispatch_mutation(
    tmp_path: Path,
    transport_type: type,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    assert trunk.parameter_receipt().checkpoint_verified

    original = transport_type.__getattribute__

    def changed_getattribute(self, name):
        value = original(self, name)
        if name in ("byte_features", "token_ids"):
            return value + 1
        return value

    transport_type.__getattribute__ = changed_getattribute
    try:
        assert not trunk.parameter_receipt().checkpoint_verified
    finally:
        del transport_type.__getattribute__
    assert trunk.parameter_receipt().checkpoint_verified


def test_verified_receipt_fails_on_trunk_forward_replacement(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    assert trunk.parameter_receipt().checkpoint_verified

    original = FrozenShohinTrunk.forward

    def changed_forward(self, *args, **kwargs):
        features = original(self, *args, **kwargs)
        return type(features)(
            block_indices=features.block_indices,
            token_features=features.token_features + 1.0,
            token_valid=features.token_valid,
            byte_features=features.byte_features + 1.0,
            byte_valid=features.byte_valid,
            payload_lengths=features.payload_lengths,
        )

    FrozenShohinTrunk.forward = changed_forward
    try:
        assert not trunk.parameter_receipt().checkpoint_verified
    finally:
        FrozenShohinTrunk.forward = original
    assert trunk.parameter_receipt().checkpoint_verified


def test_verified_receipt_fails_on_trunk_instance_method_override(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    assert trunk.parameter_receipt().checkpoint_verified

    trunk._validate_inputs = lambda *args, **kwargs: ((), ())
    assert not trunk.parameter_receipt().checkpoint_verified


def test_verified_receipt_fails_on_trunk_execution_configuration_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    assert trunk.parameter_receipt().checkpoint_verified

    original = trunk.block_indices
    trunk.block_indices = (1, 2)
    try:
        assert not trunk.parameter_receipt().checkpoint_verified
    finally:
        trunk.block_indices = original
    assert trunk.parameter_receipt().checkpoint_verified


def test_verified_receipt_fails_on_nonpersistent_runtime_buffer_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    assert trunk.parameter_receipt().checkpoint_verified

    with torch.no_grad():
        trunk.parent.cos.flatten()[0].add_(1.0)
    assert not trunk.parameter_receipt().checkpoint_verified


def test_wrong_checkpoint_hash_fails_before_model_construction(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    wrong = ("0" if expected_sha256[0] != "0" else "1") + expected_sha256[1:]
    with pytest.raises(FrozenShohinTrunkError, match="SHA-256 mismatch"):
        FrozenShohinTrunk.from_checkpoint(
            checkpoint,
            expected_sha256=wrong,
            block_indices=(0,),
        )


def test_raw_post_block_features_are_exactly_replicated_over_token_bytes(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    config, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    payloads, token_ids, token_valid, token_byte_bounds = _valid_inputs()

    features = trunk(
        payloads,
        token_ids,
        token_valid,
        token_byte_bounds,
    )

    assert features.token_features.shape == (2, 2, 3, config.d_model)
    assert features.byte_features.shape == (2, 2, 4, config.d_model)
    assert features.byte_valid.tolist() == [
        [True, True, True, True],
        [True, True, True, False],
    ]
    assert features.payload_lengths.tolist() == [4, 3]
    with torch.no_grad():
        hidden = trunk.parent.tok(token_ids)
        cos = trunk.parent.cos[: token_ids.shape[1]]
        sin = trunk.parent.sin[: token_ids.shape[1]]
        expected = []
        for index, block in enumerate(trunk.parent.blocks):
            hidden, _ = block(hidden, cos, sin)
            if index in trunk.block_indices:
                expected.append(hidden)
        expected_tokens = torch.stack(expected, dim=1)
        expected_tokens = expected_tokens * token_valid[:, None, :, None]
    assert torch.allclose(
        features.token_features,
        expected_tokens,
        atol=1e-7,
        rtol=1e-6,
    )
    assert torch.equal(
        features.byte_features[0, :, 0],
        features.token_features[0, :, 0],
    )
    for byte in (1, 2, 3):
        assert torch.equal(
            features.byte_features[0, :, byte],
            features.token_features[0, :, 1],
        )
    assert bool(features.byte_features[1, :, 3].eq(0).all())
    assert bool(features.token_features[0, :, 2].eq(0).all())
    assert not features.token_features.requires_grad
    assert not features.byte_features.requires_grad


def test_token_sequences_longer_than_parent_context_use_exact_windows(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    config, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0, 2),
    )
    token_count = config.seq_len + 5
    token_ids = (
        torch.arange(token_count, dtype=torch.long)
        % config.vocab_size
    )[None]
    payload = bytes((65 + index % 26 for index in range(token_count)))
    bounds = torch.tensor(
        tuple((index, index + 1) for index in range(token_count)),
        dtype=torch.int32,
    )[None]
    features = trunk(
        (payload,),
        token_ids,
        torch.ones((1, token_count), dtype=torch.bool),
        bounds,
    )
    expected_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for left in range(0, token_count, config.seq_len):
            right = min(token_count, left + config.seq_len)
            hidden = trunk.parent.tok(token_ids[:, left:right])
            cos = trunk.parent.cos[: right - left]
            sin = trunk.parent.sin[: right - left]
            captured: list[torch.Tensor] = []
            for index, block in enumerate(trunk.parent.blocks):
                hidden, _ = block(hidden, cos, sin)
                if index in trunk.block_indices:
                    captured.append(hidden)
            expected_chunks.append(torch.stack(captured, dim=1))
    expected = torch.cat(expected_chunks, dim=2)
    assert torch.equal(features.token_features, expected)
    assert torch.equal(
        features.byte_features,
        features.token_features,
    )


@pytest.mark.parametrize(
    ("bounds", "message"),
    (
        (
            (((0, 2), (1, 3)),),
            "overlap",
        ),
        (
            (((0, 1), (2, 3)),),
            "uncovered",
        ),
    ),
)
def test_overlapping_and_uncovered_offsets_fail_closed(
    tmp_path: Path,
    bounds: tuple[tuple[tuple[int, int], ...], ...],
    message: str,
) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(1,),
    )
    with pytest.raises(FrozenShohinTrunkError, match=message):
        trunk(
            (b"abc",),
            torch.tensor(((3, 5),), dtype=torch.long),
            torch.tensor(((True, True),), dtype=torch.bool),
            torch.tensor(bounds, dtype=torch.int64),
        )


def test_validity_and_invalid_bounds_fail_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0,),
    )
    payloads = (b"ab",)
    token_ids = torch.tensor(((3, 0, 5),), dtype=torch.long)

    with pytest.raises(FrozenShohinTrunkError, match="contiguous prefix"):
        trunk(
            payloads,
            token_ids,
            torch.tensor(((True, False, True),), dtype=torch.bool),
            torch.tensor((((0, 1), (0, 0), (1, 2)),), dtype=torch.int32),
        )

    with pytest.raises(FrozenShohinTrunkError, match="must be zero"):
        trunk(
            payloads,
            token_ids,
            torch.tensor(((True, True, False),), dtype=torch.bool),
            torch.tensor((((0, 1), (1, 2), (1, 2)),), dtype=torch.int32),
        )


def test_block_indices_and_checkpoint_schema_fail_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    with pytest.raises(FrozenShohinTrunkError, match="block indices"):
        FrozenShohinTrunk.from_checkpoint(
            checkpoint,
            expected_sha256=expected_sha256,
            block_indices=(2, 1),
        )

    malformed = tmp_path / "malformed.pt"
    torch.save({"cfg": {}, "model": {}}, malformed)
    malformed_sha256 = sha256(malformed.read_bytes()).hexdigest()
    with pytest.raises(FrozenShohinTrunkError, match="strictly instantiate"):
        FrozenShohinTrunk.from_checkpoint(
            malformed,
            expected_sha256=malformed_sha256,
            block_indices=(0,),
        )


def test_payload_allocation_is_explicitly_bounded(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny.pt"
    _, _, expected_sha256 = _tiny_checkpoint(checkpoint)
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        block_indices=(0,),
        max_payload_bytes=2,
    )
    with pytest.raises(FrozenShohinTrunkError, match="outside byte support"):
        trunk(
            (b"abc",),
            torch.tensor(((3,),), dtype=torch.long),
            torch.tensor(((True,),), dtype=torch.bool),
            torch.tensor((((0, 3),),), dtype=torch.int32),
        )
