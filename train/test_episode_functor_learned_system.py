from __future__ import annotations

import sys
from dataclasses import asdict, fields
from hashlib import sha256
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from episode_functor_learned_system import (  # noqa: E402
    LearnedEFCSystem,
    SealedFunctorBatch,
)
from episode_functor_machine import (  # noqa: E402
    HardFunctorKeys,
    MAX_ACTIONS,
    MAX_OBSERVERS,
    MAX_STATES,
)
from episode_functor_query_parser import (  # noqa: E402
    NeuralOpaqueQueryParser,
    collate_queries,
    scan_query,
)
from episode_functor_shohin_trunk import (  # noqa: E402
    FrozenShohinTrunk,
    ShohinTrunkBatch,
)
from episode_functor_witness_compiler import (  # noqa: E402
    ProofCarryingWitnessCompiler,
    collate_witness_sources,
    scan_witness_source,
)
from pipeline.episode_functor_identifiable_board import (  # noqa: E402
    GrammarFactors,
    LateQuery,
    encode_query,
    encode_source,
    generate_machine,
    hide_one_cell_per_relation,
)
from model import GPT, GPTConfig  # noqa: E402


def _payloads() -> tuple[object, bytes, bytes]:
    machine = generate_machine(
        seed="efc-learned-system-test-v1",
        split="mechanics",
        index=0,
        family="affine-f2-3",
    )
    evidence = hide_one_cell_per_relation(
        machine,
        seed="efc-learned-system-test-v1",
        split="mechanics",
        index=0,
    )
    source = encode_source(evidence, GrammarFactors(1, 1, 1))
    query = LateQuery(
        start_key=machine.state_keys[4],
        action_keys=(
            machine.action_keys[1],
            machine.action_keys[0],
            machine.action_keys[2],
        ),
        observer_key=machine.observer_keys[0],
    )
    return (
        machine,
        source,
        encode_query(query, GrammarFactors(0, 1, 0)),
    )


def _oracle_keys(machine) -> HardFunctorKeys:
    state = torch.zeros((1, MAX_STATES, 8), dtype=torch.uint8)
    action = torch.zeros((1, MAX_ACTIONS, 8), dtype=torch.uint8)
    observer = torch.zeros((1, MAX_OBSERVERS, 8), dtype=torch.uint8)
    for index, key in enumerate(machine.state_keys):
        state[0, index] = torch.tensor(
            tuple(key.to_bytes(8, "little")),
            dtype=torch.uint8,
        )
    for index, key in enumerate(machine.action_keys):
        action[0, index] = torch.tensor(
            tuple(key.to_bytes(8, "little")),
            dtype=torch.uint8,
        )
    for index, key in enumerate(machine.observer_keys):
        observer[0, index] = torch.tensor(
            tuple(key.to_bytes(8, "little")),
            dtype=torch.uint8,
        )
    return HardFunctorKeys(
        state_keys=state,
        action_keys=action,
        observer_keys=observer,
    )


def _byte_trunk_batch(payload: bytes) -> ShohinTrunkBatch:
    length = len(payload)
    return ShohinTrunkBatch(
        payloads=(payload,),
        token_ids=torch.tensor((tuple(payload),), dtype=torch.long),
        token_valid=torch.ones((1, length), dtype=torch.bool),
        token_byte_bounds=torch.tensor(
            (
                tuple(
                    (index, index + 1)
                    for index in range(length)
                ),
            ),
            dtype=torch.int32,
        ),
    )


def _small_system() -> LearnedEFCSystem:
    return LearnedEFCSystem(
        source_compiler=ProofCarryingWitnessCompiler(
            width=48,
            encoder_layers=1,
            decoder_layers=1,
            heads=3,
            feedforward=96,
            sinkhorn_iterations=16,
        ),
        query_parser=NeuralOpaqueQueryParser(
            width=48,
            layers=1,
            heads=3,
            feedforward=96,
            max_steps=8,
        ),
    )


def test_late_query_arrives_only_after_source_is_sealed_and_deleted() -> None:
    torch.manual_seed(67)
    machine, source_payload, query_payload = _payloads()
    source = collate_witness_sources(
        [scan_witness_source(source_payload)]
    )
    system = _small_system()
    compilation = system.compile_source(source, straight_through=True)
    sealed = system.seal(compilation)
    sealed = SealedFunctorBatch(
        machine=sealed.machine,
        keys=_oracle_keys(machine),
    )
    assert tuple(field.name for field in fields(sealed)) == (
        "machine",
        "keys",
    )
    assert isinstance(sealed, SealedFunctorBatch)
    assert len(sealed.deployed_wire(0)) == 1_536

    source.pointer.byte_ids.fill_(0)
    source.pointer.byte_valid.fill_(False)
    source.record_bounds.fill_(0)
    source.record_valid.fill_(False)
    del compilation
    query = collate_queries([scan_query(query_payload)])
    output = system(sealed, query)
    replay = system.execute_sealed(sealed, output.query)
    assert torch.equal(output.rollout.states, replay.states)
    assert torch.equal(output.rollout.answer, replay.answer)


def test_default_parameter_receipt_is_honest_about_missing_shohin() -> None:
    system = LearnedEFCSystem()
    receipt = system.parameter_receipt()
    assert receipt.integration_status == "not_connected"
    assert receipt.integrated_shohin == 0
    assert receipt.instantiated_total == receipt.added_total
    assert receipt.hypothetical_complete_total < 200_000_000
    assert receipt.hypothetical_headroom > 0
    assert receipt.added_total == (
        receipt.source_compiler + receipt.query_parser
    )


def test_direct_parent_constructor_cannot_claim_connected_checkpoint() -> None:
    torch.manual_seed(71)
    parent = GPT(
        GPTConfig(
            vocab_size=257,
            n_layer=1,
            n_head=3,
            n_kv_head=1,
            d_model=12,
            d_ff=24,
            seq_len=128,
            zloss=0.0,
        )
    )
    trunk = FrozenShohinTrunk(
        parent,
        checkpoint_sha256="0" * 64,
        block_indices=(0,),
    )
    system = LearnedEFCSystem(
        source_compiler=ProofCarryingWitnessCompiler(
            width=48,
            encoder_layers=1,
            decoder_layers=1,
            heads=3,
            feedforward=96,
            sinkhorn_iterations=16,
            external_feature_width=trunk.feature_width,
        ),
        query_parser=NeuralOpaqueQueryParser(
            width=48,
            layers=1,
            heads=3,
            feedforward=96,
            max_steps=8,
            external_feature_width=trunk.feature_width,
        ),
        frozen_trunk=trunk,
    )
    machine, source_payload, query_payload = _payloads()
    source = collate_witness_sources(
        [scan_witness_source(source_payload)]
    )
    compilation = system.compile_source(
        source,
        straight_through=True,
        trunk_batch=_byte_trunk_batch(source_payload),
    )
    random_seal = system.seal(compilation)
    sealed = SealedFunctorBatch(
        machine=random_seal.machine,
        keys=_oracle_keys(machine),
    )
    query = collate_queries([scan_query(query_payload)])
    output = system(
        sealed,
        query,
        trunk_batch=_byte_trunk_batch(query_payload),
    )
    receipt = system.parameter_receipt(
        protected_shohin=parent.num_params(),
        protected_checkpoint_sha256="0" * 64,
    )
    assert receipt.integration_status == "not_connected"
    assert receipt.integrated_shohin == parent.num_params()
    assert receipt.instantiated_total == (
        receipt.integrated_shohin + receipt.added_total
    )
    assert output.rollout.answer.shape == (1,)
    assert all(
        not parameter.requires_grad
        for parameter in system.frozen_trunk.parent.parameters()
    )


def test_hash_verified_loader_can_claim_exact_checkpoint(
    tmp_path: Path,
) -> None:
    torch.manual_seed(73)
    config = GPTConfig(
        vocab_size=257,
        n_layer=1,
        n_head=3,
        n_kv_head=1,
        d_model=12,
        d_ff=24,
        seq_len=128,
        zloss=0.0,
    )
    parent = GPT(config)
    checkpoint = tmp_path / "tiny.pt"
    torch.save(
        {
            "cfg": asdict(config),
            "model": parent.state_dict(),
        },
        checkpoint,
    )
    digest = sha256(checkpoint.read_bytes()).hexdigest()
    trunk = FrozenShohinTrunk.from_checkpoint(
        checkpoint,
        expected_sha256=digest,
        block_indices=(0,),
    )
    system = LearnedEFCSystem(
        source_compiler=ProofCarryingWitnessCompiler(
            width=48,
            encoder_layers=1,
            decoder_layers=1,
            heads=3,
            feedforward=96,
            sinkhorn_iterations=16,
            external_feature_width=trunk.feature_width,
        ),
        query_parser=NeuralOpaqueQueryParser(
            width=48,
            layers=1,
            heads=3,
            feedforward=96,
            max_steps=8,
            external_feature_width=trunk.feature_width,
        ),
        frozen_trunk=trunk,
    )
    receipt = system.parameter_receipt(
        protected_shohin=parent.num_params(),
        protected_checkpoint_sha256=digest,
    )
    assert receipt.integration_status == "connected"
    assert receipt.checkpoint_verified
    assert receipt.integrated_checkpoint_sha256 == digest
