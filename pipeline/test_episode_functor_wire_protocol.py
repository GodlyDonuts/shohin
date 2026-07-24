from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from pipeline.episode_functor_seal_protocol import (
    Beacon,
    ProtocolViolation,
    canonical_json_bytes,
)
from pipeline.episode_functor_wire_protocol import (
    C_RUNTIME_SOURCE,
    INDEPENDENT_GENERATOR_SOURCE,
    MACHINE_HASH_OFFSET,
    MACHINE_SIZE,
    RUST_RUNTIME_SOURCE,
    WireProtocolSpec,
    WireSealFirstRehearsal,
    decode_deployed_machine,
    decode_transcript,
    encode_deployed_machine,
    machine_byte_receipt,
)


WORLD_BEACON = Beacon(round=1_000, value="consumed-wire-world-A")
CHALLENGE_A = Beacon(round=2_000, value="consumed-wire-challenge-A")
CHALLENGE_B = Beacon(round=2_001, value="consumed-wire-challenge-B")


@pytest.fixture(scope="session")
def wire_runtimes(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Path]:
    cc = shutil.which("cc")
    rustc = shutil.which("rustc")
    if cc is None or rustc is None:
        pytest.skip("strict C and Rust compilers are required")
    root = tmp_path_factory.mktemp("episode_functor_wire_protocol")
    c_runtime = root / "runtime_c"
    rust_runtime = root / "runtime_rust"
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            str(C_RUNTIME_SOURCE),
            "-o",
            str(c_runtime),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            rustc,
            "--edition=2021",
            "-C",
            "opt-level=2",
            "-D",
            "warnings",
            str(RUST_RUNTIME_SOURCE),
            "-o",
            str(rust_runtime),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {"c": c_runtime, "rust": rust_runtime}


def _attested_spec(runtimes: dict[str, Path]) -> WireProtocolSpec:
    return WireProtocolSpec(
        runtime_binary_sha256=tuple(
            (name, sha256(runtimes[name].read_bytes()).hexdigest())
            for name in ("c", "rust")
        )
    )


def _sealed(
    root: Path,
    runtimes: dict[str, Path] | None = None,
) -> WireSealFirstRehearsal:
    spec = _attested_spec(runtimes) if runtimes is not None else WireProtocolSpec()
    rehearsal = WireSealFirstRehearsal(root, spec)
    rehearsal.supply_world_beacon(WORLD_BEACON)
    rehearsal.seal_machine()
    return rehearsal


def test_exact_machine_wire_and_byte_receipt(tmp_path: Path) -> None:
    rehearsal = _sealed(tmp_path / "wire")
    machine = rehearsal.machine_path.read_bytes()
    spec = rehearsal.spec
    receipt = machine_byte_receipt(spec)

    assert len(machine) == MACHINE_SIZE
    assert machine[:8] == b"EFCMACH\0"
    assert machine[MACHINE_HASH_OFFSET:] == sha256(
        machine[:MACHINE_HASH_OFFSET]
    ).digest()
    assert receipt["accounted_bytes"] == MACHINE_SIZE
    assert receipt["unaccounted_bytes"] == 0
    assert receipt["source_dependent_bytes_including_hash"] == 207
    assert sum(row["length"] for row in receipt["segments"]) == MACHINE_SIZE
    decoded = decode_deployed_machine(machine, spec)
    assert len(decoded.state_keys) == spec.state_count
    assert len(decoded.action_keys) == spec.action_count
    assert len(decoded.observer_keys) == spec.observer_count
    assert len(decoded.transitions) == spec.action_count
    assert len(decoded.observations) == spec.observer_count
    latent = json.loads(rehearsal.latent_path.read_bytes())
    assert decoded.transitions == tuple(
        tuple(row) for row in latent["transition_relations"]
    )
    assert decoded.observations == tuple(
        tuple(row) for row in latent["observer_maps"]
    )


def test_compiler_uses_canonical_public_evidence_only(tmp_path: Path) -> None:
    rehearsal = WireSealFirstRehearsal(tmp_path / "source")
    fixture = rehearsal.supply_world_beacon(WORLD_BEACON)
    baseline = encode_deployed_machine(fixture.evidence, rehearsal.spec)

    row = json.loads(fixture.evidence)
    row["demonstrations"] = list(reversed(row["demonstrations"]))
    row["observations"] = list(reversed(row["observations"]))
    reordered = encode_deployed_machine(
        canonical_json_bytes(row), rehearsal.spec
    )
    assert reordered == baseline

    poisoned_latent = b'{"latent":"not available to compiler"}\n'
    rehearsal.latent_path.write_bytes(poisoned_latent)
    rehearsal.seal_machine()
    assert rehearsal.machine_path.read_bytes() == baseline


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("drop-demonstration", "omits a transition"),
        ("duplicate-demonstration", "duplicate transition"),
        ("unknown-key", "inferred key counts"),
        ("query-taint", "schema contains drift or query taint"),
    ],
)
def test_compiler_fails_closed_on_incomplete_or_tainted_source(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    rehearsal = WireSealFirstRehearsal(tmp_path / mutation)
    fixture = rehearsal.supply_world_beacon(WORLD_BEACON)
    row = json.loads(fixture.evidence)
    if mutation == "drop-demonstration":
        row["demonstrations"].pop()
    elif mutation == "duplicate-demonstration":
        row["demonstrations"].append(dict(row["demonstrations"][0]))
    elif mutation == "unknown-key":
        row["demonstrations"][0]["source_key"] = 2**64 - 1
    else:
        row["query_depth"] = 6
    with pytest.raises(ProtocolViolation, match=message):
        encode_deployed_machine(canonical_json_bytes(row), rehearsal.spec)


def test_challenge_requires_source_delete_and_strict_later_beacon(
    wire_runtimes: dict[str, Path],
    tmp_path: Path,
) -> None:
    rehearsal = _sealed(tmp_path / "guards", wire_runtimes)
    with pytest.raises(ProtocolViolation, match="source must be deleted"):
        rehearsal.run_challenge(CHALLENGE_A, wire_runtimes)
    rehearsal.poison_and_delete_source()
    with pytest.raises(ProtocolViolation, match="strictly later"):
        rehearsal.run_challenge(
            Beacon(round=WORLD_BEACON.round, value="not-later"),
            wire_runtimes,
        )
    with pytest.raises(ProtocolViolation, match="must differ"):
        rehearsal.run_challenge(
            Beacon(round=2_000, value=WORLD_BEACON.value),
            wire_runtimes,
        )


def test_two_beacons_reuse_exact_machine_and_c_rust_transcripts(
    wire_runtimes: dict[str, Path],
    tmp_path: Path,
) -> None:
    rehearsal = _sealed(tmp_path / "two-beacon", wire_runtimes)
    machine_before = rehearsal.machine_path.read_bytes()
    delete_receipt = rehearsal.poison_and_delete_source()
    first = rehearsal.run_challenge(CHALLENGE_A, wire_runtimes)
    second = rehearsal.run_challenge(CHALLENGE_B, wire_runtimes)

    assert delete_receipt["source_deleted"]
    assert not rehearsal.source_path.exists()
    assert rehearsal.machine_path.read_bytes() == machine_before
    assert rehearsal.compile_count == 1
    assert first.machine_root == second.machine_root == rehearsal.machine_root
    assert first.coordinate_root != second.coordinate_root
    assert first.query_root != second.query_root
    assert first.compile_count_before == first.compile_count_after == 1
    assert second.compile_count_before == second.compile_count_after == 1

    for receipt in (first, second):
        challenge = (
            rehearsal.root
            / "challenges"
            / receipt.challenge_seed_commitment
        )
        c_transcript = challenge.joinpath("transcript.c.bin").read_bytes()
        rust_transcript = challenge.joinpath(
            "transcript.rust.bin"
        ).read_bytes()
        queries = challenge.joinpath("queries.bin").read_bytes()
        assert c_transcript == rust_transcript
        assert sha256(queries).hexdigest() == receipt.query_sha256
        assert sha256(c_transcript).hexdigest() == receipt.transcript_sha256
        records = decode_transcript(
            c_transcript, rehearsal.machine_path.read_bytes(), queries
        )
        assert len(records) == receipt.total_coordinates == 100
        assert [record.challenge_id for record in records] == list(
            range(1, 101)
        )
        semantic_queries = {
            (
                int.from_bytes(
                    queries[
                        64 + index * 320 + 8 :
                        64 + index * 320 + 16
                    ],
                    "little",
                ),
                int.from_bytes(
                    queries[
                        64 + index * 320 + 16 :
                        64 + index * 320 + 24
                    ],
                    "little",
                ),
                queries[
                    64 + index * 320 + 32 :
                    64 + index * 320 + 288
                ],
            )
            for index in range(100)
        }
        assert len(semantic_queries) == 100
        assert (
            receipt.machine_seal_event
            < receipt.challenge_seed_event
            < receipt.coordinate_commit_event
            < receipt.query_render_event
            < receipt.prediction_seal_event
            < receipt.answer_assessment_event
        )


def test_abstract_coordinates_are_sealed_before_source_keys_appear(
    wire_runtimes: dict[str, Path],
    tmp_path: Path,
) -> None:
    rehearsal = _sealed(tmp_path / "coordinates", wire_runtimes)
    machine = rehearsal.machine_path.read_bytes()
    tables = decode_deployed_machine(machine, rehearsal.spec)
    rehearsal.poison_and_delete_source()
    receipt = rehearsal.run_challenge(CHALLENGE_A, wire_runtimes)
    challenge = (
        rehearsal.root
        / "challenges"
        / receipt.challenge_seed_commitment
    )
    abstract = challenge.joinpath("abstract_coordinates.json").read_bytes()
    queries = challenge.joinpath("queries.bin").read_bytes()
    for key in (
        *tables.state_keys,
        *tables.action_keys,
        *tables.observer_keys,
    ):
        assert str(key).encode("ascii") not in abstract
        assert key.to_bytes(8, "little") in queries


def test_protocol_binds_exact_runtime_sources_and_wire_constants(
    wire_runtimes: dict[str, Path],
    tmp_path: Path,
) -> None:
    rehearsal = WireSealFirstRehearsal(
        tmp_path / "protocol", _attested_spec(wire_runtimes)
    )
    protocol = json.loads(rehearsal.root.joinpath("protocol.json").read_bytes())
    assert protocol["machine_bytes"] == 1_536
    assert protocol["machine_hash_offset"] == 1_504
    assert protocol["query_record_bytes"] == 320
    assert protocol["transcript_record_bytes"] == 32
    assert protocol["c_runtime_source_sha256"] == sha256(
        C_RUNTIME_SOURCE.read_bytes()
    ).hexdigest()
    assert protocol["rust_runtime_source_sha256"] == sha256(
        RUST_RUNTIME_SOURCE.read_bytes()
    ).hexdigest()
    assert protocol["independent_generator_source_sha256"] == sha256(
        INDEPENDENT_GENERATOR_SOURCE.read_bytes()
    ).hexdigest()
    assert protocol["world_generator"] == (
        "efc-independent-counter-world-v1"
    )
    assert protocol["runtime_binaries_attested"]
    assert protocol["runtime_binary_sha256"] == {
        name: sha256(path.read_bytes()).hexdigest()
        for name, path in wire_runtimes.items()
    }


def test_independent_generator_has_no_original_generator_or_query_api(
    tmp_path: Path,
) -> None:
    source = INDEPENDENT_GENERATOR_SOURCE.read_text(encoding="utf-8")
    executable_source = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert "episode_functor_seal_protocol" not in executable_source
    assert "generate_consumed_world_fixture" not in executable_source
    assert "def generate_independent_world(" in source
    signature = source.split("def generate_independent_world(", 1)[1].split(
        ") -> IndependentWorld:", 1
    )[0]
    assert "query" not in signature
    assert "challenge" not in signature

    rehearsal = WireSealFirstRehearsal(tmp_path / "independent")
    fixture = rehearsal.supply_world_beacon(WORLD_BEACON)
    assert fixture.admissibility_receipt["query_fields_seen"] == 0
    assert fixture.admissibility_receipt["generator"] == (
        "efc-independent-counter-world-v1"
    )
    assert fixture.admissibility_receipt["accepted_candidate"] >= 0


def test_second_compile_and_reused_challenge_are_rejected(
    wire_runtimes: dict[str, Path],
    tmp_path: Path,
) -> None:
    rehearsal = _sealed(tmp_path / "single-shot", wire_runtimes)
    with pytest.raises(ProtocolViolation, match="single-shot"):
        rehearsal.seal_machine()
    rehearsal.poison_and_delete_source()
    rehearsal.run_challenge(CHALLENGE_A, wire_runtimes)
    with pytest.raises(ProtocolViolation, match="already been consumed"):
        rehearsal.run_challenge(CHALLENGE_A, wire_runtimes)


def test_unattested_or_changed_runtime_and_latent_are_rejected(
    wire_runtimes: dict[str, Path],
    tmp_path: Path,
) -> None:
    unattested = _sealed(tmp_path / "unattested")
    unattested.poison_and_delete_source()
    with pytest.raises(ProtocolViolation, match="not frozen"):
        unattested.run_challenge(CHALLENGE_A, wire_runtimes)

    changed_runtime = _sealed(
        tmp_path / "changed-runtime", wire_runtimes
    )
    changed_runtime.poison_and_delete_source()
    fake_runtime = tmp_path / "fake-runtime"
    fake_runtime.write_bytes(b"not the attested runtime")
    with pytest.raises(ProtocolViolation, match="differs"):
        changed_runtime.run_challenge(
            CHALLENGE_A,
            {"c": fake_runtime, "rust": wire_runtimes["rust"]},
        )

    changed_latent = _sealed(
        tmp_path / "changed-latent", wire_runtimes
    )
    changed_latent.poison_and_delete_source()
    changed_latent.latent_path.write_bytes(b'{"changed":true}\n')
    with pytest.raises(ProtocolViolation, match="latent differs"):
        changed_latent.run_challenge(CHALLENGE_A, wire_runtimes)


def test_mutated_protocol_or_machine_is_rejected_before_next_phase(
    wire_runtimes: dict[str, Path],
    tmp_path: Path,
) -> None:
    changed_protocol = WireSealFirstRehearsal(
        tmp_path / "changed-protocol", _attested_spec(wire_runtimes)
    )
    changed_protocol.root.joinpath("protocol.json").write_bytes(b"{}\n")
    with pytest.raises(ProtocolViolation, match="protocol bytes changed"):
        changed_protocol.supply_world_beacon(WORLD_BEACON)

    changed_machine = _sealed(
        tmp_path / "changed-machine", wire_runtimes
    )
    machine = bytearray(changed_machine.machine_path.read_bytes())
    machine[64] ^= 1
    machine[MACHINE_HASH_OFFSET:] = sha256(
        machine[:MACHINE_HASH_OFFSET]
    ).digest()
    changed_machine.machine_path.write_bytes(machine)
    with pytest.raises(ProtocolViolation, match="machine bytes changed"):
        changed_machine.poison_and_delete_source()


def test_event_files_form_a_hash_chain(tmp_path: Path) -> None:
    rehearsal = _sealed(tmp_path / "event-chain")
    rehearsal.poison_and_delete_source()
    previous = bytes(32).hex()
    for path in sorted(rehearsal.root.joinpath("events").glob("*.json")):
        payload = path.read_bytes()
        row = json.loads(payload)
        assert row["previous_event_sha256"] == previous
        previous = sha256(payload).hexdigest()
