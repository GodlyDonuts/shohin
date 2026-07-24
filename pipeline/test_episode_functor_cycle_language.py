from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from pipeline.episode_functor_cycle_language import (
    CycleLanguageError,
    MAGIC,
    MAX_LINE_BYTES,
    MAX_LINES,
    MAX_TOKEN_COUNT,
    NORMALIZED_SCHEMA,
    RENDERER_CHOICE,
    decode_cycle_language,
    encode_cycle_language,
)
from pipeline.episode_functor_independent_world import (
    generate_independent_world,
)
from pipeline.episode_functor_wire_protocol import (
    MACHINE_SIZE,
    WireProtocolSpec,
    decode_deployed_machine,
    encode_deployed_machine,
)


STATES = (
    0x00000000000000A1,
    0x00000000000000B2,
    0x00000000000000C3,
    0x00000000000000D4,
    0x00000000000000E5,
)
ACTIONS = (
    0x0000000000000A01,
    0x0000000000000A02,
    0x0000000000000A03,
)
OBSERVERS = (
    0x0000000000000B01,
    0x0000000000000B02,
)
TRANSITIONS = (
    STATES,
    (STATES[1], STATES[0], STATES[3], STATES[4], STATES[2]),
    (STATES[1], STATES[2], STATES[3], STATES[4], STATES[0]),
)
OBSERVATIONS = (
    (0, 1, 0, 2, 1),
    (4, 3, 2, 1, 0),
)
INDEX_TRANSITIONS = (
    (0, 1, 2, 3, 4),
    (1, 0, 3, 4, 2),
    (1, 2, 3, 4, 0),
)
WORLD_KWARGS = {
    "protocol_root": "78" * 32,
    "beacon_round": 40_000,
    "beacon_value": "cycle-language-generated-world",
    "state_count": 5,
    "action_count": 3,
    "observer_count": 2,
    "answer_count": 5,
    "renderer_count": 1,
}


def _normalized_row(renderer: int = 0) -> dict[str, object]:
    demonstrations = [
        {
            "action_key": action,
            "source_key": state,
            "target_key": relation[state_slot],
        }
        for action, relation in zip(ACTIONS, TRANSITIONS, strict=True)
        for state_slot, state in enumerate(STATES)
    ]
    observations = [
        {
            "answer": answers[state_slot],
            "observer_key": observer,
            "state_key": state,
        }
        for observer, answers in zip(OBSERVERS, OBSERVATIONS, strict=True)
        for state_slot, state in enumerate(STATES)
    ]
    return {
        "demonstrations": list(reversed(demonstrations)),
        "observations": observations[::2] + observations[1::2],
        "renderer_choice": renderer,
        "schema": NORMALIZED_SCHEMA,
    }


def _canonical_json(row: object) -> bytes:
    return (
        json.dumps(
            row,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _test_local_wire_adapter(cycle_payload: bytes) -> bytes:
    """Exercise the deployed compiler's admitted cycle-language path."""

    return encode_deployed_machine(
        cycle_payload,
        WireProtocolSpec(source_renderer_count=3),
    )


def _replace_lines(
    payload: bytes,
    first: int,
    second: int,
) -> bytes:
    lines = payload.decode("ascii").splitlines()
    lines[first], lines[second] = lines[second], lines[first]
    return ("\n".join(lines) + "\n").encode("ascii")


def test_cycle_program_has_expected_structurally_distinct_canonical_grammar() -> None:
    payload = encode_cycle_language(_normalized_row())
    text = payload.decode("ascii")

    assert text.startswith(
        "EFC-CYCLE-PROGRAM-V1\n"
        "states = [s#00000000000000a1,s#00000000000000b2,"
        "s#00000000000000c3,s#00000000000000d4,"
        "s#00000000000000e5];\n"
        "actions = {\n"
    )
    assert (
        "  a#0000000000000a01 := "
        "cycle[s#00000000000000a1] * "
        "cycle[s#00000000000000b2] * "
        "cycle[s#00000000000000c3] * "
        "cycle[s#00000000000000d4] * "
        "cycle[s#00000000000000e5];\n"
    ) in text
    assert (
        "  a#0000000000000a02 := "
        "cycle[s#00000000000000a1,s#00000000000000b2] * "
        "cycle[s#00000000000000c3,s#00000000000000d4,"
        "s#00000000000000e5];\n"
    ) in text
    assert (
        "  o#0000000000000b01 := "
        "class[0]{s#00000000000000a1,s#00000000000000c3} + "
        "class[1]{s#00000000000000b2,s#00000000000000e5} + "
        "class[2]{s#00000000000000d4};\n"
    ) in text
    assert text.endswith("};\nhalt.\n")
    for event_row_term in (
        '"demonstrations"',
        '"observations"',
        "action_key",
        "source_key",
        "target_key",
        "observer_key",
        "\tD\t",
        "\tO\t",
    ):
        assert event_row_term not in text


def test_canonical_roundtrip_and_renderer_two_normalization() -> None:
    first = encode_cycle_language(_normalized_row(renderer=1))
    decoded = decode_cycle_language(first)
    second = encode_cycle_language(decoded)

    assert first == second
    assert decoded["schema"] == NORMALIZED_SCHEMA
    assert decoded["renderer_choice"] == RENDERER_CHOICE
    assert len(decoded["demonstrations"]) == len(STATES) * len(ACTIONS)
    assert len(decoded["observations"]) == len(STATES) * len(OBSERVERS)
    assert decoded["demonstrations"][0] == {
        "action_key": ACTIONS[0],
        "source_key": STATES[0],
        "target_key": STATES[0],
    }


def test_shuffled_event_rows_have_one_canonical_cycle_program() -> None:
    row_a = _normalized_row()
    row_b = copy.deepcopy(row_a)
    row_b["demonstrations"] = list(reversed(row_b["demonstrations"]))
    row_b["observations"] = list(reversed(row_b["observations"]))

    assert encode_cycle_language(row_a) == encode_cycle_language(row_b)


def test_generated_json_and_cycle_sources_compile_to_identical_machine() -> None:
    world = generate_independent_world(**WORLD_KWARGS)
    json_machine = encode_deployed_machine(world.evidence, WireProtocolSpec())
    json_row = json.loads(world.evidence)
    cycle_payload = encode_cycle_language(json_row)
    cycle_machine = _test_local_wire_adapter(cycle_payload)

    assert cycle_payload != world.evidence
    assert len(json_machine) == MACHINE_SIZE == 1_536
    assert cycle_machine == json_machine
    decoded = decode_deployed_machine(cycle_machine, WireProtocolSpec())
    assert decoded.transitions == world.transitions
    assert decoded.observations == world.observers


def test_hand_program_compiles_to_expected_gauge_tables_and_exact_wire() -> None:
    row = _normalized_row()
    json_machine = encode_deployed_machine(_canonical_json(row), WireProtocolSpec())
    cycle_machine = _test_local_wire_adapter(encode_cycle_language(row))
    decoded = decode_deployed_machine(cycle_machine, WireProtocolSpec())

    assert len(cycle_machine) == 1_536
    assert cycle_machine == json_machine
    assert decoded.state_keys == STATES
    assert decoded.action_keys == ACTIONS
    assert decoded.observer_keys == OBSERVERS
    assert decoded.transitions == INDEX_TRANSITIONS
    assert decoded.observations == OBSERVATIONS


def _noncanonical_mutations(payload: bytes) -> tuple[bytes, ...]:
    return (
        payload[:-1],
        payload.replace(b"\n", b"\r\n"),
        payload.replace(b"states = ", b"states  = ", 1),
        payload.replace(
            b"s#00000000000000a1,s#00000000000000b2",
            b"s#00000000000000b2,s#00000000000000a1",
            1,
        ),
        _replace_lines(payload, 3, 4),
        payload.replace(
            b"cycle[s#00000000000000a1,s#00000000000000b2]",
            b"cycle[s#00000000000000b2,s#00000000000000a1]",
            1,
        ),
        payload.replace(
            b"cycle[s#00000000000000a1] * cycle[s#00000000000000b2]",
            b"cycle[s#00000000000000b2] * cycle[s#00000000000000a1]",
            1,
        ),
        payload.replace(
            b"class[0]{s#00000000000000a1,s#00000000000000c3} + "
            b"class[1]{s#00000000000000b2,s#00000000000000e5}",
            b"class[1]{s#00000000000000b2,s#00000000000000e5} + "
            b"class[0]{s#00000000000000a1,s#00000000000000c3}",
            1,
        ),
        payload.replace(
            b"class[0]{s#00000000000000a1,s#00000000000000c3}",
            b"class[0]{s#00000000000000c3,s#00000000000000a1}",
            1,
        ),
        payload.replace(b"a#0000000000000a01", b"a#0000000000000A01", 1),
    )


@pytest.mark.parametrize("mutation_index", range(10))
def test_noncanonical_programs_fail_closed(mutation_index: int) -> None:
    payload = encode_cycle_language(_normalized_row())
    mutation = _noncanonical_mutations(payload)[mutation_index]

    assert mutation != payload
    with pytest.raises(CycleLanguageError):
        decode_cycle_language(mutation)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: b"\xff" + payload[1:],
        lambda payload: payload.replace(MAGIC.encode(), b"EFC-EVENT-ROWS-V1", 1),
        lambda payload: payload.replace(
            b"cycle[s#00000000000000a1]",
            b"cycle[s#00000000000000a1,s#00000000000000a1]",
            1,
        ),
        lambda payload: payload.replace(
            b"cycle[s#00000000000000e5]",
            b"cycle[s#00000000000000ff]",
            1,
        ),
        lambda payload: payload.replace(
            b" * cycle[s#00000000000000e5]",
            b"",
            1,
        ),
        lambda payload: payload.replace(
            b"class[2]{s#00000000000000d4}",
            b"class[0]{s#00000000000000d4}",
            1,
        ),
        lambda payload: payload.replace(
            b"class[2]{s#00000000000000d4}",
            b"class[2]{s#00000000000000ff}",
            1,
        ),
        lambda payload: payload.replace(
            b" + class[2]{s#00000000000000d4}",
            b"",
            1,
        ),
        lambda payload: payload.replace(
            b"class[4]",
            b"class[18446744073709551616]",
            1,
        ),
        lambda payload: payload.replace(b"class[4]", b"class[04]", 1),
        lambda payload: payload + b"trailing\n",
    ],
)
def test_malformed_programs_fail_closed(mutation: object) -> None:
    payload = encode_cycle_language(_normalized_row())
    changed = mutation(payload)

    assert changed != payload
    with pytest.raises(CycleLanguageError):
        decode_cycle_language(changed)


def test_parser_enforces_line_token_and_line_count_bounds_before_grammar() -> None:
    overlong = b"x" * (MAX_LINE_BYTES + 1) + b"\n"
    too_many_lines = b"x\n" * (MAX_LINES + 1)
    too_many_tokens = (
        MAGIC.encode("ascii")
        + b"\n"
        + (b"+ " * 800)
        + b"\n"
        + (b"+ " * 800)
        + b"\n"
        + (b"+ " * (MAX_TOKEN_COUNT - 1_500))
        + b"\n"
    )

    with pytest.raises(CycleLanguageError, match="line bound"):
        decode_cycle_language(overlong)
    with pytest.raises(CycleLanguageError, match="line bound"):
        decode_cycle_language(too_many_lines)
    with pytest.raises(CycleLanguageError, match="token bound"):
        decode_cycle_language(too_many_tokens)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.update(renderer_choice=3),
        lambda row: row["demonstrations"].pop(),
        lambda row: row["observations"].pop(),
        lambda row: row["demonstrations"][0].update(target_key=STATES[1]),
        lambda row: row["observations"][0].update(answer=True),
        lambda row: row["observations"][0].update(state_key=0xFF),
    ],
)
def test_encoder_rejects_invalid_normalized_semantics(mutate: object) -> None:
    row = _normalized_row()
    mutate(row)

    with pytest.raises(CycleLanguageError):
        encode_cycle_language(row)


def test_cycle_language_module_has_only_standard_library_imports() -> None:
    module_path = Path(
        __import__(
            "pipeline.episode_functor_cycle_language",
            fromlist=["__file__"],
        ).__file__
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots <= {"__future__", "dataclasses", "re", "typing"}
    assert not imported_roots & {
        "pipeline",
        "train",
        "tools",
    }
