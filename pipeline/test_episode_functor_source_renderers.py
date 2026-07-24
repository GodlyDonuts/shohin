from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.episode_functor_independent_world import (
    generate_independent_world,
)
from pipeline.episode_functor_source_renderers import (
    SourceRendererError,
    decode_json_events,
    decode_line_events,
    encode_line_events,
)
from pipeline.episode_functor_wire_protocol import (
    WireProtocolSpec,
    decode_deployed_machine,
    encode_deployed_machine,
)


WORLD_KWARGS = {
    "protocol_root": "56" * 32,
    "beacon_round": 30_000,
    "beacon_value": "consumed-renderer-world",
    "state_count": 5,
    "action_count": 3,
    "observer_count": 2,
    "answer_count": 5,
    "renderer_count": 1,
}


def test_json_and_line_renderers_compile_to_identical_machine() -> None:
    world = generate_independent_world(**WORLD_KWARGS)
    json_payload = world.evidence
    line_payload = encode_line_events(json_payload)
    spec = WireProtocolSpec()

    assert line_payload != json_payload
    assert decode_json_events(json_payload)["renderer_choice"] == 0
    assert decode_line_events(line_payload)["renderer_choice"] == 1
    json_machine = encode_deployed_machine(json_payload, spec)
    line_machine = encode_deployed_machine(line_payload, spec)
    assert json_machine == line_machine
    decoded = decode_deployed_machine(json_machine, spec)
    assert decoded.transitions == world.transitions
    assert decoded.observations == world.observers


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.replace(b"EFC-RAW-LINES-V1", b"BAD-RAW-LINES-V1"),
        lambda payload: payload.replace(b"\nD\t", b"\nX\t", 1),
        lambda payload: payload.replace(b"\nO\t", b"\nD\t", 1),
        lambda payload: payload[:-1],
        lambda payload: payload.replace(b"\t0\n", b"\t00\n", 1),
        lambda payload: payload + b"\n",
    ],
)
def test_line_renderer_fails_closed_on_noncanonical_mutations(
    mutation: object,
) -> None:
    world = generate_independent_world(**WORLD_KWARGS)
    line_payload = encode_line_events(world.evidence)
    changed = mutation(line_payload)
    assert changed != line_payload
    with pytest.raises(SourceRendererError):
        decode_line_events(changed)


def test_renderer_normalized_events_differ_only_in_renderer_tag() -> None:
    world = generate_independent_world(**WORLD_KWARGS)
    json_row = decode_json_events(world.evidence)
    line_row = decode_line_events(encode_line_events(world.evidence))
    assert json_row["demonstrations"] == line_row["demonstrations"]
    assert json_row["observations"] == line_row["observations"]
    assert json_row["renderer_choice"] == 0
    assert line_row["renderer_choice"] == 1
    assert json.dumps(json_row, sort_keys=True) != json.dumps(line_row, sort_keys=True)


def test_renderer_identity_is_enforced_by_source_syntax() -> None:
    world = generate_independent_world(**WORLD_KWARGS)
    json_row = json.loads(world.evidence)
    json_row["renderer_choice"] = 1
    forged_json = (
        json.dumps(json_row, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    with pytest.raises(SourceRendererError, match="does not match"):
        decode_json_events(forged_json)
    with pytest.raises(SourceRendererError, match="does not match"):
        encode_line_events(forged_json)


def test_renderer_rejects_non_utf8_and_unbounded_decimal() -> None:
    world = generate_independent_world(**WORLD_KWARGS)
    with pytest.raises(SourceRendererError, match="malformed"):
        decode_json_events(b"\xff")
    line_payload = encode_line_events(world.evidence)
    mutated = line_payload.replace(
        b"\t0\n",
        b"\t" + b"9" * 4_301 + b"\n",
        1,
    )
    with pytest.raises(SourceRendererError, match="noncanonical"):
        decode_line_events(mutated)


def test_source_renderer_has_no_compiler_generator_protocol_or_runtime_import() -> None:
    source = Path(
        __import__(
            "pipeline.episode_functor_source_renderers",
            fromlist=["__file__"],
        ).__file__
    ).read_text(encoding="utf-8")
    executable_source = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    for forbidden in (
        "episode_functor_independent_world",
        "episode_functor_multiworld_custody",
        "episode_functor_seal_protocol",
        "episode_functor_wire_protocol",
        "episode_functor_runtime",
        "encode_deployed_machine",
    ):
        assert forbidden not in executable_source
