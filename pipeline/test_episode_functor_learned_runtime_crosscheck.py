from __future__ import annotations

import sys
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from episode_functor_machine import (  # noqa: E402
    HardFunctorKeys,
    HardFunctorMachine,
    LearnedFunctorWireSpec,
    MAX_ACTIONS,
    MAX_OBSERVERS,
    MAX_STATES,
)
from pipeline.episode_functor_seal_protocol import AbstractCoordinate  # noqa: E402
from pipeline.episode_functor_wire_protocol import (  # noqa: E402
    decode_transcript,
    encode_query_panel,
)
from pipeline.test_episode_functor_runtime_crosscheck import run_both  # noqa: E402


C_SOURCE = ROOT / "tools" / "episode_functor_runtime_c.c"
RUST_SOURCE = ROOT / "tools" / "episode_functor_runtime_rust.rs"


def _key_rows(
    values: tuple[int, ...],
    maximum: int,
) -> torch.Tensor:
    rows = torch.zeros((1, maximum, 8), dtype=torch.uint8)
    for index, value in enumerate(values):
        rows[0, index] = torch.tensor(
            tuple(value.to_bytes(8, "little")),
            dtype=torch.uint8,
        )
    return rows


def _learned_machine() -> tuple[HardFunctorMachine, HardFunctorKeys]:
    state_active = torch.zeros((1, MAX_STATES), dtype=torch.uint8)
    action_active = torch.zeros((1, MAX_ACTIONS), dtype=torch.uint8)
    observer_active = torch.zeros((1, MAX_OBSERVERS), dtype=torch.uint8)
    state_active[:, :8] = 1
    action_active[:, :3] = 1
    observer_active[:, :2] = 1
    transitions = torch.zeros(
        (1, MAX_ACTIONS, MAX_STATES),
        dtype=torch.uint8,
    )
    for action, relation in enumerate(
        (
            (1, 2, 3, 4, 5, 6, 7, 0),
            (0, 2, 1, 3, 4, 6, 5, 7),
            (4, 5, 6, 7, 0, 1, 2, 3),
        )
    ):
        transitions[0, action, :8] = torch.tensor(
            relation,
            dtype=torch.uint8,
        )
    observations = torch.zeros(
        (1, MAX_OBSERVERS, MAX_STATES),
        dtype=torch.uint8,
    )
    observations[0, 0, :8] = torch.tensor(
        (0, 0, 1, 1, 2, 2, 3, 3),
        dtype=torch.uint8,
    )
    observations[0, 1, :8] = torch.tensor(
        (3, 2, 1, 0, 3, 2, 1, 0),
        dtype=torch.uint8,
    )
    machine = HardFunctorMachine(
        state_active=state_active,
        action_active=action_active,
        observer_active=observer_active,
        action_next=transitions,
        observer_answer=observations,
    )
    keys = HardFunctorKeys(
        state_keys=_key_rows(tuple(range(101, 109)), MAX_STATES),
        action_keys=_key_rows((201, 202, 203), MAX_ACTIONS),
        observer_keys=_key_rows((301, 302), MAX_OBSERVERS),
    )
    return machine, keys


@pytest.fixture(scope="session")
def learned_runtimes(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Path]:
    cc = shutil.which("cc")
    rustc = shutil.which("rustc")
    if cc is None or rustc is None:
        pytest.skip("strict C and Rust compilers are required")
    root = tmp_path_factory.mktemp("episode_functor_learned_crosscheck")
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
            str(C_SOURCE),
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
            str(RUST_SOURCE),
            "-o",
            str(rust_runtime),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {"c": c_runtime, "rust": rust_runtime}


def test_primary_k8_y4_machine_executes_identically_in_c_and_rust(
    learned_runtimes: dict[str, Path],
    tmp_path: Path,
) -> None:
    machine, keys = _learned_machine()
    wire = machine.deployed_wire(keys, 0)
    coordinates = tuple(
        AbstractCoordinate(
            world=0,
            start=start,
            actions=actions,
            observer=observer,
            renderer=0,
        )
        for start, actions, observer in (
            (0, (), 0),
            (3, (0,), 1),
            (7, (2, 1), 0),
            (5, (1, 0, 2), 1),
            (2, (0, 1, 0, 2, 2), 0),
            (6, (2, 0, 1, 2, 0, 1, 1), 1),
        )
    )
    queries = encode_query_panel(
        wire,
        LearnedFunctorWireSpec(),
        coordinates,
    )
    c_transcript, rust_transcript = run_both(
        learned_runtimes,
        tmp_path,
        wire,
        queries,
    )
    assert c_transcript == rust_transcript
    records = decode_transcript(c_transcript, wire, queries)
    for coordinate, record in zip(coordinates, records, strict=True):
        state = coordinate.start
        for action in coordinate.actions:
            state = int(machine.action_next[0, action, state])
        assert record.final_state_slot == state
        assert record.final_state_key == int.from_bytes(
            bytes(keys.state_keys[0, state].tolist()),
            "little",
        )
        assert record.answer == int(
            machine.observer_answer[
                0,
                coordinate.observer,
                state,
            ]
        )
