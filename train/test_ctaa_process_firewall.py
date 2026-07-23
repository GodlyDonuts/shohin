from __future__ import annotations

import json
import inspect
from pathlib import Path
import subprocess
import sys

import torch

from ctaa_neural_core import ClosureFeatureTransitionCore, OuterProductTransitionControl
from ctaa_packet_io import write_packet_file, write_query_file
from ctaa_trunk_compiler import HardCTAAPacket, HardCTAAQuery
from run_ctaa_packet_executor import CORE_SCHEMA, EXECUTION_SCHEMA


def test_fresh_process_execution_and_late_query_receive_no_source(tmp_path: Path) -> None:
    torch.manual_seed(17)
    core = ClosureFeatureTransitionCore().eval()
    with torch.no_grad():
        for parameter in core.parameters():
            parameter.zero_()
        core.network[-1].bias.reshape(3, 3)[:, 0] = 1.0
    core_path = tmp_path / "core.pt"
    torch.save(
        {"schema": CORE_SCHEMA, "kind": "closure_feature", "state": core.state_dict()},
        core_path,
    )
    core_path.chmod(0o444)
    packet = HardCTAAPacket(
        action_cards=torch.tensor(
            [[[1, 0, 2], [2, 2, 0], [0, 1, 1], [2, 0, 1]]],
            dtype=torch.uint8,
        ),
        initial_state=torch.tensor([[0, 0, 0]], dtype=torch.uint8),
        opcode_schedule=torch.tensor(
            [[0, 1, 4, *([0] * 38)]],
            dtype=torch.uint8,
        ),
        opcode_to_card=torch.arange(4, dtype=torch.uint8)[None],
    )
    query = HardCTAAQuery(position=torch.tensor([1], dtype=torch.uint8))
    packet_path = tmp_path / "packet.bin"
    query_path = tmp_path / "query.bin"
    execution_path = tmp_path / "execution.pt"
    answer_path = tmp_path / "answer.json"
    write_packet_file(packet_path, packet)
    assert not query_path.exists()
    train_dir = Path(__file__).resolve().parent
    subprocess.run(
        [
            sys.executable,
            str(train_dir / "run_ctaa_packet_executor.py"),
            "--packet",
            str(packet_path),
            "--core",
            str(core_path),
            "--output",
            str(execution_path),
        ],
        check=True,
        cwd=train_dir.parent,
        capture_output=True,
        text=True,
    )
    assert execution_path.exists()
    assert execution_path.stat().st_mode & 0o222 == 0
    assert not query_path.exists()
    write_query_file(query_path, query)
    subprocess.run(
        [
            sys.executable,
            str(train_dir / "run_ctaa_late_query.py"),
            "--execution",
            str(execution_path),
            "--query",
            str(query_path),
            "--output",
            str(answer_path),
        ],
        check=True,
        cwd=train_dir.parent,
        capture_output=True,
        text=True,
    )
    execution = torch.load(execution_path, map_location="cpu", weights_only=True)
    answer = json.loads(answer_path.read_text())
    direct = packet.execute_dual(core)
    assert execution["schema"] == EXECUTION_SCHEMA
    assert set(execution) == {
        "schema",
        "core_kind",
        "packet_sha256",
        "core_sha256",
        "state_route",
        "halted",
        "composed_cards",
        "composed_states",
    }
    assert torch.equal(execution["state_route"].long(), direct.state_route.states)
    expected = query.position.new_tensor(
        [int(direct.state_route.states[0, -1, 1])]
    ).tolist()
    assert answer["answers"] == expected
    assert execution_path.stat().st_mode & 0o222 == 0
    assert answer_path.stat().st_mode & 0o222 == 0


def test_source_blind_cli_has_no_source_or_model_argument() -> None:
    train_dir = Path(__file__).resolve().parent
    executor = (train_dir / "run_ctaa_packet_executor.py").read_text()
    query = (train_dir / "run_ctaa_late_query.py").read_text()
    for source in (executor, query):
        assert "--source" not in source
        assert "--tokenizer" not in source
        assert "from model import" not in source


def test_late_query_rejects_route_disagreement(tmp_path: Path) -> None:
    execution_path = tmp_path / "execution.pt"
    query_path = tmp_path / "query.bin"
    answer_path = tmp_path / "answer.json"
    state_route = torch.zeros((1, 42, 3), dtype=torch.uint8)
    composed_states = state_route.clone()
    composed_states[:, -1, 0] = 1
    halted = torch.zeros((1, 42), dtype=torch.bool)
    halted[:, 3:] = True
    torch.save(
        {
            "schema": EXECUTION_SCHEMA,
            "state_route": state_route,
            "composed_states": composed_states,
            "halted": halted,
        },
        execution_path,
    )
    write_query_file(
        query_path,
        HardCTAAQuery(position=torch.tensor([0], dtype=torch.uint8)),
    )
    train_dir = Path(__file__).resolve().parent
    result = subprocess.run(
        [
            sys.executable,
            str(train_dir / "run_ctaa_late_query.py"),
            "--execution",
            str(execution_path),
            "--query",
            str(query_path),
            "--output",
            str(answer_path),
        ],
        check=False,
        cwd=train_dir.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "routes disagree" in result.stderr
    assert not answer_path.exists()


def test_executor_core_call_has_no_schedule_or_future_event_argument() -> None:
    class SpyCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        def forward(self, action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            self.calls.append((tuple(action.shape), tuple(state.shape)))
            return torch.nn.functional.one_hot(state, 3).float().log()

    from ctaa_neural_core import execute_streamed_state_route

    core = SpyCore()
    cards = torch.tensor(
        [[[1, 0, 2], [2, 2, 0], [0, 1, 1], [2, 0, 1]]],
        dtype=torch.long,
    )
    schedule = torch.tensor([[0, 1, 4, *([3] * 38)]], dtype=torch.long)
    initial = torch.tensor([[0, 1, 2]], dtype=torch.long)
    execute_streamed_state_route(core, 3, cards, schedule, initial)
    assert len(core.calls) == 2
    assert set(core.calls) == {((1, 3), (1, 3))}


def test_production_core_objects_expose_transition_only_forward_interfaces() -> None:
    for core_type in (ClosureFeatureTransitionCore, OuterProductTransitionControl):
        assert not hasattr(core_type, "execute_hard")
        assert not hasattr(core_type, "execute_dual_hard")
        assert tuple(inspect.signature(core_type.forward).parameters) == (
            "self",
            "left_ids",
            "right_ids",
        )
