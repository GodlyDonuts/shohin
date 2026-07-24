from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.episode_functor_runtime_custody import write_json_fsync
from run_episode_functor_hankel_canary import (
    HankelCanaryError,
    _write_complete_receipt,
    verify_complete_output,
)


def test_complete_receipt_authenticates_exact_output(tmp_path: Path) -> None:
    output = tmp_path / "result"
    output.mkdir()
    write_json_fsync(output / "canary_report.json", {"decision": "test"})
    report_sha256 = _write_complete_receipt(output)
    complete = verify_complete_output(
        output,
        expected_report_sha256=report_sha256,
    )
    assert complete["files_sha256"][0]["sha256"] == report_sha256


def test_complete_receipt_rejects_mutation_and_extra_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    output.mkdir()
    write_json_fsync(output / "canary_report.json", {"decision": "test"})
    _write_complete_receipt(output)
    (output / "canary_report.json").write_text(
        json.dumps({"decision": "mutated"}),
        encoding="ascii",
    )
    with pytest.raises(HankelCanaryError, match="receipt differs"):
        verify_complete_output(output)

    output = tmp_path / "extra"
    output.mkdir()
    write_json_fsync(output / "canary_report.json", {"decision": "test"})
    _write_complete_receipt(output)
    (output / "extra.txt").write_text("x", encoding="ascii")
    with pytest.raises(HankelCanaryError, match="closure differs"):
        verify_complete_output(output)
