"""Static custody checks for the ER-CST witness confirmation wrapper."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "train" / "jobs" / "er_cst_witness_equality_confirmation.sbatch"


def test_confirmation_job_is_single_read_and_training_free() -> None:
    text = JOB.read_text()
    required = (
        "confirm_er_cst_witness_equality.py",
        "assess_er_cst_witness_equality_confirmation.py",
        "test \"$(stat -c '%a' \"$DATA/confirmation.jsonl\")\" = 600",
        "test \"$(find \"$DATA/access\" -maxdepth 1 -type f | wc -l)\" -eq 1",
        "test ! -e \"$OUTDIR\"",
        "er_cst_witness_confirmation_*.json",
    )
    assert all(value in text for value in required)
    assert "pilot_er_cst_witness_equality.py" not in text
    assert "--seed" not in text
