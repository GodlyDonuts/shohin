from __future__ import annotations

from pipeline.prepare_episode_functor_hankel_qualification import (
    ROOT,
    _runtime_source_paths,
)


def test_runtime_source_closure_excludes_board_and_oracle() -> None:
    relative = {
        str(path.relative_to(ROOT))
        for path in _runtime_source_paths()
    }
    assert "train/landlock_stage_exec.py" in relative
    assert "train/run_episode_functor_hankel_canary.py" in relative
    assert "pipeline/episode_functor_identifiable_board.py" not in relative
    assert (
        "pipeline/episode_functor_qualification_supervisor.py"
        not in relative
    )
    assert (
        "pipeline/prepare_episode_functor_hankel_qualification.py"
        not in relative
    )
