#!/usr/bin/env python3
"""Materialize physically disjoint EPISODE workspace custody inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pipeline.episode_action_binding_board as episode_board
import pipeline.episode_workspace_custody as custody_module
import pipeline.generate_episode_action_binding_corpus as generator_module
from pipeline.episode_workspace_custody import (
    DEFAULT_CUSTODY_BUNDLE,
    DEFAULT_SOURCE_CORPUS,
    committed_source_receipt,
    materialize_custody_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_CUSTODY_BUNDLE)
    parser.add_argument("--expected-source-sha256", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_receipt = committed_source_receipt(
        Path(__file__).resolve(),
        args.expected_source_sha256,
        (
            Path(custody_module.__file__),
            Path(episode_board.__file__),
            Path(generator_module.__file__),
        ),
    )
    receipt = materialize_custody_bundle(
        args.source,
        args.output,
        generator_source_receipt=source_receipt,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.absolute()),
                "custody_manifest_sha256": receipt["custody_manifest_sha256"],
                "counts": receipt["counts"],
                "pretraining_started": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
