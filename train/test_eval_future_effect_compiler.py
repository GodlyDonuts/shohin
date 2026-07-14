#!/usr/bin/env python3
"""Noisy-policy contracts for active future-effect evaluation."""

import torch

from eval_future_effect_compiler import infer_hypothesis
from future_distinction_cell import hypothesis_effect_codes, legal_operator_hypotheses
from future_effect_algebra import redundant_probe_bank


class OracleCompiler:
    def __init__(self, code):
        self.code = code

    def predict_effect(self, _output, state, query):
        states, queries = redundant_probe_bank(dtype=state.dtype, device=state.device)
        state_index = int((states - state).abs().sum(-1).argmin().item())
        query_index = int((queries - query).abs().sum(-1).argmin().item())
        return self.code[query_index * states.shape[0] + state_index] / 64.0


def main():
    hypotheses = legal_operator_hypotheses(range(1, 10), dtype=torch.float32)
    states, queries = redundant_probe_bank(dtype=torch.float32)
    codes = hypothesis_effect_codes(hypotheses, states, queries)
    for target in (0, 8, 17, len(hypotheses) - 1):
        compiler = OracleCompiler(codes[target])
        active, trace = infer_hypothesis(
            compiler, {}, codes, states, queries,
            policy="active", steps=3, seed=3, target=target, plausible_count=64,
        )
        oracle, _ = infer_hypothesis(
            compiler, {}, codes, states, queries,
            policy="oracle", steps=3, seed=3, target=target, plausible_count=64,
        )
        assert active == target
        assert oracle == target
        assert len(trace) == 3
    print("future-effect evaluator policy tests passed")


if __name__ == "__main__":
    main()
