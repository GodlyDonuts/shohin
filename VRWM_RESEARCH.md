# Verified Recursive Working Memory (VRWM)

## Status

**Research prototype. Not trained, not promoted, and not evidence of general reasoning.**

VRWM is a constrained context-scaling experiment for Shohin. It keeps the
125.1M model's native 2,048-token context and avoids changing the protected
pretraining architecture. A controller gives the model one instruction and a
constant-size canonical working-memory capsule. The model must emit the next
capsule. The controller forwards exactly that model-emitted state to the next
turn and retrieves the next instruction from an external program store.

The controller does **not** repair a state, choose a best sample, execute a
replacement answer, or inject a gold intermediate value. A malformed or wrong
state terminates the rollout. That boundary makes closed-loop success a useful
falsifiable signal rather than a hidden solver.

## Why V7 Was Insufficient

V7 trained static `write`, `repair`, and `reuse` prompt contracts. Its state
often contained the terminal answer, and its held-out score rose while an
independent interview remained 0/8 for compact reuse. It learned a response
format, not an autoregressive transition policy.

VRWM instead uses one repeated transition contract:

```text
Working memory: wm:a=3;b=-2
Instruction: add the current value of b to a.
Answer: wm:a=1;b=-2
```

The next turn receives only that emitted state and the next instruction. The
program itself can be arbitrarily long because the controller retrieves one
operation at a time; the model's working context remains constant size.

## Research Hypotheses and Falsification

1. **Transition learning:** after isolated SFT, exact one-step state accuracy
   must rise on unseen values and instruction combinations.
2. **Closed-loop memory:** a state emitted by the model must be valid and
   correct at every transition. Any error fails the whole program.
3. **Length extrapolation:** training programs have at most four transitions;
   evaluation programs have 8, 16, and 32. Passing only four-step episodes is
   not context scaling.
4. **Readout:** after a successful model-generated rollout, the model must read
   a requested variable from its own final memory. The controller never produces
   the answer for it.
5. **Transfer boundary:** passing the synthetic protocol would show a narrow
   executable working-memory skill, not general language reasoning. Broad
   interaction, benchmark, and code gates remain required.

## Relation to Existing Work

Segment-recurrent and compressed-memory transformers already exist, including
[Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881) and
[Associative Recurrent Memory Transformer](https://arxiv.org/abs/2407.04841).
VRWM does **not** claim to be the first memory architecture. Its practical
contribution here is a small-model, architecture-preserving, interpreter-checked
text-memory protocol with explicit closed-loop and length-extrapolation gates.
ReAct also motivates separating a model's reasoning trace from actions against
an external environment, but VRWM's controller has no answer-producing tool and
is deliberately restricted to memory transport.

## Admission and Experiment Plan

1. Generate a frozen, solver-checked VRWM train JSONL plus held-out episode
   manifest. Audit prompt overlap, malformed rows, duplicates, and exact token
   packing before any GPU job.
2. Measure the raw 180k checkpoint on the held-out episodes. This establishes
   whether SFT creates a new behavior rather than revealing one already present.
3. Run a small isolated SFT from `best_step180000.pt`; never touch the flagship.
   A two-GPU allocation may be used for independent ablations or an audited DDP
   SFT implementation, but not by wasting a second GPU on a single-process job.
4. Require improvement on all three: one-step exact state, closed-loop 8/16/32
   transitions, and independent direct interaction. A format-only score is a
   rejection.
5. Only after a real gain, extend the protocol to learned program segmentation
   and broader symbolic/code tasks. Do not claim general or latent reasoning
   from the arithmetic-state experiment.
