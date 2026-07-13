# Verified Recursive Working Memory (VRWM)

## Status

**Research prototype. Isolated r3/r4 SFTs are trained; r5 repair SFT is running. None is promoted or
evidence of general reasoning.**

The raw control is complete on the hash-verified 180k checkpoint. Across five
prompt-disjoint regimes (five episodes each), it scored **0/25** exact first
transitions and **0/25** closed-loop programs. The raw model commonly copies
the literal `wm:a=<integer>;b=<integer>` template or emits an untyped arithmetic
fragment. This establishes that any later exact state behavior is learned by
the isolated experiment rather than already latent in the checkpoint. Canonical
artifact: `artifacts/eval_history/vrwm_raw180k_local_mps_r1.json`, MD5
`e135513d67eee427cb7089df0dd231ff`.

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

## Measured Results

- Raw H100 full p80 control: **0/400** exact first transitions and **0/400** closed-loop programs.
- r3 one-epoch SFT (497,274 rows): **43/400** closed-loop programs on default prompts but **0/50** on
  held-out paraphrase prompts. Reject as template-bound.
- r4 state-only (513,902 rows): **32/400** default and **2/400** semantic closed-loop programs.
- r4 deterministic-scratch (513,902 rows): **120/400** default and **21/400** semantic closed-loop
  programs. Default split detail: length 4 **58/80**, length 8 **32/80**, length 16 **12/80**, length 32
  **3/80**, and wide-range length 8 **15/80**. Every successful readout was conditioned on a model-created
  successful rollout. This is the first real narrow executable-state signal, but its low semantic and
  long-horizon transfer explicitly fails the generalization bar.

r5 adds two proposal-check examples per transition and trains the same model to emit a corrected canonical
state. Its evaluation compares draft and one model-only repair pass on the same frozen p80 episodes. The
controller records both responses and never calculates, selects, or substitutes a state. To advance, r5
must improve semantic closed-loop and long-horizon performance over r4 scratch; default-template movement
alone is not enough.

## Current Inputs

- Candidate r1 is rejected: its small prompt space produced 166,766 normalized
  duplicate prompts.
- Candidate r2 passes quality but has only 2,263 packed sequences, below the
  5,000-sequence minimum for an SFT transition-policy ablation.
- Candidate r3 is an admissible completed input: **497,274** unique rows,
  **0** malformed rows, duplicate prompts, exact evaluation rows, or 13-gram
  evaluation rows, and **18,013** packed 2,048-token sequences. Data SHA-256:
  `b2a688e1f7aa6c79dd65ed1944fa5dc00cd022acfc793896ecf4696c94d4089f`;
  local/Newton MD5: `36a747cfdb31bebcf96fd06bb0fd3950`. The 400 held-out
  episodes reserve input values outside the train range and test 4, 8, 16, and
  32 transitions. It completed one isolated epoch as documented above.
- r4 is a controlled prompt/trace ablation with two 513,902-row branches. The state-only and scratch
  variants share a seed, train/evaluation episodes, two training prompt styles (default/paraphrase), and
  reserved semantic evaluation wording; only response form differs.
- r5 is the active repair branch: **1,409,072** rows, **68,347** packed sequences, two repair proposals
  per transition, deterministic scratch responses, and the same reserved semantic p80 episodes. Its data
  SHA-256 is `011282f032963a40b8b39ab9572808de1d3473ef2b57ef727526fb9d00985c76`.
