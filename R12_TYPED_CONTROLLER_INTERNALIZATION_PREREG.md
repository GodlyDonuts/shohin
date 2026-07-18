# R12 Typed Controller Internalization

**Status:** ACTIVE EXPERIMENT. Capability improvement attempt from the strongest
current foothold (source-scheduled raw-260k confirmation: 115/256 scheduled vs
9/256 whole / 16/256 direct). Not an R12 novelty claim. Documents a training
protocol that steals the SSC taxonomy controller shape and ACW's discrete
register discipline.

**Protocol:** `R12-TYPED-CONTROLLER-v1`

## 1. Starting from current best

Immutable confirmation `689542` established:

- local arithmetic executor exists under an external schedule (115/256);
- whole free-form decode scores 9/256 but *reaches* the answer in 45/256 before
  continuation/parser destruction;
- 214/256 whole responses loop; 0/1920 calls emit EOS;
- updater-likelihood for naive internalization is negative.

Taxonomy implication (verbatim intent): train a typed controller carrying
`(state, next_operation, operand, cursor, done)` with one op per transition and
an explicit DONE policy, evaluated as one uninterrupted model call.

## 2. Mechanism (stolen pieces)

| Source | What is reused |
|---|---|
| SSC taxonomy | controller fields, DONE/EOS policy, family set |
| ACW | hard discrete registers, one-write discipline, source-deleted state |
| DRS | single-transition competence as the executor substrate |
| Problem/Work renderer | frozen SSC format that worked |

Not reused: free-form CoT, latent soft tokens, broad teacher SFT mixes.

## 3. Interface

Single-line register file:

```text
state=<int>; ops=<op> <arg> | ...; cursor=<int>
```

Model completion (one or more lines, then EOS):

```text
<op> <arg> -> <next>; cursor=<int>; done=<0|1>
...
answer=<int>
```

Training mixes:

1. **atomic** — one transition from gold state (keeps DRS/SSC executor);
2. **rollout** — full multi-step episode from initial state to `done=1` + answer;
3. **resume** — mid-cursor continuation (forces cursor use, not bag-of-ops).

## 4. Gates (locked before scores)

From immutable `best_step200000.pt`, one epoch, isolated output:

| Metric | Floor |
|---|---:|
| Typed held-out final accuracy | ≥ 0.35 |
| Typed − raw-direct on same held-out | ≥ +0.10 |
| Typed on frozen SSC question set (re-rendered) | ≥ 0.30 |
| Atomic step exactness on held-out | ≥ 0.70 |
| Fraction of typed rollouts that emit `done=1` before cap | ≥ 0.80 |

Fail any → reject. No threshold shopping.

## 5. Explicit non-claims

Passing shows only that a discrete controller contract can be internalized
enough to compose the already-present executor without an external scheduler.
It does not establish general reasoning, language bridging, or ACW Track C.

## 6. Artifacts

```text
artifacts/r12/typed_controller_v1/
  train.jsonl
  heldout.jsonl
  audit.json
  sft/   # isolated checkpoint
  eval_*.json
  decision.json
```
