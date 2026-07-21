# R12 Closure-Tied Action Algebra

## Status

Pre-neural CPU mechanics and theory. No board seed, learned model, H100 job, or
capability result is authorized by this document.

## Evidence-aligned hypothesis

Shohin repeatedly learns useful local representations and atomic operations,
then loses them at the write/commit/consume boundary. A decodable residual, an
exact external motor, or a controller alone is insufficient. The necessary
local invariant is a model-owned causal transition whose committed output is
the exact object consumed by the next step.

Closure-Tied Action Algebra (CTAA) uses one shared low-rank bilinear core for:

```text
action x state  -> next state
action x action -> composed action
```

For model-compiled actions `a` and `b`, state `s`, tied transition `rho`, and
action composition `mu`, the learned system must satisfy:

```text
rho(mu(b, a), s) = rho(b, rho(a, s))
```

The committed categorical state is passed directly to the next tied update.
There is no generated-text parse, integer parse, token re-embedding, KV-history
bypass, host arithmetic, verifier feedback, retry, or output-to-state channel.
The neural falsifier must learn a model-owned absorbing STOP. The CPU mechanics
below provide only a reference interpreter with a fixed halt boundary; they do
not demonstrate learned halt or source deletion. The terminal reader sees only
the committed state and a late query.

This is not claimed as a new computational primitive. Finite action monoids,
bilinear representations, tied recurrence, categorical commits, and persistent
excitation are established components. The open claim is narrower: whether
tying action composition to action application, while training a behaviorally
separating transition basis, produces a measurable long-horizon learnability
advantage over a parameter/state/FLOP-matched generic recurrence.

## CPU reference algebra

The scoreless reference uses all `3^3 = 27` total copy maps on three categorical
positions and all `3^3 = 27` states. It exhausts:

- 729 atomic action/state cells;
- 19,683 action-pair/state closure checks;
- 531,441 action-triple/state associativity checks;
- noncommuting order twins;
- all value alpha permutations and position-storage reindexings;
- a limited, frozen continuation/query signature that separates every state;
- a packet interface that accepts no source object;
- every interpreter halt boundary under every legal one-action suffix; and
- deterministic donor-state suffix replay.

Every CPU gate must pass before neural source is designed. Passing proves only
that the target algebra and interventions are coherent.

## Neural falsifier boundary

This document does not yet preregister or authorize the neural experiment. A
separate source-frozen protocol must bind, before any board or training seed is
drawn: the number of independent seeds; board and operation-split sizes;
aggregation rule and confidence intervals; exact train/development/confirmation
custody; identical data, initialization, optimizer, update, reader, and tuning
budgets; total active and trainable parameters; measured training and inference
FLOPs; and the exclusion of fresh operations from every excitation and query
basis. No neural result may be scored from the mechanics contract alone.

The first neural test must use the protected raw-300k Shohin trunk causally,
not add it only to a parameter ledger. The trunk has exactly 125,081,664
parameters, so the neural mechanism may add at most 24,918,335 parameters and
the complete system may contain at most 149,999,999 parameters. Source
definitions enter through Shohin residuals; zero, shuffle, and cross-example
residual transplants are mandatory.

Training may include atomic transitions and two-action closure identities. It
may not include long-program trajectories, terminal answers, execution repair,
or query-conditioned state. A frozen separating continuation/query basis will
define behavioral transition signatures. Curriculum allocation may maximize
the minimum non-gauge information direction of the transition Jacobian, but
selection cost, labels, and FLOPs must be charged.

Matched arms must include:

1. CTAA with closure and behavioral persistent excitation;
2. parameter/state/FLOP-matched generic tied recurrence;
3. CTAA without closure loss;
4. CTAA with uniform rather than excitation-weighted allocation; and
5. shuffled action-law labels.

Every generic-recurrence control must match treatment trainable parameters
within 0.1%, measured training FLOPs within 5%, and committed-state bytes
exactly. Any selection or excitation estimator is charged to the treatment's
compute and supervision budget.

The future board must use fresh operations and renderers, model-owned schedule
and halt, source deletion before execution, and unseen depths through at least
32. Advancement requires at least 99.9% conditional transition accuracy in
every stratum, 90% exact depth-32 chains, 95% donor-state following, exact
alpha/source-poison/query-isolation/STOP invariance, and a preregistered
depth-16 advantage of at least 10 percentage points over every favorable neural
control. If the matched generic recurrence comes within 3 percentage points of
CTAA, the specific closure-tied attribution is rejected even if both systems
score highly. A pass would establish bounded learned action algebra and
autonomous causal-state reuse, not unrestricted reasoning.
