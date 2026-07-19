# R12 S6 Contextual Affine Law Induction Preregistration

**Status:** frozen before neural implementation, board generation, fit, model
access, or score. Exhaustive CPU mechanics and collapse testing are authorized.

**Claim class:** bounded unseen-law induction and recurrent categorical execution.
This is not an unrestricted natural-language, planning, or general-reasoning
claim.

## 1. Question

S5 proves that a 4,934-parameter neural generator can replace a hand-authored
local action table and recurrently compose six learned primitive cells. Its
operation semantics are nevertheless fixed before inference: every operation is
one of twelve known left/right language atoms, and the runtime supplies the
parsed amount and bounded invocation schedule.

S6 asks a strictly stronger question:

> Can a learned module infer a previously untrained operation law from a minimal
> identifying context, apply that law to a source-deleted categorical state, and
> recurrently compose it through unseen programs?

Increasing the S5 generator width is not a treatment. S5 already exactly matches
the host upper bound, and its input domain has only six cells. S6 spends capacity
on a missing function: conditional law induction.

## 2. Capability Object

Let `m` be an odd prime and let positions be the finite field `Z_m`. A list state
is a permutation of `m` distinct identities. An operation law is indexed by

```
(a, b), where a in Z_m \ {0} and b in Z_m,
```

and maps a target's current position `x` to

```
d_(a,b)(x) = a*x + b mod m.
```

The state update removes the selected identity from its current position and
inserts it at `d_(a,b)(x)`. This is a deterministic update on the categorical
assignment register. Programs may use multiple operation laws and are generally
order-sensitive because pop-insert updates change later source positions.

The law parameters `(a, b)` are never provided to the learned module. A law card
contains exactly two demonstrations:

```
0 -> y0
1 -> y1
```

where `y0 = b` and `y1 = a + b mod m`. At each atomic application the treatment
receives only `(m, y0, y1, x)` and predicts the destination position. It never
receives the source text, identity name, law index, `(a,b)`, recurrent-program
label, final state, query answer, development row, or confirmation row.

## 3. Identifiability Theorem

**Theorem 1 (two-witness affine identification).** For prime `m`, every valid
law card `(m, y0, y1)` with `y1 != y0` identifies exactly one affine law:

```
b = y0
a = y1 - y0 mod m.
```

Conversely, one witness `0 -> y0` is insufficient: it fixes `b` while leaving
all `m-1` nonzero slopes possible.

**Proof.** Substituting `x=0` gives `b=y0`. Substituting `x=1` then gives
`a+b=y1`, hence `a=y1-y0 mod m`. Nonzero `a` follows from `y1 != y0`. With only
`x=0`, every nonzero `a` produces the same observation. QED.

**Corollary 1 (unseen-law execution).** A single scale-uniform destination rule
that reconstructs `(a,b)` from the two witnesses can execute every one of the
`m(m-1)` laws without law-specific parameters.

**Corollary 2 (composition).** If the same induced destination rule is tied
across events, recurrent application implements the ordered product of the
corresponding pop-insert state updates. No recurrent trajectory labels are
required to define the result.

This theorem establishes identifiability, not neural learnability. The neural
experiment tests whether the candidate learns the shared rule rather than a
table over admitted training laws.

## 4. Resource Boundary And Exact Collapse

At fixed `m`, any finite board can be solved by a lookup table. S6 therefore
makes no claim of separation from arbitrary static circuits. The named
memorization comparator stores a destination for every `(law, x)` pair, requiring
`m^2(m-1)` categorical entries at scale `m`. The affine representation stores
two field elements per law plus one shared application rule.

A hand-authored affine decoder computes the answer exactly with less learned
capacity than the treatment. It is the favorable host ceiling and prevents S6
from claiming a new mathematical primitive. The permitted claim is narrower:

- the treatment has no per-law trainable parameters;
- development laws are absent from all training targets;
- one tied learned rule must infer and execute those laws from their cards;
- a matched law-ID memorizer must fail on new IDs;
- deranging the card while preserving all tensor shapes must destroy execution.

The exhaustive CPU falsifier must establish before neural implementation:

1. card uniqueness for every law at `m in {5, 7, 11, 13}`;
2. one-witness ambiguity of exactly `m-1` laws;
3. exact reconstruction and destination closure for every law and position;
4. exact pop-insert permutation closure;
5. noncommutative order twins with a separating late query at every scale;
6. mutually disjoint train, development, and reserved-confirmation law sets;
7. absence of any law-ID or `(a,b)` field in treatment inputs; and
8. a complete retained-bit, parameter, source-access, and external-execution
   ledger.

Failure of any item rejects S6 before neural work.

## 5. Prior-Art Boundary

The broad ingredients are established: neural program interpreters, recurrent
algorithm learners, neural algorithmic reasoning, conditional program induction,
and learned transition/world models all predate S6. A law card is also a form of
in-context specification, and a transformer conditioned on it is not a new
primitive. Relevant primary references include:

- Neural Programmer-Interpreters: https://arxiv.org/abs/1511.06279
- Neural GPUs Learn Algorithms: https://arxiv.org/abs/1511.08228
- Neural Algorithmic Reasoning: https://arxiv.org/abs/2105.02761
- Learning to Theorize the World from Observation:
  https://arxiv.org/abs/2605.03413
- Slots, Transitions, Loops:
  https://arxiv.org/abs/2606.12316
- A Symbolic Neural CPU:
  https://arxiv.org/abs/2607.10021

S6 does not claim those components as novel. Its possible contribution is the
specific data-minimal, generator-factored protocol: disjoint law-level holdout,
minimal identifying witnesses, zero recurrent supervision, source-deleted tied
execution, independent causal card interventions, and sealed confirmation behind
a sub-150M Shohin system.

## 6. Frozen Law Split

Admitted moduli are `5`, `7`, and `11`. Modulus `13` is a scale diagnostic and
cannot contribute training targets or primary promotion credit.

For each valid `(m,a,b)`, define

```
bucket = sha256("s6-law-v1|m|a|b").digest()[0] mod 5.
```

- buckets `2,3,4`: training laws;
- bucket `0`: development laws;
- bucket `1`: reserved-confirmation laws.

The builder must verify at every admitted modulus that each split is nonempty,
that all destination values and all card coordinates occur in training, and that
the three law sets are pairwise disjoint. Confirmation laws may be enumerated for
split auditing, but no confirmation programs, rows, seed, or score may be created
before development promotion.

Training consists only of atomic rows covering every position of every training
law. Repetition for optimization is allowed, but no distinct recurrent or answer
target may be added. The development board contains independently generated
depth-three-through-eight programs using development laws only, arbitrary nonce
law names, random initial assignments, late position queries, and at least two
different laws per multi-law stratum.

## 7. Sole Treatment

The treatment is a card-conditioned categorical destination predictor:

1. learned embeddings represent modulus, role, input coordinate, and output
   coordinate;
2. a small transformer reads `[LAW, SUPPORT_0, SUPPORT_1, QUERY]`;
3. the query state predicts one of the valid positions under a modulus mask;
4. a hard-forward destination drives the exact categorical pop-insert register;
5. the same predictor weights are tied across every event.

The module may use at most **8,000,000** unique trainable parameters. The complete
S4 parser plus S5 register/generator plus S6 module must remain strictly below
150,000,000 unique parameters. Unused headroom is deliberately reserved for the
later active-step/halt controller; parameter count is a ceiling, not an objective.

The S4 language parser is not part of this first law-induction claim. S6.1 uses a
categorical law/event interface so that semantic induction can be isolated from
language grounding. A pass authorizes S6.2 natural-language law cards; it does not
allow S6.1 scores to be described as unrestricted native language reasoning.

## 8. Matched Controls

- **Host affine ceiling:** exact theorem decoder plus exact categorical state.
- **Law-ID memorizer:** same or favorable parameter/compute budget, receives an
  arbitrary training-law ID and current position but no card. Every development
  ID maps to one shared OOV identity.
- **Deranged card:** unchanged treatment weights with complete cards rotated
  among development laws within the same modulus.
- **One-witness ablation:** hide `SUPPORT_1` while preserving sequence length and
  model compute.
- **State reset:** reset the assignment register before every event.
- **Untied recurrence:** favorable separate predictor copies by event depth, with
  at least treatment parameter count; it receives no development laws.

No control may receive `(a,b)`, exact destinations, host states, or final answers
at inference unless it is explicitly the host ceiling.

## 9. Development Gates

All gates are required:

1. CPU falsifier passes every obligation in section 4.
2. Treatment fits at least 99% of atomic training cells.
3. Treatment reaches at least 95% destination accuracy over all atomic cells of
   held-out development laws.
4. End-to-end development reaches at least 95% exact final state and answer.
5. Exact final state is at least 92% at every depth three through eight.
6. Treatment state and answer are each within 1 percentage point of the host
   affine ceiling.
7. Deranged-card exact state falls at least 40 percentage points.
8. One-witness exact state falls at least 30 percentage points.
9. State-reset exact state falls at least 20 percentage points.
10. Law-ID memorizer exact state trails treatment by at least 40 points.
11. At least 95% state accuracy holds in the multi-law stratum and nonce-law-name
    permutation leaves treatment outputs bit-identical.
12. The module is below 8M parameters, the whole system is below 150M, training
    has zero development/confirmation laws and zero recurrent/answer examples,
    and development access occurs exactly once.

The modulus-13 scale diagnostic is reported but not a primary gate. It may justify
a stronger later uniformity claim only if frozen before access and at least 80%
exact state without modulus-13 training targets.

Passing all primary gates qualifies exactly one independently seeded confirmation
of unchanged weights, architecture, split, decoder, controls, and thresholds.
Failure rejects S6.1 without changing promoted S5.

## 10. Claim Boundary And Next Stage

A confirmed pass would establish that Shohin's bounded reasoning stack can infer
and recurrently apply operation laws absent from training, when each law is given
through a minimal categorical identifying context. It would remove the fixed
operation-table boundary of S5 more strongly than adding capacity to the six-cell
generator.

It would not establish natural-language semantic discovery, autonomous plan
construction, model-owned law-card binding, learned replay count, learned halt,
free-form serialization, or public benchmark improvement. S6.2 must ground law
cards and operation references from whole-source language. S7 must then replace
the bounded event-list schedule with an active-step/continue/halt controller.
