# R12 S7 Learned Cayley Law Compiler Preregistration

**Status:** source/theory freeze before any S7 score-bearing board  
**Predecessor:** S6 generic contextual transformer, formally rejected  
**Claim class:** bounded unseen-law induction under an explicit cyclic-group prior

## 1. Target failure

S6 proved that mathematical identifiability is not enough. Two demonstrations
uniquely identify every affine law, both neural arms fit all 961 train cells,
yet the generic transformer reaches only 24.528% held-out atomic destinations
and 8.154% recurrent exact state. The next mechanism must force compositional
reuse rather than reward a larger lookup surface.

## 2. Mechanism

For each prime modulus `m`, observed location symbols are hidden behind an
arbitrary bijection `pi_m`. S7 learns only:

1. the observed-symbol successor generator `S_m`; and
2. the observed symbol representing latent zero.

It receives a new law card with demonstrations `0->y0`, `1->y1` and a current
symbol `x`. It does not receive slope, intercept, a law ID, a destination table,
or recurrent examples.

The compiler uses only the learned successor, equality, and bounded recurrent
application:

1. walk from `y0` to `y1` while walking from zero in parallel; the second walk
   is the slope symbol;
2. walk a cursor from zero to `x`;
3. for each cursor step, walk a destination by the inferred slope;
4. return the destination when the cursor equals `x`.

No `%`, multiplication, subtraction, affine coefficient recovery, or
per-law table is used by the treatment compiler.

## 3. Exact theorem

Let `S_m(s) = pi_m^-1(pi_m(s)+1 mod m)` and let
`z_m = pi_m^-1(0)`. For any bijective affine law

```text
d(r) = a*r+b mod m, a != 0,
```

encode its card as

```text
y0 = pi_m^-1(b)
y1 = pi_m^-1(a+b)
x  = pi_m^-1(r).
```

The first parallel walk terminates after exactly `a` successor steps and
therefore represents the slope without exposing its integer. The outer walk
terminates after exactly `r` successor steps; each outer step advances the
destination by `a` successors. The returned symbol is

```text
pi_m^-1(b+r*a mod m) = pi_m^-1(d(r)).
```

Thus an exact learned generator and zero anchor imply exact unseen-law atomic
execution. Recurrent list-state execution follows by induction because each
event's destination is exact and pop-insert is closed on permutations.

## 4. What is and is not learned

Learned:

- 23 successor cells across moduli 5, 7, and 11;
- three zero anchors;
- no law-specific parameter.

Architectural prior / runtime:

- every admitted location space is a finite cycle;
- equality is exact;
- nested loops have a fixed maximum of `m` by `m` successor applications;
- pop-insert state mutation and event invocation remain structural.

This is equivalent to repeated addition in a learned Cayley graph. It is not a
new universal reasoning primitive, learned field arithmetic, natural-language
semantic induction, model-owned event parsing, or learned open-ended halt. Its
stronger claim over S5 is narrow but real if confirmed: the operation law
itself is absent from training and is compiled from contextual examples.

## 5. Fresh custody split

S6 development laws are closed and cannot score S7. For each primary modulus:

- S7 training uses S6 training laws plus the identity and successor anchors;
- all other S6 development laws are excluded from scoring and training;
- the never-read S6 reserved-confirmation laws, excluding any promoted anchor,
  are split before board generation into S7 development and S7 confirmation;
- S7 confirmation rows are generated and hash-sealed but must not be opened by
  any evaluator until development qualifies every gate.

A new post-commit seed creates a hidden symbol permutation per modulus and all
program rows. The hidden permutation changes every card and destination byte,
so S6's canonical-symbol score does not leak S7 outputs.

## 6. Training arms

### Treatment

One trainable successor-logit row per observed symbol plus one zero-logit vector
per modulus. Frozen training:

- successor cross-entropy on exactly `5+7+11 = 23` anchor cells;
- zero cross-entropy on exactly three anchors;
- AdamW, fixed seed/schedule chosen before the board;
- no train-law cards beyond the two anchors, recurrent state, answers,
  development laws, or confirmation laws.

Maximum treatment parameter count is 218. The promoted stack remains far below
150M.

### Favorable ordinary-transformer control

The frozen S6 4,753,677-parameter transformer is reinitialized and receives all
atomic cells from S7 train laws under the new hidden symbols. It gets strictly
more trainable parameters and examples than treatment. Its result tests whether
the fresh anchors alone rescue generic interpolation.

### Structure-breaking generator control

An equal-size generator is trained to perfect fit on `S_m^2`, a distinct
single-cycle successor because all moduli are odd primes. It receives the same
23 cells and three zero anchors. It must fit its assigned cells but should fail
cards encoded under the true generator.

## 7. Development interventions

Evaluate exactly once on the frozen development board:

1. host affine theorem/executor ceiling;
2. learned Cayley treatment;
3. favorable ordinary transformer;
4. `S^2` structure-breaking generator;
5. deranged law cards;
6. one-witness default-unit completion;
7. state reset between events;
8. nonce identity renaming;
9. held-out-law atomic cells;
10. depth-three through depth-eight recurrent programs.

## 8. Immutable development gates

All must pass:

- treatment successor and zero anchor fit: 100%;
- structure-breaking successor and zero fit: 100%;
- ordinary-transformer atomic train fit: at least 99%;
- treatment held-out atomic destination: at least 99%;
- treatment recurrent exact state: at least 98%;
- treatment answers: at least 98%;
- every depth exact state: at least 96%;
- treatment within one point of host state and answer;
- treatment exceeds ordinary transformer by at least 40 state points;
- true treatment exceeds `S^2` control by at least 60 state points;
- deranged card drops state by at least 60 points;
- one-witness default drops state by at least 40 points;
- reset drops state by at least 20 points;
- nonce identity renaming is bit-identical;
- one development access and zero confirmation accesses;
- complete system remains below 150M.

Failure closes S7 v1. No width, update, threshold, board, or score repair is
allowed after development access. Passing authorizes one unchanged-weight read
of the already sealed confirmation board.

## 9. Pre-score implementation sequence

1. Commit this theorem, equivalence boundary, mechanics, falsifier, tests, model,
   trainer, evaluator, and assessor.
2. Run the CPU falsifier over exhaustive hidden bindings at moduli 5 and 7 and
   deterministic sampled bindings at 11 and 13.
3. Commit the admitted CPU report.
4. Draw board and training seeds only after that commit.
5. Build and hash-seal train, development, and confirmation bytes.
6. Commit the board receipt before one serial H100 run.

