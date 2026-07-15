# R12 Reasoning Invention Charter

**Status:** theory phase only; no implementation, data build, fit, score, or GPU
job is authorized.

**Effective:** 2026-07-15. This charter supersedes architecture-first reasoning
experiments. R9, R10, and R11 remain evidence and matched controls, not active
mechanism templates.

## 1. Why this charter exists

Shohin has already falsified several easy stories:

- adding visible or latent traces can teach formatting without exact transport;
- fixed recurrence can be an unrolled feed-forward computation;
- dynamic recurrence can still learn the same local classifier as static controls;
- external algebra can solve a board without establishing neural reasoning;
- source-conditioned slots, matrices, or adapters can reduce to retrieval, fast
  weights, or a hypernetwork;
- more pretraining has not yet produced reliable direct reasoning behavior.

Workers must therefore stop beginning with familiar modules and searching for a
claim afterward. R12 begins with a mathematical capability and derives the
necessary state and operator before considering a realization.

## 2. The finite-circuit boundary

For fixed context length, finite precision, and bounded runtime, every
deterministic classical mechanism can be unrolled into a finite acyclic circuit.
Loops become repeated subgraphs, memory reads become multiplexers, generated
weights can be substituted, and fixed external computation can be inlined.

Consequently, "not equivalent to any static classifier" is not an admissible
requirement: it rejects every bounded implementation. Novelty must instead be
stated relative to an explicit resource-scaled comparator family. No R12 report
may claim separation from all static computation.

## 3. Operational definition

At scale `n`, let `Sigma_n`, `Q_n`, and `A_n` be event, late-query, and answer
sets. A history is `h in Sigma_n*`; `c` is an admissible future continuation;
and the answer relation is

```
R_n(hc, q) subset A_n,  R_n(hc, q) != empty.
```

Histories are causally equivalent exactly when no admissible future can
distinguish them:

```
h ==_R h'  iff  for every c and q, R_n(hc, q) = R_n(h'c, q).
```

The causal state is the equivalence class `S_n(h) = [h]`. A model family is an
R12 systematic reasoner only if one finite rule specifies every scale, its error
tends downward as scale grows, the number of required causal states is unbounded,
and it has a stated asymptotic resource advantage over a named comparator class.

This definition concerns uniform late-query causal composition. It does not by
itself establish discovery, semantic understanding, proof insight, or general
intelligence.

## 4. The object that must be realized

Define the counterfactual residual of a history:

```
rho_h(c, q) = R_n(hc, q).
```

An event acts through a residual derivative:

```
(partial_e rho)(c, q) = rho(ec, q).
```

Any candidate realization must satisfy all six axioms:

1. **Closure:** `partial_e rho` is another valid residual state.
2. **Composition:** `partial_empty = I` and
   `partial_(uv) = partial_v compose partial_u`.
3. **Observation:** `O_q(rho) = rho(empty, q)`.
4. **Extensionality:** two states are equal exactly when all future
   continuation-query answers agree.
5. **Separation:** distinct states admit a distinguishing continuation-query
   pair.
6. **Uniformity:** one finite rule specifies updates and observations at every
   tested scale; there is no scale-specific advice table.

Ambiguity must preserve every future-distinguishable class. Averaging distinct
operators or answers is not a valid uncertainty representation when a future
query can separate them.

## 5. Necessary resource obligations

If the causal quotient has `N_n` states, an exact query-oblivious state needs at
least

```
B >= log2(N_n)
```

history-dependent bits. Every candidate must count dynamic context, caches,
stored source, intermediate tensors, generated parameters, and external state.
Model parameters are only the fixed description length.

The update law must realize the action induced on the causal quotient:

```
U_e([h]) = [he]
U_(uv) = U_v compose U_u.
```

An order-sensitive witness therefore requires a noncommutative update. Any
commutative pool, expected operator that aliases distinct futures, or fixed
template inventory fails before training.

## 6. First theorem-backed witness

For `m` objects, let events be adjacent transpositions `tau_i = (i, i+1)`.
For a word `w = e_1 ... e_L`, define

```
pi_w = e_L compose ... compose e_1.
```

The query `j` is revealed only after the word and asks for `pi_w(j)`. Once all
permutations are reachable, the causal quotient has exactly `m!` states, so an
exact query-blind state needs at least `log2(m!)` bits. On the `m=2`
restriction, the answer is parity, giving a clean separation from
polynomial-size constant-depth AND/OR/NOT circuits. This is a separation from
`AC0`, not from arbitrary transformers or threshold circuits; stronger relevant
separations are open complexity questions.

The finite falsifier uses `m in {5, 8, 12}` and increasing unseen lengths. It
must include:

- equivalent words generated by involution, distant commutation, and braid
  relations;
- non-equivalent order twins with a known separating late query;
- identical continuations appended after equivalent and non-equivalent prefixes;
- every late query, not a selected easy query;
- a balanced `m=2` parity restriction while length doubles;
- state-capacity and compute ledgers checked against the information bound.

One exact counterexample kills an exact residual-composition claim. Passing a
finite board does not prove the asymptotic claim because a finite lookup table
can pass any finite board.

## 7. Mandatory invention gates

Every future worker must produce these artifacts in order:

1. **Capability theorem:** relation, comparator class, resource measure, and
   proof or explicitly labeled conjecture.
2. **Axiomatic primitive:** state and operators defined without neural-module
   vocabulary.
3. **Equivalence dossier:** algebraic checks against SFT, fixed/tied recurrence,
   retrieval, fast weights, hypernetworks, external execution, and finite
   unrolling.
4. **Exact collapse test:** a symbolic or exhaustive CPU test that tries to
   reduce the proposal to those controls. A successful reduction rejects it.
5. **Prior-art boundary:** search after the object is defined, then state the
   exact delta. A combination of known ingredients is not a new primitive.
6. **Finite falsifier:** frozen scale extrapolation, causal interchange,
   equivalent-state invariance, non-equivalent-state separation, and full
   resource accounting.
7. **Matched controls:** every known realization receives matched or favorable
   parameters, state, and compute.
8. **Score-blind confirmation:** one immutable implementation and one frozen
   confirmation generation. No board, seed, threshold, or artifact shopping.

No neural implementation is authorized through gate 4. No Shohin fit is
authorized through gate 6. No H100 experiment is authorized through gate 7.

## 8. Rejected starting points

The following are controls, not R12 inventions:

- more CoT/SFT/RL, teacher traces, self-review, or verifier reranking;
- more hidden slots, recurrent loops, adaptive depth, equilibrium iterations,
  or test-time search;
- KV memory, retrieval, replay, latent scratchpads, or context compression;
- source-generated matrices, adapters, gates, or weights;
- a hard-coded symbolic solver, parser, executor, algebra, or tree that computes
  the answer outside the learned mechanism;
- persistent product trees or Schur-complement boundary actions presented as new
  primitives. They may be strong controls but are known mathematical machinery
  plus source-conditioned state.

A future proposal may use a known component only after the primitive has been
derived independently and only if the component is not the claimed invention.

## 9. Current decision

The active research object is **uniform late-query causal composition through a
counterfactual residual algebra**. This is a requirement, not yet a mechanism.
No candidate implementation has survived the invention gates. That is the
honest frontier: R12 has a theory target and a falsifier, but not yet a new SoTA
method or real reasoning result.

The next authorized action is mathematical: derive a realizable operator that
satisfies the residual axioms and has a nontrivial resource separation against a
strong, explicit comparator. Architecture design remains blocked until that
derivation survives an independent equivalence attack.
