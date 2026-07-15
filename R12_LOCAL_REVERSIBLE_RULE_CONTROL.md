# R12 Local Reversible Rule Control

**Status:** retained as the strongest finite nonlinear structured-action
control found so far. It gives a real polynomial description and sample
advantage over global tables and defeats low-rank linear comparators, but the
advantage is purchased by handed local coordinates and rule sharing.

## 1. Family

Let the exact state be `x in {0,1}^n`. Each event supplies a rule label and a
labeled tuple of at most `k` wires. The label selects one of `L` unknown local
reversible maps

```
g_l : {0,1}^k -> {0,1}^k.
```

The event replaces only the selected coordinates by `g_l` of their current
values. Training traces expose pre-state, post-state, label, and affected wire
tuple; each observed output bit is independently flipped with probability
`eta < 1/2`.

This is nonlinear when the rule set contains Toffoli-type gates. NOT and
Toffoli generate universal reversible Boolean computation, so the family is
not a disguised linear automaton.

## 2. Learnability theorem under handed locality

For each rule, input pattern, and output bit, majority vote has error at most
`exp(-Theta(m(1-2 eta)^2))` after `m` occurrences. With balanced coverage of
all rule-pattern cells, a union bound gives exact recovery with probability at
least `1-delta` after

```
T = O(
  L 2^k / (1-2 eta)^2
  * log(L k 2^k / delta)
)
```

examples, up to the coverage constant. The learned presentation uses
`O(L k 2^k + log n)` bits plus the wire labels, retains exactly `n` state bits,
and applies an event in `O(k)` work. A global transition table instead has
`2^n` rows per event.

This is a real structured resource separation. It is also an ordinary local
rule learner once the coordinates and sharing map are supplied.

## 3. Separation from low-rank predictive-state controls

Let `N=2^n` and let the reachable action be transitive on the `N` states. Use
all balanced Boolean readouts as late queries. The state-by-readout incidence
matrix has full row rank `N`. More strongly, every rank-`d` approximation has
normalized mean-square error at least

```
(N-d) / (4(N-1)).
```

Thus a low-dimensional WFA/OOM/PSR/Hankel model cannot uniformly approximate
the balanced readout family unless `d` is exponential. The local nonlinear
presentation remains polynomial.

The separation is useful because it rules out the project's easiest linear
collapse. It does not rule out locality-aware neural cellular automata,
program learners, sparse circuits, or equivariant models.

## 4. Fatal hidden-coordinate assumption

Conjugate the global state by an arbitrary bijection

```
phi : {0,1}^n -> {0,1}^n.
```

The transformed dynamics `phi g phi^-1` have identical abstract transition
behavior but generally destroy every visible locality and sparsity property.
Ordinary input/output traces identify the action only up to such a conjugacy
unless observations anchor the coordinates. Therefore the sample theorem does
not establish that a learner can discover the local presentation.

The same problem survives softer versions:

- supplying wire tuples is already a program trace;
- supplying shared rule labels is meta-data about the factorization;
- runtime state noise is not corrected merely because training labels are
  denoised;
- reversible dynamics cannot erase accumulated state corruption without extra
  redundancy and irreversible correction;
- known locality-aware, kernel, equivariant, and program-induction controls can
  exploit the same gift.

## 5. Verdict

This family should be used as a hard control for any later R12 mechanism. A
candidate must learn a robust local or modular presentation from ordinary
partial observations, without receiving wire coordinates, rule labels, or the
sharing map, and must survive runtime noise. Until a theorem supplies those
missing steps, no Shohin fit or H100 job is authorized.
