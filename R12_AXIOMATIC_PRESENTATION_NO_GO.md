# R12 Axiomatic Presentation Identifiability No-Go

**Status:** rejected as an R12 invention. Generator/relation curricula remain
valid controls, but finite relation loss does not identify a neural
homomorphism or guarantee unseen-composition reasoning.

## 1. Candidate

The candidate attempted to teach a small set of typed generators and axioms,
hold out long compositions, and use relation-equivalent words plus
source-deleted state interchanges to force a learned compositional action.

There is a correct extrapolation theorem, but its assumptions already contain
the hard part: every generator map must be identified on a complete domain or
a determining set. Relations certify an identified action; they do not identify
it from finite unrestricted neural behavior.

## 2. Presentation-factorization theorem

Let `Q` be a finite typed generator graph, `F(Q)` its free category, and

```
C = F(Q) / equiv_R
```

the category presented by relations `R={u_i=v_i}`. Give every object `o` a
state set `X_o` and every generator `a:o->p` a learned map

```
T_hat_a : X_o -> X_p.
```

For a path `w=a_1...a_k`, define `T_hat_w` by composition. If

```
T_hat_(u_i)(x) = T_hat_(v_i)(x)
```

for every defining relation and every state in its complete domain, then the
generator assignment factors uniquely through `C`. Thus it defines a functor

```
T_hat : C -> Set
```

and every unseen word receives the homomorphic action determined by its
generators.

The proof is the universal property of a presented category: the generator
assignment first defines a functor on the free category; equality on every
defining relation makes it constant on the generated congruence, so it factors
uniquely through the quotient.

## 3. Identification requires a determining set

To identify a target action `T_star`, each generator additionally needs:

1. a declared hypothesis class `H_a`;
2. a determining set `D_o` such that two maps in `H_a` agreeing on `D_o`
   agree on all of `X_o`;
3. exact local coverage `T_hat_a(x)=T_star_a(x)` for every `x in D_o`.

Only then does local equality imply `T_hat_a=T_star_a` globally and therefore
`T_hat_w=T_star_w` for every held-out word. Once the generator maps are
identified, relation loss is mathematically redundant for prediction. It is a
consistency certificate.

Faithfulness is not needed to predict the action, but it is needed to identify
abstract words. A nonfaithful target reveals only the quotient by its kernel.
Even with exhaustive causal interchanges, internal coordinates remain
identifiable only up to objectwise bijections, or natural isomorphism.

## 4. Finite-test no-free-lunch theorem

Consider any frozen finite suite of state-generator transitions for an
unrestricted hypothesis class. If one reachable transition `(z,a)` is never
exercised, define a patched updater that equals the target everywhere in the
suite but changes `T_hat_a(z)` and routes the first unseen word reaching `z` to
a different state. Every tested relation and interchange remains zero-loss.

Therefore a finite suite certifies arbitrary future words only if one of these
holds:

- the finite state domain is tested exhaustively; or
- the hypothesis class is restricted so the tested states form a determining
  set.

Ordinary neural networks permit finite-set patching, so finite relation tests
alone are not determining. The trivial action can satisfy many presentations;
multiple inequivalent and conjugate representations can satisfy the same
relations; and nonfaithful actions can alias distinct words.

Approximate relations weaken the claim further. If a word equality requires
many relator applications, local defects can accumulate with the presentation
area, governed in the worst case by its Dehn function. Small training relator
loss does not imply horizon-independent semantic error.

## 5. Real but limited resource advantage

For `N` states and `k` generators, explicit generator tables require about

```
k N log2(N) bits
```

and `kN` covered transitions. Exhaustively checking each defining relation on
every state costs

```
N * sum_(u=v in R) (|u|+|v|)
```

generator applications and then certifies all words for that exact action.

A noncompositional lookup system storing `M_L` distinct actions of length at
most `L` can require `Theta(M_L N log N)` bits, exponentially larger when
`M_L` grows exponentially. This is a valid separation from lookup
memorization. It is not a separation from a transformer, RNN, weighted
automaton, or any other shared-weight learner that can implement the same
generator composition.

## 6. Prior-art boundary

- Factorization through generators and relations is the standard universal
  property of presented algebraic objects.
- Auxiliary losses that impose group representation structure are already
  studied as algebraic priors for approximately equivariant networks.
- Source-state swaps are interchange intervention training.
- Transformers have already generalized permutation words from smaller to
  larger symmetric groups under a tailored curriculum.
- Autoregressive compositional task theory already gives exponential task
  coverage from near-linear component-task coverage under explicit
  compositional assumptions.

Primary sources:

- Ali, Lio, and Vicary, *Algebraic Priors for Approximately Equivariant
  Networks* (2026 revision): https://arxiv.org/abs/2506.08244
- Geiger et al., *Inducing Causal Structure for Interpretable Neural
  Networks* (ICML 2022):
  https://proceedings.mlr.press/v162/geiger22a.html
- Petschack, Garbali, and de Gier, *Learning the symmetric group: large from
  small* (2026): https://arxiv.org/abs/2502.12717
- Abedsoltan et al., *Task Generalization With AutoRegressive Compositional
  Structure* (ICML 2025): https://arxiv.org/abs/2502.08991

## 7. Decision

Reject finite axiom/relation loss as an R12 reasoning primitive. It may remain
a useful curriculum and evaluation control, but it cannot support a claim of
identified neural homomorphism or indefinite composition unless the project
first proves exhaustive state coverage or a hypothesis-specific determining
set.

The unresolved problem is not how to state algebraic relations. It is how a
small learner acquires a restricted, robust hypothesis class whose local
coverage is both feasible and sufficient, without hard-coding the target
algebra. No CPU falsifier or Shohin fit follows from the presentation theorem
alone.
