# R12 Matroid Closure Deduction Target

**Status:** retained as a theorem-backed deduction target and matched control,
not yet an R12 mechanism. It gives a sharp generalization advantage once a
matroid closure class is known, but does not explain how ordinary traces reveal
that class or its latent coordinates.

## 1. Residual system

Let `M=(E,I)` be a finite matroid of rank `r`. A history contributes a set of
premises `S subset E`; a late query asks whether `q` is implied:

```
answer(S,q) = 1[q in cl_M(S)].
```

Two histories are causally equivalent exactly when they have the same closure.
The exact residual states are therefore the flats of `M`. The online action is

```
F --add(e)--> cl_M(F union {e}),
```

and a query is the membership test `q in F`.

This is genuinely deduction-shaped: premises can imply an element never stated
in the history. The smallest binary witness has rank two with `c=a+b`; premises
`{a,b}` imply `c`.

## 2. Exact-state cost does not disappear

For the projective binary matroid on

```
E = GF(2)^r \ {0},
```

flats correspond to linear subspaces with the zero vector removed. The number
of flats is

```
sum_(j=0)^r GaussianBinomial(r,j;2)
  = 2^(r^2/4 + O(r)).
```

An exact promise-free causal state therefore still requires
`r^2/4 + O(r)` bits in the worst case. Matroid structure reduces the cost of
describing and learning readouts; it does not violate the residual information
bound.

## 3. VC-dimension theorem

For one fixed rank-`r` matroid, let

```
H_M = { q -> 1[q in F] : F is a flat of M }.
```

Then

```
VCdim(H_M) = r.
```

**Lower bound.** Let `B` be a basis. For every `A subset B`, the flat `cl(A)`
intersects `B` in exactly `A`, so `B` is shattered.

**Upper bound.** If a set `S` is shattered, then for every `x in S` there is a
flat containing `S\{x}` but not `x`. Thus `x` is not in `cl(S\{x})`, and `S`
is independent. Hence `|S| <= r`.

Consequently, realizable PAC prediction of membership in an unknown flat has
sample complexity polynomial in `r`, approximately

```
O((r log(1/epsilon) + log(1/delta)) / epsilon),
```

whereas the promise-free class of arbitrary subsets of `E` has VC dimension
`|E|`. On the binary projective family, this is a separation between `r` and
`2^r-1` in the declared hypothesis class.

## 4. Why this is not yet the mechanism

The theorem assumes that the learner is already restricted to the flats of one
matroid. Ordinary premise/query examples do not by themselves reveal:

1. which matroid or closure operator is in force;
2. whether a compact representation exists;
3. the element coordinates or circuits needed for efficient updates;
4. whether the observed relation is matroidal rather than an arbitrary Horn
   closure system;
5. a robust update rule under label and state noise.

If the hypothesis class ranges over unrestricted matroids, the advantage can
vanish: every subset is a flat of the free matroid, so the union class recovers
the full subset class. If a binary matrix representation is supplied, the
solution is incremental Gaussian elimination. If closure or independence
queries are supplied, this becomes standard matroid-oracle learning. Neither
case explains latent structure discovery from ordinary language traces.

## 5. Required gate before a CPU falsifier

A surviving mechanism must prove all of the following without a matroid oracle
or handed coordinates:

1. a finite ordinary-trace distribution identifies a restricted closure class;
2. the determining set is polynomial and generated without target-specific
   board shopping;
3. online state/update/query costs are polynomial in rank and stable to a named
   noise model;
4. an unstructured and a structure-aware control receive the same observations;
5. the learned mechanism extrapolates to unseen circuits or closure chains,
   not merely unseen surface forms.

Until that theorem exists, matroid closure is a high-value R12 board family and
control target, not an authorized neural experiment.
