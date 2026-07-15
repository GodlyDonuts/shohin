# R12 Matroid Closure Deduction Target

**Status:** retained as a theorem-backed deduction target and fixed-field
linear-algebra control; rejected as a general R12 mechanism. Fixed-field linear
matroids are compact but reduce to representation discovery plus Gaussian
elimination. General sparse-paving matroids require exponential passive
determining sets.

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

## 5. Passive determining-set theorem

For premise/query domain

```
X_n = {(A,e): A subset E\{e}},
h_M(A,e) = 1[e in cl_M(A)],
```

a passive dataset `D` exactly identifies `M` inside class `H` iff

```
D intersects Delta(M,N) for every N != M,
Delta(M,N) = {x : h_M(x) != h_N(x)}.
```

Under sample distribution `P`, let

```
gamma_M = min_(N != M) P(Delta(M,N)).
```

If `gamma_M=0`, exact identification is impossible at every sample size. For a
finite class, a union bound gives the sufficient scale

```
m >= (log(|H|-1) + log(1/delta)) / gamma_M.
```

Passive success therefore requires support on a determining set; it does not
emerge from the exchange axiom alone.

## 6. Fixed-field positive and sparse-paving no-go

A rank-`r` matroid representable over `GF(q)` is induced by an `r` by `n`
matrix, so the number of hypotheses is at most `q^(rn)` and

```
VCdim <= rn log_2 q.
```

This gives polynomial information-theoretic sample complexity relative to an
arbitrary closure table. It does not by itself give a polynomial passive
algorithm. For binary matroids, a target-aware teacher can expose a basis and
all fundamental circuits in `O(nr)` labels, after which the normalized matrix
is fixed. That is curated teaching, and known query-learning access is stronger
than ordinary passive traces.

The opposite regime is explicit. Let `U_(r,n)` be the uniform matroid and, for
each `r`-subset `C`, let `M_C` have `C` as its only circuit-hyperplane. The
closure labels of `U_(r,n)` and `M_C` differ only on a witness set `D_C`, and
the `D_C` are pairwise disjoint. Any teaching set distinguishing the uniform
matroid from every such alternative must hit all of them:

```
TD(U_(r,n)) = binomial(n,r).
```

This is exponential near `r=n/2`. More generally, sparse-paving matroids
correspond to stable sets of the Johnson graph
([Pendavingh and van der Pol](https://arxiv.org/abs/1411.0935)), yielding the
lower bound

```
VCdim >= binomial(n,r) / (r(n-r)+1).
```

At middle rank this is `Omega(2^n/n^(5/2))`. Independent label noise adds the
usual `(1-2 eta)^-2` repetition factor and does not repair rare-witness
coverage. False inclusions in monotone closure state also persist unless the
premises are retained and recomputed or protected by error correction.

Horn-closure learning does not escape the access issue: polynomial algorithms
use closure and equivalence queries
([Arias et al.](https://arxiv.org/abs/1503.09025)), not ordinary passive traces.

## 7. Required gate before a CPU falsifier

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

Until that theorem exists, binary/projective closure is a favorable
linear-algebra control, not an authorized neural experiment. Reconsideration
requires a nonrepresentable subclass with polynomial passive determining sets,
polynomial-time representation discovery, bounded static description, and
runtime noise correction without supplied coordinates or oracle access.
