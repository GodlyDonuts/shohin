# R12 Query-Distributional Context No-Go

**Status:** tight average-case context law; reject as a novel context-scaling
mechanism.

## Model

Let `X_1,...,X_n` be independent uniform source bits. A one-pass encoder commits
to `b` source-dependent bits before an independent late query `Q` is drawn from
known probabilities `mu_{n,i}`. Source tokens, KV cache, retrieval, external
memory, and source-dependent oracle access are unavailable after commitment.
The answer is `X_Q` and loss is average bit error.

For coordinate errors `e_i`, every randomized encoder obeys

```
b >= I(X;S) >= sum_i (1 - h_2(e_i)).
```

The exact task rate-distortion problem is therefore

```
R_n(D) = min sum_i (1-h_2(d_i))
         subject to sum_i mu_i d_i <= D,  0 <= d_i <= 1/2.
```

If `mu_(1) >= ... >= mu_(n)`, a useful finite converse is

```
D_n*(b) >= max_{m>b} mu_(m) m h_2^{-1}(1-b/m).
```

Storing the `k` most likely source bits gives

```
D_n*(k+O(log n)) <= (1/2) sum_{i>k} mu_(i).
```

## Concentration characterization

Define

```
K_n(delta) = min{|A| : mu_n(A) >= 1-delta}.
```

For independent source bits, sublinear retained memory and vanishing average
error exist exactly when uniformly computable sets of size `o(n)` carry
`1-o(1)` query mass. For necessity, let `G={i:e_i<=sqrt(D)}`. Then

```
mu(G) >= 1-sqrt(D)
|G| <= b / (1-h_2(sqrt(D))).
```

Thus `b=o(n)` and `D->0` force query concentration on `o(n)` coordinates.
The error falls because discarded context is almost never queried; conditional
error on discarded positions and worst-case error remain `1/2`.

## Tight scaling control

For recency rank `r=n-i+1`, let `mu_{n,i}` be proportional to `r^-alpha` with
`alpha>1`, and keep the newest `k=n^beta` bits for `0<beta<1`. A circular buffer
achieves

```
D_n = O(n^{-beta(alpha-1)}),
```

while the converse gives `D_n*(b)=Theta(b^{1-alpha})`. Online updates and
queries cost `O(1)` word operations. The matching implementation is a sliding
window cache or one-way streaming sketch.

## Collapse audit

- the committed state is a weighted INDEX/random-access sketch;
- the objective is functional source coding with query side information;
- PSR/AIS and predictive rate-distortion cover structured predictive variants;
- reservoirs or weighted caches implement the same average-case retention;
- hierarchical summaries help only when the source/query function has a compact
  sufficient statistic, which is source structure rather than a new mechanism.

Exact all-query memory remains `n` bits. Reopen only for a learnability or
computation separation on structured sources against a resource-matched
comparator. No CPU falsifier or Shohin fit is authorized.
