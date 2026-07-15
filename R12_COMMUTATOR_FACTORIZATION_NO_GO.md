# R12 Commutator Factorization No-Go

**Status:** rejected as an R12 invention. Observable event commutators can
recover a conditional central/direct product of transition groups, but do not
identify a product residual state. Complete positive cases reduce to known
group, automata, trace-monoid, or Cartesian-graph decomposition.

## 1. Observable commutator support

Let `X` be `N` finite residual states, `A` be `m` labeled events with
permutation transitions `T_a`, and let joint late-query signatures separate
states. Define

```
S(a,b) = {x in X : T_a T_b x != T_b T_a x}.
```

Separating queries make `S(a,b)` experimentally observable when states can be
prepared reproducibly. State-uniform event independence means `S(a,b)` is
empty.

## 2. Commutator-to-central-product theorem

Let `G=<T_a:a in A>`. Connect event labels that do not commute and let the
connected components be `A_1,...,A_k`, with

```
G_i = <T_a : a in A_i>.
```

Then the `G_i` commute pairwise and multiplication gives a surjection

```
mu : product_i G_i -> G.
```

Every coordinate of `ker(mu)` lies in

```
G_i intersect <G_j:j!=i> subset Z(G_i).
```

Commutators therefore identify at most a central product. The group is the
direct product of the `G_i` exactly when all cross-intersections are trivial;
centerless factors are sufficient.

This algebraic condition still does not factor the state.

## 3. Stabilizer obstruction

Assume the action is transitive and fix state `x`. Let

```
H = stabilizer_G(x),
H_i = H intersect G_i.
```

Even when `G` is a direct product, a factorwise state decomposition

```
X isomorphic to product_i G_i/H_i
```

exists iff

```
H = product_i H_i,
```

equivalently `|Gx|=product_i |G_i x|`. A diagonal or subdirect stabilizer
couples the apparent modules.

The smallest useful counterexamples are:

1. on `Z_3`, `a(x)=x+1` and `b(x)=x-1` commute but generate the same `C_3`,
   not two modules;
2. `S_3 x S_3` acts faithfully on six states `X=S_3` by
   `(g,h).x=g x h^-1`; its factors commute and form a true direct product, but
   the stabilizer is diagonal `S_3`, so `6 != 6*6` and no product state exists;
3. the regular `C_2 x C_2` action has three inequivalent choices of two
   `C_2` factors, so abelian centralizers do not select one decomposition.

## 4. Probe and information cost

With reproducible access to every state, recovering all event tables costs
`mNq` readouts for `q` separating query bits. Directly testing every pair costs

```
2 binomial(m,2) N q
```

readouts. Independent bit noise `eta<1/2` adds the repetition factor

```
O((1-2eta)^-2 log(Pq/delta)).
```

Without full-support state preparation or a positive minimum state
probability, no finite exact guarantee exists. An adversary can change one
unobserved state-event transition and destroy commutation or factorization.
Thus unrestricted state-uniform locality needs `Omega(mN)` transition probes.

If a product chart and event assignment are already known, local tables use

```
sum_i m_i n_i log n_i
```

bits instead of `mN log N`, for `N=product_i n_i`. But an opaque chart can cost
`log(N!)=Theta(N log N)` bits and exact discovery already pays the extensional
probe cost. Retained state does not shrink:

```
sum_i log n_i = log N.
```

Query savings additionally require low-arity factorized readouts.

## 5. Prior-art collapse

- finite machine decomposition is covered more generally by Krohn-Rhodes
  cascade/wreath products;
- direct-product decomposition of permutation groups is polynomial-time from
  generators ([Wilson](https://arxiv.org/abs/1005.0548));
- Cartesian transition-graph factorization and permutation-automata
  decomposition are established algorithms;
- trace monoids encode commuting event words but do not prove separate storage
  coordinates;
- subdirect and diagonal couplings are the standard Goursat obstruction.

A structure-aware group, automata, graph-factorization, or message-passing
control receives the same conditional resource gain.

## 6. Verdict

No CPU experiment is authorized. A survivor must expose an ordinary-trace
invariant that defeats central and subdirect coupling, identifies modules from
sub-extensional coverage, and yields a resource advantage over established
decomposition algorithms. Pairwise commutators do not.
