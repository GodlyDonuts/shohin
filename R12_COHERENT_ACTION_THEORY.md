# R12 Coherent Action Extension Audit

**Status:** theorem-backed control; rejected as a new reasoning primitive.

**Implementation authority:** none. This document authorizes no data build,
model change, fit, score, CPU board, or GPU job.

## 1. Decision

An entire nonexpansive event-monoid action can be extended coherently into a
hyperconvex function space. All monoid relations then hold everywhere, and a
merged fiber incurs no additional error as a word grows.

That positive result does not provide compressed reasoning. The construction
stores an event-closed observable profile and updates it by coordinate
substitution. In the unrestricted finite case its dimension is the number of
exact states times the size of the transition monoid. With restricted
observables it is a predictive-state or Koopman profile. Quantizing the profile
is a constructive rate-distortion bound, not an escape from late-query
information lower bounds.

The useful negative result is smaller: extending every generator separately,
even into a hyperconvex ambient space, does not imply that the extensions can
preserve the monoid relations. Relation coherence must be imposed on the whole
action.

## 2. Coherent extension theorem

Let `(X,d)` be a bounded metric space of diameter `D`. Let a monoid `M` act on
the right by nonexpansive maps `T_u`, with

```
T_(uv) = T_v compose T_u.
```

Let `A = {T_u : u in M}` be the transition monoid and define

```
Y = [0,D]^(A x X)
Phi(x)[A,z] = d(Ax,z)
```

with the sup metric. For each event `e`, define

```
(Tilde_e y)[A,z] = y[A compose T_e,z].
```

Then:

1. `Phi` is an isometric embedding.
2. Every `Tilde_e` is nonexpansive and
   `Tilde_e Phi(x) = Phi(T_e x)`.
3. The whole action is coherent:
   `Tilde_(uv) = Tilde_v compose Tilde_u`.
4. `Y` is hyperconvex.
5. For a fiber `F subset X` of diameter `Delta`, the coordinatewise midrange

   ```
   c_F[A,z] = (sup_(x in F) d(Ax,z) + inf_(x in F) d(Ax,z)) / 2
   ```

   has optimal covering radius exactly `Delta/2` around `Phi(F)`.
6. If `h:X -> R` is `L`-Lipschitz and `h_tilde` is a same-constant extension
   from `Phi(X)` to `Y`, then for every word `w` and every `x in F`,

   ```
   |h_tilde(Tilde_w c_F) - h(T_w x)| <= L Delta / 2.
   ```

   The bound is independent of word length.

### Proof

For any `x,x'`, nonexpansiveness gives

```
|d(Ax,z) - d(Ax',z)| <= d(Ax,Ax') <= d(x,x').
```

The identity coordinate and `z=x` attain equality, so `Phi` is isometric.
Coordinate substitution is nonexpansive. Closure of the transition monoid gives
`A compose T_e in A`, and direct substitution proves equivariance and the
monoid law.

A product of closed intervals with the sup metric is hyperconvex. The midrange
has radius half the largest coordinate range. Isometry makes that largest range
exactly `Delta`; no center can have radius below half the diameter. Finally,
nonexpansiveness and equivariance give

```
d(Tilde_w c_F, Phi(T_w x)) <= d(c_F, Phi(x)) <= Delta/2,
```

and the readout bound follows.

## 3. Observable-profile form and exact cost

The same construction needs only an event-closed observable family `O`. Suppose
`g compose T_e in O` for every `g in O` and event `e`, and define

```
d_O(x,x') = sup_(g in O) |g(x)-g(x')|.
```

Then `Phi_O(x)[g]=g(x)` and

```
(Tilde_e y)[g] = y[g compose T_e]
```

give the same coherent theorem in `Y_O=[0,D]^O`. If `O` is finite with `s`
members, the ambient dimension is `s`. For the unrestricted distance-profile
construction on finite `X`, if `|X|=n` and the transition monoid has `q`
distinct maps, the displayed construction has `nq` coordinates.

This is the central cost. A small `s` exists only when the future-observable
profile already has a small invariant span or restricted predictive dimension.
That is the compression assumption, not a consequence of the theorem.

## 4. Quantization and relation error

Use a coordinate grid of spacing at most `delta`. Nearest-grid quantization and
coordinate substitution commute. With `K=ceil(D/delta)`, an `s`-coordinate
state uses at most

```
b = s ceil(log2(K+1))
```

bits. Pairwise metric distortion is at most `delta`; a quantized merge center
obeys

```
readout_error <= L (Delta/2 + delta/2)
```

for every word, while all monoid relations remain exact on the grid.

By contrast, if arbitrary nonexpansive generator extensions have uniform
one-step equivariance error at most `eta`, the generic length-`n` bound is
`n eta`, and it is sharp without extra contraction. If a defining relation has
global defect at most `kappa`, replacing `k` relators inside nonexpansive word
contexts changes a state by at most `k kappa`. Presentation area therefore
controls how local relation defects amplify.

## 5. Smallest prescribed-ambient obstruction

Let

```
X = {1,2} subset Y = {0,1,2},  d(i,j)=|i-j|,
M = <a | a^2 = 1>,
T_a(1)=2, T_a(2)=1.
```

The map `T_a` has nonexpansive extensions to `Y`; for example

```
S(0)=2, S(1)=2, S(2)=1.
```

But no nonexpansive extension can satisfy `S^2=id_Y`. An involutive
nonexpansive self-map has a nonexpansive inverse and is therefore an isometry.
No isometry of this three-point line can swap `1` and `2`. In fact,

```
inf_S sup_(y in Y) d(S^2 y,y) = 1.
```

The same lower bound holds in the hyperconvex prescribed ambient interval
`[0,2]`. This obstruction is cardinality-minimal: a nontrivial involution needs
two exact points, and a proper ambient extension needs a third.

This does not contradict the positive theorem. It proves that an arbitrary
chosen ambient space can block coherent extension; the function-space theorem
constructs a different equivariant ambient space large enough to carry the
whole action.

## 6. Collapse and prior-art boundary

The positive construction is coinduction into a function space. Its update is
the pullback action on observables. With finite predictive readouts it is an
observable-profile, predictive-state, or Koopman representation. A finite
invariant linear span is an ordinary equivariant linear representation.

The general neighborhood is established mathematics:

- [Linearizability of Non-expansive Semigroup Actions on Metric
  Spaces](https://arxiv.org/abs/math/0612553) proves that a nonexpansive
  semigroup action is linearizable exactly when its orbits are bounded.
- [On the Dynamics of Lipschitz
  Operators](https://arxiv.org/abs/2011.10800) uses the universal
  Lipschitz-free-space linearization of a Lipschitz map.
- [Injective Hulls of Certain Discrete Metric Spaces and
  Groups](https://arxiv.org/abs/1107.5971) reviews Isbell injective hulls and
  shows that even finite metric spaces can require nontrivial polyhedral hulls.

No novelty claim is allowed for coherent function-space extension,
hyperconvexity, coordinate pullback, or the quantized profile. The exact
coordinate formula and the three-point obstruction are retained as project
controls, not as a proposed mechanism.

## 7. Consequence for R12

Coherent extension is solved but does not survive the invention gate. It trades
horizon error for explicit storage of an event-closed future-observable profile.
For arbitrary late queries that profile inherits the same information lower
bounds as the exact residual state.

Do not implement this construction. A future R12 survivor must instead prove a
uniform advantage in **learnability, dynamic sparsity, amortized verification,
or another named resource** while preserving a broad late-query family. It
must not count a restricted observable family as free or call coordinate
substitution internal reasoning.
