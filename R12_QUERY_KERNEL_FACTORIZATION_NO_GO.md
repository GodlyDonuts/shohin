# R12 Query-Kernel Factorization No-Go

**Status:** exact diagnostic theorem, rejected as a new reasoning mechanism.
Task-native queries canonically identify predictive quotients, but the result is
Moore-machine output projection plus universal-algebra factor congruences.

## 1. Future-stable query kernels

For event maps `T_a:X->X`, query family `F`, and outputs `o_q`, define

```
kappa_F = {(x,y): o_q(T_w x)=o_q(T_w y) for every q in F and event word w}.
```

`kappa_F` is the greatest event congruence contained in the immediate query
kernel. It is the exact behavioral quotient relevant to that query family.

For query families `F_1,...,F_k`, the map

```
x -> ([x]_(kappa_1),..., [x]_(kappa_k))
```

embeds the minimal joint residual machine into the product of its query
quotients. Every projection is surjective, so the image is always a subdirect
product. It is a full direct product only when every tuple of quotient classes
is jointly realizable.

A sufficient Chinese-remainder certificate is:

1. the intersection of the kernels is equality;
2. the kernels are pairwise comaximal;
3. their generated congruence lattice is distributive;
4. the congruences permute.

If the query-generated factor congruences form a finite Boolean algebra, its
co-atoms give canonical factors up to permutation. A query depends only on
coordinate set `S` exactly when the intersection of those coordinate kernels is
contained in the query's output kernel.

## 2. Exact counterexamples

### 2.1 Smallest subdirect obstruction

Use three states, identity dynamics, and two query signatures

```
00, 01, 11.
```

The two kernels meet at equality and join universally, but signature `10` is
missing. The state space is a proper subdirect image rather than a product.

### 2.2 Pairwise tests are insufficient

On `F_2^2`, expose queries `x`, `y`, and `x xor y`. Every pair supplies valid
coordinates, but the three-bit image contains only four parity-consistent
tuples rather than eight. The generated congruence lattice is the
nondistributive `M_3`. Symmetric exposure of all three queries does not select a
canonical basis.

### 2.3 Coupled dynamics consume factors

Under CNOT, the future-stable kernel of the target-bit query collapses to
equality because a later target readout can reveal the control. Query-kernel CRT
therefore finds genuinely independent predictive modules, not interacting
reasoning modules.

### 2.4 Finite traces do not certify the kernels

Any unobserved state-event transition can be changed to violate a proposed
congruence while preserving the finite transcript. Unrestricted exact
certification requires extensional transition coverage.

## 3. Resource ledger

Let `N=|X|`, `m=|Sigma|`, `p` be total query-output bits, and `s` be separating
signature bits. Complete deterministic reconstruction uses

```
C = N p + N m s
```

readout bits. With independent flip noise `eta<1/2`, repeat each cell on the
order of

```
2/(1-2 eta)^2 * log(2C/delta).
```

Passive data additionally needs every required cell to have positive mass. If
some cell has probability zero, exact identification is impossible.

For true factor sizes `n_i`, a supplied decomposition can reduce transition
description from roughly `m N log N` to `m sum_i n_i log n_i`. It does not beat
the residual-state information lower bound `log N`.

## 4. Prior-art boundary

The CRT conditions are standard congruence decomposition. Output-projected
Moore-machine learning and product-automata learning already exploit the same
component reduction. Proper subdirect images are the same missing-combination
phenomenon as lossless-join theory; future-query coordinates also sit inside
predictive-state, observable-operator, and weighted-automata representations.

## 5. Decision

Use future-stable query congruences only as a control that diagnoses whether a
task really contains independent predictive modules. No CPU falsifier or
Shohin mechanism is authorized. Reconsider only with a theorem that learns
interacting modules from ordinary traces and beats product automata, PSRs,
tensor-factor models, and congruence decomposition under matched information.
