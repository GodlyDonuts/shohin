# R12 Holonomy State No-Go

**Status:** rejected as an R12 invention. Loop holonomy is a valid gauge-orbit
observable and can identify a declared finite-dimensional connection up to
conjugacy. It does not identify the current causal state, and complete finite
signatures reduce to ordinary operator recurrence or PSR machinery.

## 1. Candidate object

For a typed event graph, assign every edge `e:v->w` an invertible residual
transport

```
T_e : F_v -> F_w.
```

For history `h=e_t...e_1`, let `T_h=T_(e_t)...T_(e_1)`. A loop `l:v->v` has
holonomy `H_l=T_l`, and two paths `p,q:v->w` have defect

```
C_(p,q) = T_q^-1 T_p.
```

Under hidden-coordinate changes `g_v`,

```
T'_e = g_w T_e g_v^-1,
H'_l = g_v H_l g_v^-1.
```

Traces, spectra, characters, and other conjugacy-class functions are therefore
gauge invariant. They identify properties of the orbit; by definition they
cannot select a hidden gauge.

## 2. Strong finite survivor

For a connected finite graph and compact matrix group `K subset U(d)`, choose
a spanning tree and gauge every tree edge to identity. The remaining

```
m = |E|-|V|+1
```

chord transports are fundamental-loop holonomies. They determine the
connection up to one simultaneous global conjugation, and every unseen history
is a word in those `m` operators. Finite joint trace-word invariants can
separate simultaneous unitary-conjugacy orbits for fixed `d,m`.

The two-generator `SU(2)` case is explicit. Write

```
A = x I + i a.sigma,
B = y I + i b.sigma.
```

The three signatures

```
x = tr(A)/2,
y = tr(B)/2,
z = tr(A B^-1)/2
```

recover the norms and inner product `a.b=z-xy`; their Gram matrix determines
`(A,B)` up to simultaneous conjugation. Under nondegeneracy margin `kappa`, a
signature error `epsilon` yields generator error on the order of
`epsilon/kappa^3` and length-`L` word error on the order of
`L epsilon/kappa^3`.

Static storage is `O(md^2 b)` bits, dynamic operator state is `O(d^2 b)`, and
each event costs `O(d^3)`. This is exponentially shorter than a table of all
words but exactly matches matrix recurrence and observable-operator controls.

## 3. State obstruction

Holonomy describes the connection, not the current point in its fiber. Two
future-distinguishable causal states under one connection have identical
state-independent loop signatures. Adding state-dependent probe responses
creates continuation/query prediction coordinates, which is a PSR/OOM.

The smallest example uses one observable vertex, hidden fiber `{1,2,3}`, gauge
group `S_3`, and events `a,b`:

```
M_0: A=B=(12)
M_1: A=(12), B=(13).
```

Both individual permutation traces equal one. The composite has trace three in
`M_0` and zero in `M_1`, so joint loops recover relative operator orientation.
Once recovered, unseen behavior is ordinary permutation multiplication. No
loop trace reveals which hidden point is currently occupied.

Local loops are also incomplete: a flat `U(1)` connection on a noncontractible
cycle can have zero local defect and nontrivial global holonomy. Unrestricted
nonlinear actions admit compactly supported off-probe perturbations that
preserve every finite loop test and alter a later composition.

## 4. Exact collapse test

For any finite proposal:

1. enumerate transport tuples modulo vertex gauge;
2. map each orbit to the proposed finite signature;
3. reject if one signature fiber contains two orbits differing on a target
   word/query;
4. repeat on joint `(transport,current_state)` orbits;
5. if every fiber is singleton, reconstruct a canonical tuple and run the
   ordinary recurrence `U_(t+1)=A_(e_t) U_t`;
6. reject novelty if this matched recurrence is exact;
7. under noise, use minimum signature separation `Delta_n`; vanishing
   `Delta_n` forces at least `Omega(Delta_n^-2)` samples.

Incomplete signatures fail identification; complete signatures reconstruct an
established operator model.

## 5. Prior-art boundary and verdict

Connection reconstruction from loop holonomies, periodic-orbit cocycle
identification, gauge-equivariant computation, synchronization, cycle
consistency, and PSR/OOM operator learning already occupy every surviving
case. Loop-based state correction additionally requires redundant state-bearing
measurements and becomes synchronization, error correction, or denoising.

No CPU falsifier is authorized. Reconsideration requires naturally available
loop observations that identify the joint action-state orbit with a uniform
margin, correct runtime noise, and beat equally informed operator recurrence,
PSR, synchronization, and ECC controls.
