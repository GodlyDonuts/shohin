# R12 Dynamic Frontier Compression No-Go

**Status:** tight context-scaling law, rejected as a new reasoning primitive.
The surviving construction is frontier dynamic programming plus ordinary
recurrence.

## 1. Address-aware frontier theorem

At a processing cut `t`, let `B_t` be the possible active dependency frontiers.
Factor `v` has `q_v` possible states and the already closed portion has `s_t`
distinguishable summaries. If every future answer depends on the past only
through this frontier, then the residual-state count obeys

```
N_t <= s_t * sum_(B in B_t) product_(v in B) q_v,
b_t = ceil(log_2 N_t).
```

The bound is exact when future queries distinguish every support, assignment,
and closed summary.

Consequences:

- fixed public width `w`: `b_t = w log_2 q + log_2 s_t`;
- unknown named frontier among `n` factors: add `log_2 binomial(n,w)`;
- deterministic `k`-local updates cost `O(k)` once the structure is known;
- general constraint/probabilistic messages may require `Theta(q^w)` storage
  and work.

Minimizing the maximum frontier over processing orders is the established
vertex-separation/pathwidth boundary.

## 2. Nonlinear order-sensitive control

The discrete Heisenberg action

```
(x,y,z) star (a,b,c) = (x+a, y+b, z+c+x*b)
```

is nonlinear and order-sensitive. For `A=(1,0,0)` and `B=(0,1,0)`,

```
A star B = (1,1,1),
B star A = (1,1,0).
```

Length-`T` histories have polynomially many residuals, so exact state grows
only logarithmically in `T`; a free-group control has exponentially many
residuals and needs linear-in-`T` bits. This is the classical polynomial-growth
group boundary, not a new context law.

## 3. Discoverability boundary

Passive recovery of a dependency hypergraph needs bounded interaction order,
an observable faithfulness margin, positive probability for every required
contrast, and a unique minimal factorization. Under symmetric label noise, the
sample scale has the ordinary inverse-square dependence on the faithfulness
margin and `1-2 eta`.

Without those assumptions, one unseen event-query edge defeats safe deletion.
For every finite trace radius, a finite residual quotient can match a free
action on the entire observed ball while having radically different long-run
growth. Finite ordinary traces therefore cannot certify future residual
innovation.

## 4. Collapse audit

- Pathwidth/treewidth supplies the same frontier law.
- Factor graphs and dynamic Bayesian networks send the same separator messages.
- Tensor-network contraction cost is governed by the same cut width.
- OBDD width counts the same residual subfunctions at a cut.
- Segment trees accelerate associative composition without reducing summary
  information.
- Recurrent memory directly stores frontier assignments or group coordinates.
- Runtime robustness reduces to ordinary error-correcting or fault-tolerant
  computation.

## 5. Decision

Dynamic-frontier summaries beat raw-history storage and arbitrary transition
tables, but not the strongest structure-aware comparator. A CPU experiment
would reproduce known dynamic programming or algebraic accumulation. No R12
implementation is authorized. A future candidate must expose behaviorally
observable structure that uniquely identifies itself from ordinary traces,
survives runtime noise, and beats a matched structure-aware recurrent model.
