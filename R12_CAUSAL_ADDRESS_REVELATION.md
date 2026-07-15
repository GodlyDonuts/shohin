# R12 Causal-Address Revelation

**Status:** **REJECTED AS AN R12 REASONING MECHANISM AFTER INDEPENDENT AUDIT.**
The private-bank simultaneous-message theorem is valid, but it is not a lower
bound for a normal centralized neural model. Arbitrary cross-bank preprocessing
can store a composite table or a segment tree, and a depth-matched Transformer
already has adaptive routing rounds. No CPU fit or neural implementation is
authorized.

This document preserves the useful theorem as a diagnostic for physically
isolated memory banks and records the exact collapse that rejected it.

## 1. Residual family

Let

```
f_1,...,f_H : [m] -> [m].
```

After the functions are committed and the source is unavailable, a late query
is an interval and start address

```
q=(a,b,x),
F(q)=f_b circle ... circle f_(a+1)(x).
```

Each function is stored in a private internal bank. A controller can obtain
bank information only through internal messages. The comparison is between
messages chosen simultaneously and messages chosen adaptively after earlier
bank replies. This private-bank restriction is essential and is not a faithful
model of ordinary centralized neural memory.

## 2. Retained-state lower bound

If the post-commit query family contains every length-one interval, those
queries recover every table entry. There are `m^(mH)` different residual rows,
so exact retained memory is both necessary and sufficient at

```
M* = H m log_2 m bits.
```

Iteration does not compress the source or evade the closed late-query theorem.

If the only possible query is the full composition, this statement is false:
the residual state is only the composite table and needs `m log_2 m` bits. The
original proposal mixed these two query families; the audit corrected the
quantifier before any experiment.

## 3. Exact simultaneous-round theorem

For the full chain starting at known `x`, bank one need reveal only `f_1(x)`,
costing `log_2 m` bits. Every later bank must send before its realized input
address is known.

Fix one later bank `j`. If two functions `g,g'` share a message but differ at
address `y`, choose upstream functions that route the chain to `y` and choose
all downstream functions as identities. The simultaneous transcript is then
identical while the required answer differs. The bank message must therefore
identify its entire function table.

For integer bit accounting, the exact deterministic worst-case payload is

```
C_sim = ceil(log_2 m) + (H-1) ceil(m log_2 m) bits.
```

The argument is additive because fixing every other bank leaves an independent
random-access requirement for the selected bank.

For worst-case output error at most `epsilon`, the random-access-code reduction
and Fano's inequality give the per-symbol term

```
log m - h_2(epsilon) - epsilon log(m-1),
```

and the corresponding lower bound is that term multiplied by
`1+(H-1)m`, subject to the frozen randomized-protocol model.

This does not automatically lower-bound average accuracy under uniformly random
function chains. The worst-case proof uses adversarial upstream routers and an
identity suffix; random maps can collide and erase distinctions.

## 4. Iterative protocol

Adaptive internal computation reveals one address at a time:

```
y_0=x,
y_j=f_j(y_(j-1)).
```

It uses `H` rounds, workspace `log m + log H`, and exactly

```
C_iter = H ceil(log_2 m)
```

bank-to-controller payload bits. Address traffic adds a comparable term if it
is charged. Ignoring address traffic consistently, the activation-bandwidth
ratio is

```
C_sim / C_iter = (1+(H-1)m)/H = Theta(m).
```

No outside information enters. The advantage comes from revealing the next
address before selecting the next memory cell.

## 5. Smallest strict witness

For `H=m=2`:

- exact retained state is four bits for two Boolean maps;
- a two-round controller reads `f_1(x)` and then the addressed bit of `f_2`,
  transmitting two bits;
- a simultaneous exact controller needs `f_1(x)` plus both bits of `f_2`,
  transmitting three bits;
- if restricted to two simultaneous bits, the second bank is a `2->1`
  one-bit random-access code and average success is at most `3/4`.

This is the smallest finite strict separation.

## 6. Necessary collapse control

For translations

```
f_j(y)=y+a_j mod m,
```

every bank can simultaneously reveal `a_j`; the decoder sums them with
`H log m` bits. Iteration has no activation advantage. The carrying condition
is therefore not generic compositionality but:

> Later operations have large counterfactual address width, the realized
> address is revealed only by earlier computation, and no short global
> composition summary is available.

If banks can cross-compile source-dependent summaries before the query, if a
decoder gets unrestricted global memory access, or if the function family has
a short composition law, the claimed separation can disappear.

## 7. What certificates do not add

A stored certificate counts as retained state. An internally generated
certificate is downstream of retained state and adds no source information.
Checking every link `y_j=f_j(y_(j-1))` requires the same adaptive addresses as
performing the chain. An external prover changes the closed protocol.

Certificates can improve reliability only with a separately trusted verifier
or independent error model. Shared parameters and correlated errors supply no
general gain.

## 8. Prior-art boundary

This is a modular specialization of pointer-chasing round hierarchies, not a
new lower-bound family. Nisan and Wigderson describe `k`-round pointer chasing
in `O(k log m)` communication and a large loss with one fewer effective round
([primary paper](https://www.math.ias.edu/~avi/PUBLICATIONS/MYPAPERS/NW91/proc.pdf)).

The project-level contribution under test is the mapping:

```
internal thought round <-> one causally addressed state activation.
```

Any experiment must compare against equally informed attention, memory-network,
RNN, and operator controls and must not claim a new communication theorem.

## 9. Original finite falsifier, now rejected

Use fresh random function banks with

```
H in {2,4,8},
m in {8,16,32}.
```

The recurrent arm performs one addressed bank read per internal step. The
simultaneous arm receives the same retained tables and activation budget but
must choose all addresses before any reply. Required controls:

1. translation functions, where the gap must disappear;
2. centralized unrestricted-memory control, labeled as outside the theorem;
3. equal parameter, optimizer, data, total activated entries, and report
   budgets;
4. explicit round, FLOP, wall-time, and memory ledgers;
5. fresh held-out tables, unseen starts, unseen `H/m` cells, and depth scaling;
6. exact symbolic protocol baselines and random-access upper bounds;
7. no shared table summaries that leak composition before the query.

A pass would only reproduce the theorem forced by the private-bank access
restriction. It would not establish an advantage against a fair centralized
model, so this experiment is not authorized.

## 10. Independent collapse audit

The fair comparator is an arbitrary static data structure

```
P(F) in ({0,1}^w)^S,
```

with cross-bank preprocessing. Its query algorithm may probe `t` cells in `r`
adaptive rounds. A complete resource report must include storage `S w`,
preprocessing work, rounds, probes, query work, and error.

CAR's private-bank formula is not a lower bound in this model:

1. For full-chain queries, preprocess the composite table
   `g=f_H circle ... circle f_1` and answer with one lookup.
2. If length-one queries must remain available, retain the raw tables and add
   `g`, costing only `(H+1)m log_2 m`, a relative `1+1/H` storage overhead.
3. For arbitrary intervals, a segment tree stores hierarchical compositions in
   fewer than `2Hm` table entries and answers with logarithmically many
   sequential lookups.
4. A depth-matched tied Transformer, recurrent-attention model, or neural RAM
   already has the adaptive routing rounds granted only to the CAR treatment.

Therefore the observed separation would be a preprocessing-versus-lazy-
evaluation or physically-isolated-bank tradeoff. It does not identify a new
source of reasoning, context compression, or extrapolation. Establishing a
centralized space-round-probe lower bound would require a different theorem in
the cell-probe model.

## 11. Final decision

- Preserve the exact `H=m=2` witness as a symbolic communication control.
- Do not train a CAR neural arm or build the proposed CPU board.
- Do not call CAR latent reasoning, context compaction, or a novel complexity
  class.
- Reconsider adaptive routing only if a future theorem beats composite-table,
  segment-tree, depth-matched Transformer, recurrent-memory, and arbitrary-
  preprocessing controls under one explicit Pareto resource ledger.
