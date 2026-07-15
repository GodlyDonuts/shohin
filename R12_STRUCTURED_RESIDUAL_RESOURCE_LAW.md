# R12 Structured Residual Resource Law

**Status:** theorem-backed boundary and control result. It narrows the R12
target but does not authorize a neural implementation or GPU experiment.

## 1. Question

Can a structured causal state use fewer resources than the residual quotient
while still answering every admissible future query? The answer depends on the
resource:

- **No** for distinguishable states or retained information.
- **Yes** for description length, update cost, and global certification versus
  an explicit extensional transition table.
- **Not yet established** for learning a short nonlinear action from ordinary
  noisy traces.

This distinction prevents a compact coordinate system from being mistaken for
information compression beyond the task's causal quotient.

## 2. Residual-factor no-go

Let

```
rho_h(u,q) = F(hu,q)
```

be the residual behavior after history `h`. Suppose a realization has state
`E(h)`, event updates `U_a`, and readouts `D_q` satisfying

```
D_q(U_u(E(h))) = F(hu,q)
```

for every reachable history, continuation, and admissible query.

If `E(h)=E(k)`, every later update and readout is identical, so
`rho_h=rho_k`. Therefore the reachable internal system has a well-defined
equivariant surjection onto the residual system:

```
pi(E(h)) = rho_h.
```

Consequently, for residual set `R`,

```
|S_reach| >= |R|
b_state >= ceil(log2 |R|).
```

An internal realization may refine one residual state into several coordinate
states, but it cannot merge two future-distinguishable residuals. Under a
robust approximate decoder, cardinality is replaced by the relevant packing
number of residual behaviors. A representation can beat an explicit table; it
cannot beat the best succinct realization of its own residual transducer on
the information axis.

## 3. Exact description and certification separation

Define the behavior Hankel matrix over a field by

```
H[h,(u,q)] = F(hu,q).
```

If `rank(H)=r`, row coordinates provide an exact `r`-dimensional linear
realization and right residuals induce linear update operators. Conversely,
every `r`-dimensional linear realization implies `rank(H)<=r`. This is the
classical Hankel/minimal-linear-realization boundary, not a new primitive.

### 3.1 Concrete family

Let

```
G_r = (Z/2Z)^r.
```

Event `a_i` flips bit `i`; query `q_j` asks for bit `j`. Equivalently, use a
sign state `z in {-1,+1}^r`, flip one coordinate per event, and return `z_j`.

The exact resource ledger is:

- residual states: `2^r`;
- retained information: exactly `r` bits;
- Hankel rank: exactly `r`, because all columns are signed coordinate
  functions and the `r` coordinate functions are independent;
- event update: `O(1)`;
- query: `O(1)`;
- perturbation growth in the exact sign representation: none;
- explicit extensional transition table: `r * 2^r` entries.

The short presentation

```
<a_1,...,a_r | a_i^2=e, a_i a_j=a_j a_i>
```

plus a faithful full-rank character representation certifies the complete
action with polynomially many algebraic checks. By contrast, a verifier given
only an arbitrary black-box transition table must inspect every entry: one
unread entry can be corrupted without affecting its transcript.

This is a genuine exponential description and global-certification separation
relative to an explicit black-box table. It is not a state-memory separation.
It is already the territory of finite-dimensional linear realizations,
weighted automata, observable operator models, and predictive-state
representations.

## 4. Context law for structured source languages

Let `X` be a subshift and let `p_X(n)` count its admissible length-`n` blocks.
If every coordinate of the deleted block may be queried later, two distinct
blocks are residual-distinguishable at a coordinate where they differ. The
exact number of residual states is therefore `p_X(n)`, and optimal retained
information is

```
b_X(n) = ceil(log2 p_X(n)).
```

With a shared decoder for the source language, an enumerative index attains
this bound. Hence the asymptotic context rate is

```
lim b_X(n)/n = h_top(X)/log(2).
```

For a Sturmian system, `p_X(n)=n+1`, so an admissible length-`n` block can be
indexed in `Theta(log n)` bits instead of `n` raw bits. This does not violate
the late-query lower bound: the source family itself contains only `n+1`
possible blocks. A circle-phase representation still requires increasing
precision to distinguish all of them.

Low topological entropy alone does not imply a cheap online algorithm. A
language may have few blocks but make identification, ranking, update, or
decode computationally hard. Sparse positions can also carry arbitrary
information while preserving zero asymptotic entropy.

## 5. Exact conditions for useful context scaling

A usable structured-context theorem requires all four conditions:

1. **Short shared presentation.** A uniform description of the residual action
   is shared across tasks or identifiable before source deletion.
2. **Robust faithful representation.** The state uses bounded precision, has
   separating readouts, survives noise, and admits sparse or otherwise cheap
   updates.
3. **Sublinear residual innovation.** Conditional on the shared presentation
   `theta`, the task family satisfies

   ```
   H(S_n | theta) >= H(R_n | theta) = o(n).
   ```

4. **Efficient discovery and execution.** The presentation can be learned,
   states can be ranked/encoded, and updates and queries can be executed online
   within the claimed resources.

Without condition 1, the presentation is hidden source-dependent memory.
Without condition 2, real-valued coordinates hide unbounded precision.
Without condition 3, no sublinear context representation exists. Without
condition 4, entropy is an information statement rather than an implementable
context mechanism.

## 6. Prior-art boundary

- Finite Hankel rank equals minimal linear realization dimension; spectral
  learning estimates those operators from data.
- Multiplicity automata, observable operator models, and predictive-state
  representations have a unified sequential-systems formulation.
- Weighted automata over fields can be exponentially more compact than finite
  automata and remain actively learnable in an oracle model.
- Sturmian factor complexity is exactly `n+1`.

Primary sources:

- Denis, Gybels, and Habrard, *Dimension-free Concentration Bounds on Hankel
  Matrices for Spectral Learning* (JMLR 2016):
  https://www.jmlr.org/papers/volume17/14-501/14-501.pdf
- Thon and Jaeger, *Links Between Multiplicity Automata, Observable Operator
  Models and Predictive State Representations* (JMLR 2015):
  https://jmlr.org/papers/v16/thon15a.html
- Kaznatcheev and Panangaden, *Weighted automata are compact and actively
  learnable* (2021): https://arxiv.org/abs/2011.10498
- De Luca and Fici, *On the Lie complexity of Sturmian words* (2022):
  https://arxiv.org/abs/2206.00995

## 7. Decision and next theorem target

Retain the resource law as an accounting theorem and favorable linear control.
Do not implement the bit-flip family as an R12 candidate: it is exactly a
low-rank weighted automaton. Do not claim that a low-dimensional vector beats
the residual information bound.

The remaining admissible target is a **nonlinear learnability separation**:
one small learner must discover and stably execute a short action presentation
from ordinary noisy traces with polynomial resources, while a named
unstructured residual learner requires exponentially more samples, parameters,
or verification work. The claim must include bounded precision and online
identification. Until such a family survives the equivalence and prior-art
gates, no CPU neural falsifier or Shohin fit is authorized.
