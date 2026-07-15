# R12 Closed Late-Query Information No-Go

**Status:** proved lower-bound control; no mechanism result.

**Implementation authority:** none. This document authorizes no data, code,
fit, score, CPU board, or GPU job.

## 1. Closed late-query protocol

For `n >= 2`, let

```
X_n = {0,1}^n,
Q_n = {1,...,n},
F_n(x,i) = x_i.
```

A fixed finite description, independent of `n`, reads `x` and commits to a
retained state `S` before learning `i`. The source then becomes inaccessible.
An adversary reveals `i`, and the mechanism must answer `x_i`.

Every input-dependent context token, activation, transcript, certificate,
scratch symbol, cache entry, or external byte still accessible after commitment
is part of `S`. Searchable old context is retained memory; an outside prover or
source is an external information channel.

## 2. Closed-configuration theorem

Let a complete post-commit configuration evolve by one fixed rule

```
C_(t+1) = tau_D(C_t, z_t, R_t),
```

where `z_t` is the newly supplied symbol and `R_t` is independent randomness.
Let `M_A(n)=ceil(log2 |C_n|)` be retained-state capacity. Define a residual row

```
rho_n(x) = (F_n(x,q))_(q in Q_n),
N_n = |{rho_n(x) : x in X_n}|.
```

Then:

1. the mechanism is already a closed uniform transducer when its complete
   configuration is taken as state;
2. exact correctness requires

   ```
   M_A(n) >= ceil(log2 N_n);
   ```

3. if `Z` is any internally generated post-commit transcript, then

   ```
   X -> (S,R) -> Z
   I(X;S,Z | R) = I(X;S | R).
   ```

Post-commit time can transform retained information but cannot recreate source
information that is absent from the state.

### Proof

The complete configuration and fixed update rule are a transducer definition.
If two inputs yield the same retained state, every subsequent computation sees
the same state, query, and coin distribution. Exact correctness therefore
requires the two inputs to have identical residual rows. Distinct residual rows
need distinct retained states. The information identity follows from the Markov
chain and data processing.

## 3. Late-INDEX lower bounds

For late INDEX, `rho_n(x)=x`, so `N_n=2^n` and exact correctness requires at
least `n` retained bits.

Let `X` be uniform and suppose every bit decoder has error at most
`epsilon < 1/2`. If `p_i` is the error on bit `i`, then

```
M_A(n)
  >= I(X;S | R)
  = n - H(X | S,R)
  >= n - sum_i h2(p_i)
  >= n (1 - h2(epsilon)).
```

Thus constant-error randomization still requires linear retained information.
If `A_i` is the event that bit `i` was inspected while the source was available,
uniform independent inputs also give

```
E[input probes] >= sum_i Pr(A_i) >= n (1 - 2 epsilon).
```

A uniform upper bound simply stores the bit array and later indexes it, using
`n+O(log n)` bits and `Theta(n)` input work. The lower bounds are tight up to
addressing overhead.

The randomized one-way INDEX lower bound is established communication-complexity
machinery; see the self-contained proof in [The One-Way Communication
Complexity of Hamming
Distance](https://theoryofcomputing.org/articles/v004a006/). The entropy form is
the classical random-access-code converse; related optimal bounds appear in
[Optimal Lower Bounds for Quantum Automata and Random Access
Codes](https://arxiv.org/abs/quant-ph/9904093).

## 4. Collapse audit

- A certified invariant answering every late query must separate every residual
  row; it cannot use fewer exact states.
- A randomized sketch is a one-way message and inherits the INDEX bound.
- Internal interactive proof messages are generated from `S` and add no source
  information. An external or input-supplied prover changes the resource model.
- Recurrence and search are state transitions of the same closed transducer.
  Collided states generate identical search distributions.
- Rate-distortion is not an escape: the decoders define a reconstructed bit
  vector, and the entropy bound is exactly its Hamming-distortion converse.
- Arbitrary-precision reals, uncounted caches, hidden source access, retrieval,
  and nonuniform advice are excluded resources, not compressed reasoning.

## 5. Smallest witness

At `n=2`, the four inputs `00,01,10,11` have four distinct rows across queries
one and two. An exact closed mechanism with at most three distinguishable
post-commit states would falsify the theorem. The residual table proves that no
such mechanism exists.

## 6. Consequence for Shohin

Longer internal computation alone cannot make arbitrary discarded context
recoverable. A first-of-kind R12 result must exploit a **structured** problem
family whose residuals have concise, computable sufficient structure. Even
then, the contribution cannot be a new exact state ontology: the exact state is
still the residual quotient.

The only remaining plausible theorem targets are resource advantages in
learnability, dynamic sparsity, amortized verification, noise stability, or
another explicitly named cost. Any proposal claiming sublinear memory for
arbitrary adversarial late queries is rejected before implementation.
