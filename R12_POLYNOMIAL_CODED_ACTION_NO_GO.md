# R12 Polynomial-Coded Action No-Go

**Status:** exact positive control; reject as a novel mechanism or fair
recurrent separation.

## Candidate

Represent each event action as a bounded-degree polynomial over a finite field,
learn it by interpolation, encode the latent state with an error-correcting
code, and apply decode-compute-reencode at every reasoning step. This gives an
exact, nonlinear, length-extrapolating and noise-robust recurrence.

## Identification and robustness theorem

Let each action be

```
U_a : F_q^d -> F_q^d
```

with coordinate degree at most `k < q`. The scalar polynomial space has

```
M = binomial(d+k, k)
```

coefficients. For sampled input states `S_a = (x_1,...,x_m)`, let `V_a` be the
multivariate evaluation matrix and define the evaluation code

```
C_{S_a} = {(p(x_1),...,p(x_m)) : degree(p) <= k}.
```

For every output coordinate:

1. noiseless exact identification is possible exactly when `rank(V_a)=M`;
2. exact recovery from at most `e` adversarially corrupted transition labels is
   possible exactly when `d_min(C_{S_a}) >= 2e+1`;
3. once every generator `U_a` is identified, every finite composition is exact
   by induction.

If `E` is a code correcting `t` physical errors and `D` its decoder, then

```
Phi_a(z) = E(U_a(D(z)))
```

is an exact `t`-robust recurrent action under the stated fault boundary.
Learning and runtime repair are therefore interpolation and decoding problems.

## Smallest nonlinear witnesses

Over `F_3`, `U(x)=x^2` with ternary repetition `E(x)=(x,x,x)` is nonlinear and
corrects one symbol error. Traces confined to `{0,1}` cannot distinguish it from
the identity, exposing the excitation requirement.

For a reversible Boolean witness, Toffoli

```
U(x,y,z) = (x,y,z xor (x and y))
```

is the smallest nonlinear reversible action: every permutation of two Boolean
bits is affine because `AGL(2,2)` already has order `24 = |S_4|`. Combining
Toffoli with a binary `[6,3,3]` code corrects one physical bit error; the Hamming
bound excludes a binary one-error-correcting encoding of three data bits with
length at most five.

## Fatal matched-comparator theorem

Give a universal recurrent comparator the same physical state bits, polynomial
degree promise, samples, and enough update computation to implement `D`,
`U_a`, and `E`. It can execute the identical interpolation and the identical
decode-compute-reencode recurrence. It therefore has the same sample complexity,
length extrapolation, and robustness.

The apparent description gap

```
binomial(d+k,k)  versus  q^d
```

is only a low-degree learner versus an arbitrary transition table. It vanishes
as an architecture-wide separation once the comparator receives the same
target promise. Removing the promise reintroduces ordinary finite-system
nonidentifiability: any finite transition map has a finite-field polynomial
representation, and off-support patches survive.

## Collapse audit

- bounded-degree interpolation is classical polynomial learning/coding;
- state repair is error-correcting or fault-tolerant computation;
- bijective maps are permutation-group actions and arbitrary maps form a
  transformation semigroup;
- unknown coordinate changes destroy the presentation-dependent degree and
  locality unless a field basis is externally anchored;
- unknown noisy linear structure already contains parity with noise;
- explicitly supplied arithmetic operators are an arithmetic inductive bias,
  not discovered reasoning.

This family remains an excellent exact control for any future proposal claiming
nonlinear extrapolation plus runtime noise stability. It is not a novel R12
primitive and does not justify a CPU falsifier, Shohin fit, or H100 experiment.

## Reopening condition

Reconsider only if a task-native observable identifies the field basis and code
from ordinary traces, and the proposed resource restriction excludes simulation
by a fairly matched universal recurrent circuit without simply denying it the
same computation.

Primary references:

- Z. Dvir and A. Shpilka, noisy interpolation sets and punctured Reed-Muller
  constructions.
- D. Spielman, reliable computation with efficient error-correcting codes.
