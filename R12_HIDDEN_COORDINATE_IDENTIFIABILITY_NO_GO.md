# R12 Hidden-Coordinate Identifiability No-Go

**Status:** ordinary partial observations and opaque interventions cannot
identify locality in a conjugacy-closed model class. A finite positive theorem
exists for atomic resets, but those interventions already carry the coordinate
factorization and collapse to interventional causal representation learning.

## 1. Adaptive conjugacy theorem

Let latent state be `x`, observation kernel `O(y|x)`, event dynamics `F_a`, and
interventions `I_e`. For any bijection, or diffeomorphism in a continuous
model, `phi`, define

```
F_a^phi = phi F_a phi^-1
I_e^phi = phi I_e phi^-1
O^phi(y|z) = O(y|phi^-1(z)).
```

Every adaptive experiment whose next action depends only on earlier observed
outputs and action labels has exactly the same transcript distribution in the
original and conjugated systems.

**Proof.** Couple the transformed latent state as `z_t=phi(x_t)`. The
observation kernels agree by construction. Conditional on an identical
observed transcript, the experiment chooses the same next label; conjugacy
then preserves the coupling at the next step. Induction proves equality for
the full adaptive transcript.

Therefore coordinate factorization, support size, sparsity, and locality are
not identifiable when the model class is closed under arbitrary conjugacy. An
injective observation can reveal an abstract state space but not privileged
product coordinates. A non-injective observation reveals at most the
controlled behavioral or bisimulation quotient.

This is the dynamical counterpart of the impossibility of unsupervised
disentanglement without inductive bias documented by
[Locatello et al.](https://proceedings.mlr.press/v97/locatello19a.html).

## 2. Strongest finite positive theorem

Let

```
X = product_(i=1)^n X_i,  |X_i|=m_i>=2,
```

with an injective arbitrary observation map. Suppose the opaque interventions
are exactly all atomic resets

```
r_(i,v)(x)_i = v,
r_(i,v)(x)_j = x_j  for j != i,
```

but their target coordinates and values are not labeled.

Then the product coordinates are identifiable up to coordinate permutation and
within-coordinate value relabeling:

1. two distinct resets commute exactly when they target different coordinates;
2. the noncommutation graph partitions interventions into coordinate groups;
3. `Fix(r_(i,v))={x:x_i=v}` recovers each coordinate's level sets;
4. every other product realization of the same full reset family preserves
   these groups and partitions.

If `M=sum_i m_i`, noiseless recovery takes `O(M^2)` paired composition probes
plus `O(M)` reset probes to decode one state, assuming state cloning or
reproducible counterfactual starts. With observable separation margin `gamma`,
replicated recovery needs

```
O(M^2 gamma^-2 log(M/delta))
```

intervention sequences. No positive margin means no uniform finite-sample
guarantee.

Once the axes are recovered, the local-rule theorem in
`R12_LOCAL_REVERSIBLE_RULE_CONTROL.md` applies.

## 3. Why the positive theorem is not the invention

Atomic resets are a coordinate oracle in algebraic form:

- their noncommutation relation names coordinate groups;
- their fixed points expose coordinate values;
- atomicity is assumed rather than discovered;
- cloning supplies unusually strong counterfactual supervision;
- a reset-aware causal-representation or program learner receives the same
  advantage.

Primary results already establish latent identification up to permutation and
simple componentwise transformations from structured interventions, including
[Ahuja et al.](https://proceedings.mlr.press/v202/ahuja23a.html) and unknown
intervention pairings in
[Varici et al.](https://proceedings.mlr.press/v238/varici24a.html). Nonlinear
ICA similarly obtains identifiability by adding an auxiliary variable that
breaks the symmetry
([Hyvarinen et al.](https://arxiv.org/abs/1805.08651)).

## 4. Operational boundary

The desired R12 mechanism needs an observable asymmetry. Sparsity or locality
alone cannot select one representative from a conjugacy orbit. But a proposed
asymmetry is invalid if it simply labels the axes, supplies the hidden program,
or reduces to known auxiliary-variable/interventional identification.

A survivor must specify:

1. the symmetry group left by ordinary observations;
2. a task-native statistic that breaks that group without coordinate labels;
3. a finite determining experiment and quantitative separation margin;
4. sample and computational complexity;
5. invariance to irrelevant surface encodings;
6. matched interventional, equivariant, and program-learning controls.

No CPU falsifier is authorized for atomic-reset recovery. It would reproduce a
known identification effect under assumptions that already expose the axes.
